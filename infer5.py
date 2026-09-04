# inference.py  —  Full pipeline: Stage 1 (MiDaS) + Stage 2 (PCM)
# Usage:
#   python inference.py --image path/to/any/image.jpg
#   python inference.py --image path/to/any/image.jpg --save output.png

import os, sys, argparse
import numpy as np
import torch
import cv2
import pyvista as pv

from pcm_utils import shift_combined, shift_near_camera_edges
from config import PROJECT_ROOT, CHECKPOINT, OUTPUT_DIR
                    

# ── paths ─────────────────────────────────────────────────────────────────────
PVCNN_ROOT   = PROJECT_ROOT
STAGE2_CKPT  = CHECKPOINT
sys.path.insert(0, PVCNN_ROOT)

# ── constants ─────────────────────────────────────────────────────────────────
INIT_FOV_DEG = 60.0
PCM_POINTS   = 8192
VOXEL_RES    = 32
MIDAS_MODEL  = "DPT_Large"


# ── Stage 1: MiDaS ───────────────────────────────────────────────────────────
def load_stage1(device):
    hub_dir = os.path.expanduser("~/.cache/torch/hub/intel-isl_MiDaS_master")
    try:
        midas      = torch.hub.load(hub_dir, MIDAS_MODEL, source="local")
        transforms = torch.hub.load(hub_dir, "transforms", source="local")
    except Exception as e:
        print(f"Local load failed, falling back to online: {e}")
        midas      = torch.hub.load("intel-isl/MiDaS", MIDAS_MODEL, trust_repo=True)
        transforms = torch.hub.load("intel-isl/MiDaS", "transforms", trust_repo=True)
    transform = transforms.dpt_transform
    return midas.eval().to(device), transform

# ── image loading ─────────────────────────────────────────────────────────────
def load_image(path):
    bgr = cv2.imread(path)
    if bgr is None or bgr.size == 0:
        try:
            from PIL import Image
            rgb = np.array(Image.open(path).convert("RGB"))
            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        except Exception as e:
            raise FileNotFoundError(f"Cannot open '{path}'. Error: {e}")
    rgb_orig = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return bgr, rgb_orig

def predict_depth(img_bgr, model, transform, device):

    img_rgb     = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    input_batch = transform(img_rgb).to(device)

    with torch.no_grad():
        pred = model(input_batch)
        pred = torch.nn.functional.interpolate(
            pred.unsqueeze(1),
            size=img_rgb.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze()

    d = pred.cpu().numpy().astype(np.float32)

    """
    MiDaS outputs disparity (larger value = closer to camera).
    We invert to get true depth ordering (larger value = farther away),
    which is required for correct pinhole unprojection.
    The inverstion : d = 1 - d  is numerically stable unlike 1/d which explodes near zero.
    """
    # Step 1: normalise raw MiDaS output to [0, 1]
    lo, hi = d.min(), d.max()
    d = (d - lo) / (hi - lo + 1e-8)

    # Step 2: invert disparity → depth
    # Before: bright pixel = close object (disparity)
    # After:  bright pixel = far object (depth)
    d = 1.0 - d

    # Step 3: re-normalise so output is always [0, 1]
    lo, hi = d.min(), d.max()
    d = (d - lo) / (hi - lo + 1e-8)
    # Step 4: resize to 640x640 for PCM input (Upsampling)
    d_hires = cv2.resize(d, (640, 640), interpolation=cv2.INTER_LINEAR)
    return d.astype(np.float32), d_hires.astype(np.float32)

# =========================================================================================


# ── Stage 2: PCM ─────────────────────────────────────────────────────────────
def load_stage2(device, ckpt_path=None):
    from models.pvcnn import ShiftPVCNN, FocalPVCNN
    path = ckpt_path or STAGE2_CKPT
    print(f"  PCM checkpoint: {path}")
    ckpt      = torch.load(path, map_location="cpu", weights_only=True)
    shift_net = ShiftPVCNN(VOXEL_RES)
    focal_net = FocalPVCNN(VOXEL_RES)
    shift_net.load_state_dict(ckpt["shift_net"])
    focal_net.load_state_dict(ckpt["focal_net"])
    return shift_net.eval().to(device), focal_net.eval().to(device)

def depth_to_pc_rgb_full(depth_np, rgb_np, fx, fy, cx, cy):
    H, W   = depth_np.shape
    rgb    = cv2.resize(rgb_np, (W, H)).astype(np.float32) / 255.0
    valid  = depth_np > 0
    us, vs = np.meshgrid(np.arange(W), np.arange(H))
    x =  (us[valid] - cx) / fx * depth_np[valid]
    y = -(vs[valid] - cy) / fy * depth_np[valid]
    z = -depth_np[valid]
    return (np.stack([x, y, z], axis=1).astype(np.float32),
            rgb[valid].astype(np.float32))

def depth_to_pc_sampled(depth_np, fx, fy, cx, cy, n=PCM_POINTS):
    H, W   = depth_np.shape
    valid  = depth_np > 0
    us, vs = np.meshgrid(np.arange(W), np.arange(H))
    x =  (us[valid] - cx) / fx * depth_np[valid]
    y = -(vs[valid] - cy) / fy * depth_np[valid]
    z = -depth_np[valid]
    pts = np.stack([x, y, z], axis=1).astype(np.float32)
    idx = np.random.choice(len(pts), n, replace=(len(pts) < n))
    return pts[idx]


def run_pcm(depth_norm, depth_hires, rgb_orig, shift_net, focal_net, device):
    H, W   = depth_norm.shape
    f_init = (W / 2.0) / np.tan(np.deg2rad(INIT_FOV_DEG / 2.0))
    cx, cy = W / 2.0, H / 2.0

    pts_pcm = depth_to_pc_sampled(depth_norm, f_init, f_init, cx, cy)
    pc_t    = torch.from_numpy(pts_pcm).unsqueeze(0).permute(0, 2, 1).to(device)

    with torch.no_grad():
        delta_d     = shift_net(pc_t).item()
        alpha_f_raw = focal_net(pc_t).item()

    # Focal network is unreliable (collapses to ~0.52) — clamp tightly
    alpha_f = float(np.clip(alpha_f_raw, 0.85, 1.15))
    delta_d = float(np.clip(delta_d, -0.2, 0.6))

    print(f"  Raw:     Δd={delta_d:+.4f}  αf={alpha_f_raw:.4f}")
    print(f"  Clamped: Δd={delta_d:+.4f}  αf={alpha_f:.4f}")

     # ── Apply combined depth+planarity weighted shift ─────────────────────────
    d_corr       = shift_combined(depth_norm,  delta_d,
                                  steepness=8.0, midpoint=0.4,
                                  edge_sensitivity=0.02,
                                  depth_weight=0.6, plane_weight=0.4)
    d_hires_corr = shift_combined(depth_hires, delta_d,
                                  steepness=8.0, midpoint=0.4,
                                  edge_sensitivity=0.02,
                                  depth_weight=0.6, plane_weight=0.4)
    
    # Pass 2: smooth near-camera edge correction
    d_corr       = shift_near_camera_edges(d_corr,       delta_d,
                                           edge_strength=1.5,
                                           depth_threshold=0.35)
    d_hires_corr = shift_near_camera_edges(d_hires_corr, delta_d,
                                           edge_strength=1.5,
                                           depth_threshold=0.35)

    f_corr = f_init * alpha_f

    Hh, Wh   = depth_hires.shape
    fh       = (Wh / 2.0) / np.tan(np.deg2rad(INIT_FOV_DEG / 2.0))
    cxh, cyh = Wh / 2.0, Hh / 2.0
    fh_corr  = fh * alpha_f

    pc_raw,  col_raw  = depth_to_pc_rgb_full(depth_hires,  rgb_orig,
                                              fh,      fh,      cxh, cyh)
    pc_corr, col_corr = depth_to_pc_rgb_full(d_hires_corr, rgb_orig,
                                              fh_corr, fh_corr, cxh, cyh)

    return pc_raw, col_raw, pc_corr, col_corr, delta_d, alpha_f, f_corr

def save_ply_for_viewing(pts, cols, path):
    """Save colored PLY — open in MeshLab, CloudCompare, or Windows 3D Viewer."""
    import struct
    cols_u8 = (cols * 255).astype(np.uint8)
    with open(path, 'w') as f:
        f.write(f"ply\nformat binary_little_endian 1.0\n"
                f"element vertex {len(pts)}\n"
                f"property float x\nproperty float y\nproperty float z\n"
                f"property uchar red\nproperty uchar green\nproperty uchar blue\n"
                f"end_header\n")
    with open(path, 'ab') as f:
        for i in range(len(pts)):
            f.write(struct.pack('fffBBB',
                pts[i,0], pts[i,1], pts[i,2],
                cols_u8[i,0], cols_u8[i,1], cols_u8[i,2]))
    print(f"  PLY saved: {path}")


# ── depth colormap ────────────────────────────────────────────────────────────
def depth_to_rgba(depth_np, size=None):
    import matplotlib
    d    = depth_np.copy()
    d    = (d - d.min()) / (d.max() - d.min() + 1e-8)
    rgba = (matplotlib.colormaps["plasma"](d) * 255).astype(np.uint8)
    if size:
        rgba = cv2.resize(rgba, size, interpolation=cv2.INTER_LINEAR)
    return rgba
# ── visualisation ─────────────────────────────────────────────────────────────
def visualize(rgb_orig, depth_np,
              pc_raw, col_raw, pc_corr, col_corr,
              delta_d, alpha_f, image_path, save_path=None):

    disp     = (640, 480)
    rgb_r    = cv2.resize(rgb_orig, disp)
    rgb_rgba = np.dstack([rgb_r, np.full((*rgb_r.shape[:2], 1), 255, np.uint8)])
    d_rgba   = depth_to_rgba(depth_np, size=disp)

    tex_rgb   = pv.Texture(rgb_rgba)
    tex_depth = pv.Texture(d_rgba)
    plane     = lambda: pv.Plane(center=(0, 0, 0), direction=(0, 0, 1),
                                 i_size=1.28, j_size=0.96,
                                 i_resolution=1, j_resolution=1)

    cloud_raw         = pv.PolyData(pc_raw)
    cloud_corr        = pv.PolyData(pc_corr)
    cloud_raw["rgb"]  = (col_raw  * 255).astype(np.uint8)
    cloud_corr["rgb"] = (col_corr * 255).astype(np.uint8)

    print(f"  Point cloud size: raw={len(pc_raw):,}  corrected={len(pc_corr):,}")

    pv.global_theme.background = "black"
    pv.global_theme.font.color = "white"
    title = (f"{os.path.basename(image_path)}  |  "
             f"Δd={delta_d:+.4f}   αf={alpha_f:.4f}")

    pl = pv.Plotter(shape=(1, 4), border=True, border_color="gray",
                    window_size=(1600, 500), title=title)

    pl.subplot(0, 0)
    pl.add_mesh(plane(), texture=tex_rgb, show_edges=False, lighting=False)
    pl.add_text("RGB Image", position="upper_left", font_size=9, color="white")
    pl.view_xy()
    pl.camera.zoom(1.15)

    pl.subplot(0, 1)
    pl.add_mesh(plane(), texture=tex_depth, show_edges=False, lighting=False)
    pl.add_text("Predicted Depth (MiDaS DPT_Large)",
                position="upper_left", font_size=9, color="white")
    pl.view_xy()
    pl.camera.zoom(1.15)

    pl.subplot(0, 2)
    pl.add_points(cloud_raw, scalars="rgb", rgb=True,
                  point_size=1.5, render_points_as_spheres=False)
    pl.add_text("Raw Point Cloud\n(before PCM)",
                position="upper_left", font_size=9, color="white")
    pl.add_text("Keys: F=Front  T=Top  L=Left  R=Right  I=Isometric",
                position="lower_left", font_size=7, color="yellow")
    pl.add_axes(line_width=3)
    pl.reset_camera()    # auto-fit camera to actual coordinate range of this cloud
    pl.view_isometric()

    pl.subplot(0, 3)
    pl.add_points(cloud_corr, scalars="rgb", rgb=True,
                  point_size=1.5, render_points_as_spheres=False)
    pl.add_text("PCM Corrected\n(Stage 1 + Stage 2)",
                position="upper_left", font_size=9, color="white")
    pl.add_text("Keys: F=Front  T=Top  L=Left  R=Right  I=Isometric",
                position="lower_left", font_size=7, color="yellow")
    pl.add_axes(line_width=3)
    pl.reset_camera()    # auto-fit camera to actual coordinate range of this cloud
    pl.view_isometric()

    pl.link_views(views=[2, 3])

    def set_view(view_fn):
        for idx in [2, 3]:
            pl.subplot(0, idx)
            view_fn()
        pl.render()

    pl.add_key_event("f", lambda: set_view(pl.view_yz))
    pl.add_key_event("t", lambda: set_view(pl.view_xy))
    pl.add_key_event("l", lambda: set_view(pl.view_xz))
    pl.add_key_event("i", lambda: set_view(pl.view_isometric))

    print("\n[Visualization]")
    print("  Panel 1: RGB input")
    print("  Panel 2: MiDaS DPT_Large depth map")
    print("  Panel 3: Raw RGB point cloud")
    print("  Panel 4: PCM-corrected RGB point cloud")
    print("  Panels 3+4 rotate together.")
    print("  Close window to exit.\n")

    if save_path:
        pl.show(auto_close=False)
        pl.screenshot(save_path)
        pl.close()
        print(f"  Saved -> {save_path}")
    else:
        pl.show(auto_close=False)
        pl.close()


# ── main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image",      required=True)
    parser.add_argument("--save",       default="")
    parser.add_argument("--checkpoint", default=STAGE2_CKPT,
                        help="Path to PCM checkpoint .pth file")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device : {device}")
    print(f"Image  : {args.image}")

    # ------- Stage 1 ---------------------------------------
    bgr, rgb_orig = load_image(args.image)

    print(f"Stage 1: MiDaS {MIDAS_MODEL} ...")
    midas, transform = load_stage1(device)
    depth_norm, depth_hires = predict_depth(bgr, midas, transform, device)
    print(f"  depth={depth_norm.shape}  hires={depth_hires.shape}")

    # ------- Stage 2 ---------------------------------------
    print("Stage 2: PCM correction ...")
    shift_net, focal_net = load_stage2(device, args.checkpoint)
    pc_raw, col_raw, pc_corr, col_corr, delta_d, alpha_f, f_corr = \
        run_pcm(depth_norm, depth_hires, rgb_orig, shift_net, focal_net, device)
    print(f"  Δd={delta_d:+.4f}  αf={alpha_f:.4f}  f_corr={f_corr:.1f}px")

    input_dir  = os.path.dirname(args.image)
    base_name  = os.path.splitext(os.path.basename(args.image))[0]

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    depth_png_path = OUTPUT_DIR / f"{base_name}_depth.png"
    raw_ply_path = OUTPUT_DIR / f"{base_name}_raw.ply"
    corrected_ply_path = OUTPUT_DIR / f"{base_name}_corrected.ply"

    # Generate colored depth (RGBA)
    depth_rgba = depth_to_rgba(depth_hires)

    # OpenCV uses BGR/BGRA, so we swap channels before saving
    depth_bgra = depth_rgba[..., [2, 1, 0, 3]]
    cv2.imwrite(depth_png_path, depth_bgra)
    print(f"  Depth map saved: {depth_png_path}")

    save_ply_for_viewing(pc_raw, col_raw, raw_ply_path)
    save_ply_for_viewing(pc_corr, col_corr, corrected_ply_path)
      

    visualize(rgb_orig, depth_norm,
              pc_raw, col_raw, pc_corr, col_corr,
              delta_d, alpha_f, args.image,
              save_path=args.save if args.save else None)

if __name__ == "__main__":
    main()
