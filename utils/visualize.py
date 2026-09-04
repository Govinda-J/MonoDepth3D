# visualize.py
# Local ScanNet evaluation tool: loads a GT depth frame, runs PCM correction,
# and shows a 4-panel PyVista comparison. Requires local ScanNet data + GUI —
# not used for deployment.
#
# Usage:
#   python utils/visualize.py
#   python utils/visualize.py --scene scene0035_00
#   python utils/visualize.py --stem  scene0035_00_frame0487
#
# Keyboard (in point-cloud panels): F=Front T=Top L=Left R=Right I=Isometric

import os, sys, glob, random, argparse
import numpy as np
import torch
import cv2
import matplotlib
import pyvista as pv
from config import PROJECT_ROOT, CHECKPOINT

# ── paths ──────────────────────────────────────────────────────────────────────
DEPTH_DIR  = r"C:\scannet_data\depth_pt"
COLOR_DIR  = r"C:\scannet_data\color_jpg"
PVCNN_ROOT = PROJECT_ROOT

# ── ScanNet metric intrinsics ──────────────────────────────────────────────────
FX, FY = 577.590698, 578.729797
CX, CY = 318.905426, 242.683609

INIT_FOV   = 60.0
NUM_POINTS = 8192
VOXEL_RES  = 32


# ── helpers ────────────────────────────────────────────────────────────────────

def fov_to_focal(fov_deg, width):
    return (width / 2.0) / np.tan(np.deg2rad(fov_deg / 2.0))


def depth_to_pc_rgb(depth_np, color_rgb_hw, fx, fy, cx, cy, max_points=None):
    """
    Unproject every valid pixel → XYZ + RGB.
    Returns (N,3) float32 xyz,  (N,3) float32 rgb in [0,1].
    If max_points given, randomly subsample.
    """
    H, W  = depth_np.shape
    valid = depth_np > 0
    if valid.sum() == 0:
        n = max_points or 1
        return np.zeros((n,3), np.float32), np.ones((n,3), np.float32)*0.5

    d  = depth_np.astype(np.float32)
    us = np.tile(np.arange(W), (H, 1))
    vs = np.tile(np.arange(H), (W, 1)).T

    x = (us - cx) / fx * d
    y = (vs - cy) / fy * d
    z = d

    xyz = np.stack([x[valid], y[valid], z[valid]], axis=1).astype(np.float32)
    rgb = color_rgb_hw[valid].astype(np.float32) / 255.0

    if max_points and len(xyz) > max_points:
        idx = np.random.choice(len(xyz), max_points, replace=False)
        xyz, rgb = xyz[idx], rgb[idx]

    return xyz, rgb


def load_model():
    sys.path.insert(0, PVCNN_ROOT)
    from models.pvcnn import ShiftPVCNN, FocalPVCNN

    ckpt      = torch.load(CHECKPOINT, map_location="cpu", weights_only=True)
    shift_net = ShiftPVCNN(VOXEL_RES)
    focal_net = FocalPVCNN(VOXEL_RES)
    shift_net.load_state_dict(ckpt["shift_net"], strict=False)
    focal_net.load_state_dict(ckpt["focal_net"], strict=False)
    shift_net.eval(); focal_net.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return shift_net.to(device), focal_net.to(device), device


def run_pcm(depth_np, shift_net, focal_net, device):
    """Normalise depth → run PCM → return corrected depth, corrected focal, Δd, αf."""
    H, W   = depth_np.shape
    f_init = fov_to_focal(INIT_FOV, W)
    cx, cy = W / 2.0, H / 2.0

    valid  = depth_np > 0
    d_norm = np.zeros_like(depth_np, dtype=np.float32)
    if valid.any():
        lo, hi = depth_np[valid].min(), depth_np[valid].max()
        d_norm[valid] = (depth_np[valid] - lo) / (hi - lo + 1e-8)

    # Build PCM input cloud from normalised depth
    d  = d_norm.copy()
    us = np.tile(np.arange(W), (H, 1))
    vs = np.tile(np.arange(H), (W, 1)).T
    v  = d_norm > 0
    pts = np.stack(
        [(us[v] - cx) / f_init * d[v],
         (vs[v] - cy) / f_init * d[v],
          d[v]], axis=1).astype(np.float32)

    # ---- SHIFT NETWORK INPUT: z-centered to match training contract ----
    # ShiftPVCNN trained on z-centered clouds (see distortions.py).
    #  At inference depth_gt.mean() == d_norm[valid].mean(), so we replicate it.
    # FocalPVCNN does NOT get this centering — its training had none.
    z_ref = float(d_norm[v].mean())
    pts[:, 2] = pts[:, 2] - z_ref
    # ──────────────────────────────────────────────────────────────────────────

    n   = len(pts)
    idx = np.random.choice(n, NUM_POINTS, replace=(n < NUM_POINTS))
    pc_shift_t = torch.from_numpy(pts[idx]).unsqueeze(0).permute(0, 2, 1).to(device)

    # FocalPVCNN gets the un-centered cloud (z = d_norm, no subtraction)
    pts_focal = np.stack(
        [(us[v] - cx) / f_init * d[v],
         (vs[v] - cy) / f_init * d[v],
          d[v]], axis=1).astype(np.float32)
    idx_f = np.random.choice(n, NUM_POINTS, replace=(n < NUM_POINTS))
    pc_focal_t = torch.from_numpy(pts_focal[idx_f]).unsqueeze(0).permute(0, 2, 1).to(device)

    with torch.no_grad():
        delta_d = shift_net(pc_shift_t).item()
        alpha_f = focal_net(pc_focal_t).item()

    d_corr = d_norm.copy()
    d_corr[valid] += delta_d
    f_corr = f_init * alpha_f

    return d_norm, d_corr, f_corr, delta_d, alpha_f


def make_depth_rgba(depth_np, cmap="plasma"):
    valid = depth_np > 0
    d = depth_np.copy().astype(np.float32)
    if valid.any():
        lo, hi = d[valid].min(), d[valid].max()
        d = (d - lo) / (hi - lo + 1e-8)
    rgba = (matplotlib.colormaps[cmap](d) * 255).astype(np.uint8)
    return cv2.resize(rgba, (640, 480), interpolation=cv2.INTER_LINEAR)


def make_pv_cloud(xyz, rgb_f):
    """xyz: (N,3) float32,  rgb_f: (N,3) float32 [0,1] → pv.PolyData with RGB."""
    cloud = pv.PolyData(xyz)
    # PyVista expects uint8 RGB as (N,3)
    cloud.point_data["RGB"] = (rgb_f * 255).clip(0, 255).astype(np.uint8)
    return cloud


# ── KEYBOARD VIEW CONTROLS ───────────────────────────────────────────────────
def make_key_handler(plotter, panels):
    """Returns a key-press callback that sets view on all linked 3D panels."""
    views = {
        "f": lambda p: p.view_yz(),        # Front
        "t": lambda p: p.view_xz(),        # Top
        "l": lambda p: p.view_xy(),        # Left
        "r": lambda p: (p.view_xy(), p.camera.azimuth(180)),  # Right
        "i": lambda p: p.view_isometric(), # Isometric
    }
    def handler(key):
        key = key.lower()
        if key in views:
            for idx in panels:
                plotter.subplot(0, idx)
                views[key](plotter)
            plotter.render()
    return handler


# ── MAIN VISUALIZATION ─────────────────────────────────────────────────────────────
def visualize(stem):
    depth_path = os.path.join(DEPTH_DIR, f"{stem}.pt")
    color_path  = os.path.join(COLOR_DIR,  f"{stem}.jpg")
    if not os.path.exists(depth_path): raise FileNotFoundError(depth_path)
    if not os.path.exists(color_path):  raise FileNotFoundError(color_path)

    # load
    depth_np   = torch.load(depth_path, weights_only=True).numpy()
    H_d, W_d   = depth_np.shape
    color_bgr  = cv2.imread(color_path)
    color_rgb  = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2RGB)
    color_rgb_d = cv2.resize(color_rgb, (W_d, H_d), interpolation=cv2.INTER_LINEAR)

    # PCM
    print(f"Device : {'cuda' if torch.cuda.is_available() else 'cpu'}")
    print(f"Frame  : {stem}")
    print("Stage 2: PCM correction ...")
    shift_net, focal_net, device = load_model()
    d_norm, d_corr, f_corr, delta_d, alpha_f = run_pcm(
        depth_np, shift_net, focal_net, device)
    print(f"  Δd={delta_d:+.4f}  αf={alpha_f:.4f}  f_corr={f_corr:.1f}px")

    # raw cloud — normalised depth, init FOV intrinsics (same as before PCM)
    f_init = fov_to_focal(INIT_FOV, W_d)
    cx_d, cy_d = W_d / 2.0, H_d / 2.0
    raw_xyz,  raw_rgb  = depth_to_pc_rgb(d_norm,  color_rgb_d,
                                          f_init, f_init, cx_d, cy_d)
    corr_xyz, corr_rgb = depth_to_pc_rgb(d_corr, color_rgb_d,
                                          f_corr, f_corr, cx_d, cy_d)
    print(f"  Point cloud size: raw={len(raw_xyz):,}  corrected={len(corr_xyz):,}")

    # images
    color_display = cv2.resize(color_rgb, (640, 480))
    color_rgba    = np.dstack([color_display, np.full((480,640,1), 255, np.uint8)])
    depth_rgba    = make_depth_rgba(depth_np, cmap="plasma")

    tex_color = pv.Texture(color_rgba)
    tex_depth = pv.Texture(depth_rgba)
    plane_c   = pv.Plane(center=(0,0,0), direction=(0,0,1),
                         i_size=1.28, j_size=0.96, i_resolution=1, j_resolution=1)
    plane_d   = pv.Plane(center=(0,0,0), direction=(0,0,1),
                         i_size=1.28, j_size=0.96, i_resolution=1, j_resolution=1)

    raw_cloud  = make_pv_cloud(raw_xyz,  raw_rgb)
    corr_cloud = make_pv_cloud(corr_xyz, corr_rgb)

    # ── plotter ────────────────────────────────────────────────────────────────
    pv.global_theme.background = "black"
    pv.global_theme.font.color = "white"

    title = (f"{stem}.jpg  |  Δd={delta_d:+.4f}  αf={alpha_f:.4f}")
    pl = pv.Plotter(
        shape=(1, 4), border=True, border_color="dimgray",
        window_size=(1800, 520), title=title,
    )

    # panel 0 — RGB
    pl.subplot(0, 0)
    pl.add_mesh(plane_c, texture=tex_color, show_edges=False, lighting=False)
    pl.add_text("RGB Image", position="upper_left", font_size=9, color="white")
    pl.view_xy(); pl.camera.zoom(1.1); pl.disable()

    # panel 1 — depth (plasma)
    pl.subplot(0, 1)
    pl.add_mesh(plane_d, texture=tex_depth, show_edges=False, lighting=False)
    pl.add_text("Predicted Depth (Stage 1)",
                position="upper_left", font_size=9, color="white")
    pl.view_xy(); pl.camera.zoom(1.1); pl.disable()

    # panel 2 — raw point cloud (before PCM)
    pl.subplot(0, 2)
    pl.add_points(raw_cloud, scalars="RGB", rgb=True,
                  point_size=1.5, render_points_as_spheres=False)
    pl.add_text("Raw Point Cloud\n(before PCM)",
                position="upper_left", font_size=9, color="white")
    pl.view_isometric()
    pl.enable_trackball_style()
    pl.add_axes()

    # panel 3 — PCM corrected
    pl.subplot(0, 3)
    pl.add_points(corr_cloud, scalars="RGB", rgb=True,
                  point_size=1.5, render_points_as_spheres=False)
    pl.add_text("PCM Corrected\n(Stage 1 + Stage 2)",
                position="upper_left", font_size=9, color="white")
    pl.view_isometric()
    pl.enable_trackball_style()
    pl.add_axes()

    # link panels 2 & 3
    pl.link_views(views=[2, 3])

    # keyboard shortcuts
    key_cb = make_key_handler(pl, panels=[2, 3])
    pl.add_key_event("f", lambda: key_cb("f"))
    pl.add_key_event("t", lambda: key_cb("t"))
    pl.add_key_event("l", lambda: key_cb("l"))
    pl.add_key_event("r", lambda: key_cb("r"))
    pl.add_key_event("i", lambda: key_cb("i"))

    hint = "Keys: F=Front  T=Top  L=Left  R=Right  I=Isometric"
    pl.subplot(0, 2); pl.add_text(hint, position="lower_left", font_size=7, color="yellow")
    pl.subplot(0, 3); pl.add_text(hint, position="lower_left", font_size=7, color="yellow")

    print("\n[Visualization]")
    print("  Panel 1: RGB input")
    print("  Panel 2: Depth map (plasma)")
    print("  Panel 3: Raw RGB point cloud")
    print("  Panel 4: PCM-corrected RGB point cloud")
    print("  Panels 3+4 rotate together.")
    print("  Keyboard shortcuts (click a panel first):")
    print("    F = Front view")
    print("    T = Top view")
    print("    L = Left view")
    print("    R = Right view")
    print("    I = Isometric view")
    print("  Close window to exit.")

    pl.show(auto_close=False)
    pl.close()


def pick_stem(args):
    if args.stem: return args.stem
    all_pts = glob.glob(os.path.join(DEPTH_DIR, "*.pt"))
    if not all_pts: raise RuntimeError(f"No .pt files in {DEPTH_DIR}")
    if args.scene:
        all_pts = [p for p in all_pts if os.path.basename(p).startswith(args.scene)]
        if not all_pts: raise RuntimeError(f"No frames for scene '{args.scene}'")
    return os.path.splitext(os.path.basename(random.choice(all_pts)))[0]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--stem",  default="")
    parser.add_argument("--scene", default="")
    args = parser.parse_args()
    stem = pick_stem(args)
    print(f"Visualising: {stem}\n")
    visualize(stem)