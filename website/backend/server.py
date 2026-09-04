"""
server.py  —  DepthCloud FastAPI backend  (v2 — sends depth map to frontend)
Place at:  website/backend/server.py

Run locally:
  cd website/backend
  uvicorn server:app --reload --port 8000

Run in Docker (see website/backend/Dockerfile):
  uvicorn website.backend.server:app --host 0.0.0.0 --port 7860
"""

import io, os, sys, time
import cv2
import numpy as np
import torch
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from huggingface_hub import hf_hub_download
from config import PROJECT_ROOT, CHECKPOINT, PVCNN_MODEL_REPO

# ── project root ───
sys.path.insert(0, str(PROJECT_ROOT))

from models.pvcnn import ShiftPVCNN, FocalPVCNN

# ── checkpoint path — overridable via env var ──────────────────────────────
# Matches the same PCM_CHECKPOINT variable used by infer5.py / visualize.py,
# so all three entry points agree on where the weights live without needing
# three different env vars. Falls back to the repo-relative default if unset.
local_checkpoint = CHECKPOINT

INIT_FOV_DEG = float(os.environ.get("PCM_INIT_FOV_DEG", 60.0))
PCM_POINTS   = int(os.environ.get("PCM_NUM_POINTS", 8192))
VOXEL_RES    = int(os.environ.get("PCM_VOXEL_RES", 32))

# ── MiDaS cache dir — derived from torch's own hub directory instead of a
# hand-typed "~/.cache/torch/hub/..." guess. torch.hub.get_dir() already
# respects the TORCH_HOME env var (set in the Dockerfile), so this line
# stays correct even if that env var changes later. ─────────────────────────
MIDAS_DIR = os.path.join(torch.hub.get_dir(), "intel-isl_MiDaS_master")

app = FastAPI(title="DepthCloud API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)

DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_shift_net = None
_focal_net = None
_midas     = None
_transform = None


@app.on_event("startup")
def load_pcm():
    global _shift_net, _focal_net
    
    '''if local_checkpoint and os.path.exists(local_checkpoint):
        print(f"Loading PVCNN checkpoint: {CHECKPOINT}")
        checkpoint_path = local_checkpoint'''
    
    print(f"Loading PVCNN checkpoint from HF...")
    checkpoint_path = hf_hub_download(
        repo_id=PVCNN_MODEL_REPO,
        filename="best_checkpoint.pth",
    )


    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    _shift_net = ShiftPVCNN(VOXEL_RES)
    _focal_net = FocalPVCNN(VOXEL_RES)
    _shift_net.load_state_dict(ckpt["shift_net"])
    _focal_net.load_state_dict(ckpt["focal_net"])
    _shift_net.eval().to(DEVICE)
    _focal_net.eval().to(DEVICE)
    print(f"PCM ready on {DEVICE}")


def _load_midas():
    global _midas, _transform
    if _midas is not None:
        return
    print("Loading MiDaS...")
    try:
        m  = torch.hub.load(MIDAS_DIR, "DPT_Large", source="local")
        tr = torch.hub.load(MIDAS_DIR, "transforms", source="local")
    except Exception as e:
        print(f"Local cache miss ({e}), downloading...")
        m  = torch.hub.load("intel-isl/MiDaS", "DPT_Large", trust_repo=True)
        tr = torch.hub.load("intel-isl/MiDaS", "transforms", trust_repo=True)
    _midas     = m.eval().to(DEVICE)
    _transform = tr.dpt_transform
    print("MiDaS ready.")


# ── depth helpers (identical to infer5.py) ─────────────────
def predict_depth(img_bgr):
    _load_midas()
    img_rgb     = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    input_batch = _transform(img_rgb).to(DEVICE)
    with torch.no_grad():
        pred = _midas(input_batch)
        pred = torch.nn.functional.interpolate(
            pred.unsqueeze(1),
            size=img_rgb.shape[:2],
            mode="bicubic", align_corners=False,
        ).squeeze()
    d = pred.cpu().numpy().astype(np.float32)
    lo, hi = d.min(), d.max();  d = (d - lo) / (hi - lo + 1e-8)
    d = 1.0 - d                                        # disparity → depth
    lo, hi = d.min(), d.max();  d = (d - lo) / (hi - lo + 1e-8)
    d_hires = cv2.resize(d, (640, 640), interpolation=cv2.INTER_LINEAR)
    return d.astype(np.float32), d_hires.astype(np.float32)


def shift_combined(depth, delta_d, steepness=8.0, midpoint=0.4,
                   edge_sensitivity=0.02, depth_weight=0.6, plane_weight=0.4):
    d_corr = depth.copy(); mask = d_corr > 0
    w_depth = np.ones_like(depth)
    w_depth[mask] = 1.0 / (1.0 + np.exp(steepness * (depth[mask] - midpoint)))
    gx = cv2.Sobel(depth, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(depth, cv2.CV_32F, 0, 1, ksize=3)
    gm = np.sqrt(gx**2 + gy**2)
    gm = (gm - gm.min()) / (gm.max() - gm.min() + 1e-8)
    w_plane = np.exp(-gm / edge_sensitivity)
    w = depth_weight * w_depth + plane_weight * w_plane
    w[mask] /= (w[mask].mean() + 1e-8)
    d_corr[mask] -= delta_d * w[mask]
    return np.clip(d_corr, 0.0, 1.0)


def shift_near_camera_edges(depth, delta_d, edge_strength=1.5, depth_threshold=0.35):
    d_corr = depth.copy(); H, W = depth.shape; mask = d_corr > 0
    xs = np.arange(W, dtype=np.float32); ys = np.arange(H, dtype=np.float32)
    XX, YY = np.meshgrid(xs, ys)
    radial  = np.sqrt(((XX-W/2)/(W/2))**2 + ((YY-H/2)/(H/2))**2) / np.sqrt(2.0)
    depth_w = 1.0 / (1.0 + np.exp(20.0 * (depth - depth_threshold)))
    d_corr[mask] += abs(delta_d) * (radial * depth_w * edge_strength)[mask]
    return np.clip(d_corr, 0.0, 1.0)


def depth_to_pc_sampled(depth_np, fx, fy, cx, cy, n=PCM_POINTS):
    H, W = depth_np.shape; valid = depth_np > 0
    us, vs = np.meshgrid(np.arange(W), np.arange(H))
    x =  (us[valid]-cx)/fx * depth_np[valid]
    y = -(vs[valid]-cy)/fy * depth_np[valid]
    z = -depth_np[valid]
    pts = np.stack([x,y,z],axis=1).astype(np.float32)
    idx = np.random.choice(len(pts), n, replace=(len(pts)<n))
    return pts[idx]


def depth_to_pc_rgb_full(depth_np, rgb_np, fx, fy, cx, cy):
    H, W = depth_np.shape
    rgb  = cv2.resize(rgb_np, (W, H)).astype(np.float32) / 255.0
    valid = depth_np > 0
    us, vs = np.meshgrid(np.arange(W), np.arange(H))
    x =  (us[valid]-cx)/fx * depth_np[valid]
    y = -(vs[valid]-cy)/fy * depth_np[valid]
    z = -depth_np[valid]
    return (np.stack([x,y,z],axis=1).astype(np.float32),
            rgb[valid].astype(np.float32))


def run_pcm(depth_norm, depth_hires, rgb_orig):
    H, W   = depth_norm.shape
    f_init = (W/2.0) / np.tan(np.deg2rad(INIT_FOV_DEG/2.0))
    cx, cy = W/2.0, H/2.0

    pts_pcm = depth_to_pc_sampled(depth_norm, f_init, f_init, cx, cy)
    pc_t    = torch.from_numpy(pts_pcm).unsqueeze(0).permute(0,2,1).to(DEVICE)

    with torch.no_grad():
        delta_d     = _shift_net(pc_t).item()
        alpha_f_raw = _focal_net(pc_t).item()

    alpha_f = float(np.clip(alpha_f_raw, 0.85, 1.15))
    delta_d = float(np.clip(delta_d,    -0.20, 0.60))
    print(f"  Δd={delta_d:+.4f}  αf={alpha_f:.4f}")

    d_corr       = shift_near_camera_edges(shift_combined(depth_norm,  delta_d), delta_d)
    d_hires_corr = shift_near_camera_edges(shift_combined(depth_hires, delta_d), delta_d)

    f_corr = f_init * alpha_f
    Hh, Wh = depth_hires.shape
    fh     = (Wh/2.0) / np.tan(np.deg2rad(INIT_FOV_DEG/2.0))
    fh_c   = fh * alpha_f
    cxh, cyh = Wh/2.0, Hh/2.0

    pc_raw,  col_raw  = depth_to_pc_rgb_full(depth_hires,  rgb_orig, fh,   fh,   cxh, cyh)
    pc_corr, col_corr = depth_to_pc_rgb_full(d_hires_corr, rgb_orig, fh_c, fh_c, cxh, cyh)

    return pc_raw, col_raw, pc_corr, col_corr, delta_d, alpha_f, f_corr


# ── routes ─────────────────────────────────────────────────
@app.get("/health")
def health():
    return {"status": "ok", "device": str(DEVICE), "model_loaded": _shift_net is not None}


@app.post("/infer")
async def infer(image: UploadFile = File(...)):
    if _shift_net is None:
        raise HTTPException(503, "Models not loaded")
    t0 = time.time()

    raw_bytes = await image.read()
    arr       = np.frombuffer(raw_bytes, np.uint8)
    img_bgr   = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise HTTPException(400, "Cannot decode image")
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    H, W    = img_bgr.shape[:2]

    depth_norm, depth_hires = predict_depth(img_bgr)

    pc_raw, col_raw, pc_corr, col_corr, delta_d, alpha_f, f_corr = \
        run_pcm(depth_norm, depth_hires, img_rgb)

    elapsed = round(time.time() - t0, 2)
    print(f"  Done {elapsed}s  raw={len(pc_raw):,}  refined={len(pc_corr):,}")

    return JSONResponse({
        "ok":             True,
        "elapsed_s":      elapsed,
        "img_w":          W,
        "img_h":          H,
        # ── depth map for frontend visualisation ──────────
        "depth_flat":     depth_norm.flatten().tolist(),  # [0..1] float, H*W values
        "depth_w":        W,
        "depth_h":        H,
        # ── PCM outputs ───────────────────────────────────
        "n_raw":          len(pc_raw),
        "n_refined":      len(pc_corr),
        "delta_d":        delta_d,
        "alpha_f":        alpha_f,
        "f_corr":         f_corr,
        "raw_cloud":      pc_raw.flatten().tolist(),
        "raw_colors":     col_raw.flatten().tolist(),
        "refined_cloud":  pc_corr.flatten().tolist(),
        "refined_colors": col_corr.flatten().tolist(),
    })
