# utils/distortions.py
# Generates synthetic depth-shift / focal-scale corruptions for training
# ShiftPVCNN and FocalPVCNN. Training-only — not used at inference time.
#
#   Shift network input  : F(u0, v0, f*, d* + Δd)  — corrupt depth, GT focal
#   Focal network input  : F(u0, v0, αf.f*, d*)     — corrupt focal, GT depth
#
# depth_gt passed in must already be normalised to [0,1] by the caller
# (scannet_dataset.py does this). Do NOT re-normalise here.
#
# KEY DESIGN DECISION — z-centering for ShiftPVCNN:
#   Raw z_mean of pc_shift = depth_scene_mean + Δd, which confounds the
#   signal across scenes. Subtracting depth_gt.mean() isolates delta_d exactly.
#   Inference-safe: depth_gt.mean() == d_norm.mean() (no GT leak).
#   infer5.py / server.py must apply this same z -= d_norm.mean() step.

import torch
import math

# ---- INTRINSICS HELPER ----
def fov_to_focal(fov_deg: float, width: int) -> float:
    return (width / 2.0) / math.tan(math.radians(fov_deg / 2.0))

# ---- TRAINING SAMPLE GENERATION ----
def generate_training_samples(depth_gt, fx, fy, cx, cy,
                               delta_d_range=(-0.25, 0.8),
                               alpha_f_range=(0.5, 1.5)):
    """
    Args:
        depth_gt        : (H, W) float tensor, already normalised to [0, 1]
        fx, fy, cx, cy  : FOV-derived intrinsics (same at train and inference)
        delta_d_range   : uniform sample range for depth shift
        alpha_f_range   : uniform sample range for focal scale

    Returns:
        (pc_shift, delta_d) : input + label for ShiftPVCNN
        (pc_focal, alpha_f) : input + label for FocalPVCNN
    """
    H, W = depth_gt.shape
    v, u = torch.meshgrid(
        torch.arange(H, dtype=torch.float32),
        torch.arange(W, dtype=torch.float32),
        indexing="ij",
    )

    delta_d = torch.FloatTensor(1).uniform_(*delta_d_range).item()
    alpha_f = torch.FloatTensor(1).uniform_(*alpha_f_range).item()

    # ---- SHIFT NETWORK INPUT : GT focal (f*), corrupted depth (d* + Δd) ----
    z_s = depth_gt + delta_d
    x_s = (u - cx) / fx * z_s
    y_s = (v - cy) / fy * z_s
    pc_shift = torch.stack([x_s, y_s, z_s], dim=-1).reshape(-1, 3)

    # Remove scene depth bias so pc_shift[:,2].mean() == delta_d exactly.
    # INFERENCE CONTRACT: caller must apply pc[:,2] -= d_norm.mean().
    z_ref = depth_gt.mean()
    pc_shift[:, 2] = pc_shift[:, 2] - z_ref

    # ---- FOCAL NETWORK INPUT : GT depth (d*), corrupted focal (αf · f*) ----
    z_f = depth_gt
    x_f = (u - cx) / (fx * alpha_f) * z_f
    y_f = (v - cy) / (fy * alpha_f) * z_f
    pc_focal = torch.stack([x_f, y_f, z_f], dim=-1).reshape(-1, 3)

    return (pc_shift, delta_d), (pc_focal, alpha_f)