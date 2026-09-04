# pcm_utils.py
# Shared depth-correction post-processing used by both infer5.py (CLI) and
# server.py (API). Keeps the correction logic in one place instead of duplicated.

import cv2
import numpy as np


# ---- SIGMOID + PLANARITY WEIGHTED SHIFT ----
def shift_combined(depth, delta_d, steepness=8.0, midpoint=0.4,
                   edge_sensitivity=0.02, depth_weight=0.6, plane_weight=0.4):
    """
    Applies delta_d correction weighted by a blend of:
      - sigmoid depth weight (favors near-camera pixels)
      - planarity weight (favors flat regions, avoids edges)
    """
    d_corr = depth.copy()
    mask   = d_corr > 0

    # Component 1: sigmoid depth weight
    w_depth       = np.ones_like(depth)
    w_depth[mask] = 1.0 / (1.0 + np.exp(steepness * (depth[mask] - midpoint)))

    # Component 2: planarity weight (inverse gradient)
    grad_x   = cv2.Sobel(depth, cv2.CV_32F, 1, 0, ksize=3)
    grad_y   = cv2.Sobel(depth, cv2.CV_32F, 0, 1, ksize=3)
    grad_mag = np.sqrt(grad_x**2 + grad_y**2)
    grad_mag = (grad_mag - grad_mag.min()) / (grad_mag.max() - grad_mag.min() + 1e-8)
    w_plane  = np.exp(-grad_mag / edge_sensitivity)

    # Combine and normalize
    weight        = depth_weight * w_depth + plane_weight * w_plane
    weight[mask] /= (weight[mask].mean() + 1e-8)

    d_corr[mask] -= delta_d * weight[mask]
    return np.clip(d_corr, 0.0, 1.0)


# ---- NEAR-CAMERA EDGE CORRECTION ----
def shift_near_camera_edges(depth, delta_d, edge_strength=1.5, depth_threshold=0.35):
    """
    Counteracts inward bending of near-camera surfaces at image edges by
    pushing radially-distant, shallow-depth points slightly farther away.
    """
    d_corr = depth.copy()
    H, W   = depth.shape
    mask   = d_corr > 0

    # Radial weight: 0 at image center, 1 at corners
    cy_n, cx_n = H / 2.0, W / 2.0
    ys = np.arange(H, dtype=np.float32)
    xs = np.arange(W, dtype=np.float32)
    XX, YY = np.meshgrid(xs, ys)
    rad_x  = ((XX - cx_n) / cx_n) ** 2
    rad_y  = ((YY - cy_n) / cy_n) ** 2
    radial = np.sqrt(rad_x + rad_y) / np.sqrt(2.0)

    # Depth weight: smooth sigmoid falloff at threshold (no hard cutoff)
    sharpness = 20.0
    depth_w   = 1.0 / (1.0 + np.exp(sharpness * (depth - depth_threshold)))

    weight = radial * depth_w * edge_strength
    correction_magnitude = abs(delta_d)

    d_corr[mask] += correction_magnitude * weight[mask]
    return np.clip(d_corr, 0.0, 1.0)