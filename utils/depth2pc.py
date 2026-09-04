# utils/depth2pc.py
# Standalone depth-to-pointcloud unprojection helper. NOT currently imported
# anywhere in the pipeline (server.py/infer5.py/visualize.py inline this logic).
import torch

# ---- DEPTH TO POINT CLOUD ----
def depth_to_pointcloud(depth, fx, fy, cx, cy):
    """
    Converts a raw depth map to a normalized GT point cloud.
    Depth is normalized to [0,1] as required by paper Section 2.1.

    Args:
        depth          : (H, W) float tensor — raw metric depth in meters
        fx, fy, cx, cy : camera intrinsics
    Returns:
        points : (H*W, 3) float tensor
    """
    d_min = depth.min()
    d_max = depth.max()
    depth_norm = (depth - d_min) / (d_max - d_min + 1e-8)

    H, W = depth_norm.shape
    device = depth_norm.device

    v, u = torch.meshgrid(
        torch.arange(H, device=device, dtype=torch.float32),
        torch.arange(W, device=device, dtype=torch.float32),
        indexing="ij"
    )

    z = depth_norm
    x = (u - cx) * z / fx
    y = (v - cy) * z / fy

    points = torch.stack([x, y, z], dim=-1).reshape(-1, 3)
    return points