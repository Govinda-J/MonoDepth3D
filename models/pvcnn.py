# models/pvcnn.py
import torch
import torch.nn as nn
from models.pvconv import PVConv


class _BasePVCNN(nn.Module):
    """Shared PVCNN backbone. Subclasses define bounds and head."""

    def __init__(self, voxel_resolution, bound_min, bound_max, head):
        super().__init__()
        self.pvconv1 = PVConv(in_channels=3,   out_channels=64,
                              resolution=voxel_resolution)
        self.pvconv2 = PVConv(in_channels=64,  out_channels=128,
                              resolution=voxel_resolution // 2)
        self.pvconv3 = PVConv(in_channels=128, out_channels=256,
                              resolution=voxel_resolution // 4)
        self.head = head
        self.register_buffer("bound_min",
            torch.tensor(bound_min, dtype=torch.float32).view(1, 3, 1))
        self.register_buffer("bound_max",
            torch.tensor(bound_max, dtype=torch.float32).view(1, 3, 1))

    def forward(self, x):
        features = x
        coords   = x.clone()
        coords   = torch.clamp(coords, self.bound_min, self.bound_max)
        coords   = 2.0 * (coords - self.bound_min) / (
                       self.bound_max - self.bound_min + 1e-8) - 1.0
        features, coords = self.pvconv1((features, coords))
        features, coords = self.pvconv2((features, coords))
        features, coords = self.pvconv3((features, coords))
        return self.head(features.max(dim=2).values).squeeze(-1)


def _shift_head():
    seq = nn.Sequential(
        nn.Linear(256, 256),
        nn.LayerNorm(256),
        nn.SiLU(inplace=True),
        nn.Dropout(0.3),
        nn.Linear(256, 1),
    )
    nn.init.xavier_uniform_(seq[-1].weight)
    # Bias init = mean of Uniform(-0.25, 0.8) = 0.275
    # After z-centering, z_mean of pc_shift == delta_d,
    # so 0.275 is the correct prior for the output.
    nn.init.constant_(seq[-1].bias, 0.275)
    return seq


def _focal_head():
    seq = nn.Sequential(
        nn.Linear(256, 256),
        nn.LayerNorm(256),
        nn.ReLU(inplace=True),
        nn.Dropout(0.5),
        nn.Linear(256, 128),
        nn.LayerNorm(128),
        nn.ReLU(inplace=True),
        nn.Dropout(0.5),
        nn.Linear(128, 1),
    )
    nn.init.xavier_uniform_(seq[-1].weight)
    nn.init.constant_(seq[-1].bias, 0.0)
    return seq


class ShiftPVCNN(_BasePVCNN):
    """
    N_d: predicts delta_d from shift-corrupted, z-centered point cloud.

    After z-centering in distortions.py:
      z = (depth_gt + delta_d) - depth_gt.mean()
        = (depth_gt - depth_gt.mean()) + delta_d

    z range after centering:
      depth_gt in [0,1], depth_gt.mean() ~ 0.35-0.50 (scene dependent)
      z_min = 0 - 0.50 + (-0.25) = -0.75  (conservative lower bound)
      z_max = 1 - 0.35 + 0.80   = +1.45   (conservative upper bound)

    x,y range: (u-cx)/fx * z_s where z_s = depth_gt + delta_d (NOT centered)
      x range stays same as before centering: ~[-0.50, +0.50] for this fx
    """
    def __init__(self, voxel_resolution=32):
        super().__init__(
            voxel_resolution,
            bound_min=[-0.65, -0.65, -0.80],
            bound_max=[ 0.65,  0.65,  1.50],
            head=_shift_head(),
        )


class FocalPVCNN(_BasePVCNN):
    """
    N_f: predicts alpha_f from focal-corrupted point cloud.
    x = (u-cx)/(fx*af)*z — the x/z ratio encodes af.
    Both x and z are required: x alone is ambiguous (far object vs wide FOV).
    Bounds: x,y ∈ [-1.25, 1.25] (af=0.5 extreme), z ∈ [0.0, 1.0]
    """
    def __init__(self, voxel_resolution=32):
        super().__init__(
            voxel_resolution,
            bound_min=[-1.25, -1.25, 0.0],
            bound_max=[ 1.25,  1.25, 1.0],
            head=_focal_head(),
        )