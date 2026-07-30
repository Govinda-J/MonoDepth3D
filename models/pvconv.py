# models/pvconv.py
# The PVConv building block: fuses point features with voxel features.
# This is what separates PVCNN from a plain PointNet.
#
# Pipeline per PVConv layer:
#   1. Voxelize point features into a dense grid
#   2. Apply 3D conv on the voxel grid
#   3. Trilinearly interpolate voxel features back to each point
#   4. Concatenate with per-point MLP features
#   5. Fuse with another MLP

import torch
import torch.nn as nn
import torch.nn.functional as F


class Voxelization(nn.Module):
    def __init__(self, resolution):
        super().__init__()
        self.resolution = resolution

    def forward(self, features, coords):
        """
        Args:
            features : (B, C, N) point features
            coords   : (B, 3, N) point xyz in [-1, 1]
        Returns:
            voxel_features : (B, C, R, R, R)
        """
        B, C, N = features.shape
        R = self.resolution
        device = features.device

        # Normalize coords to voxel indices [0, R-1]
        norm = (coords + 1) / 2  # [-1,1] -> [0,1]
        idx = (norm * (R - 1)).long().clamp(0, R - 1)  # (B, 3, N)

        # Vectorized flat index calculation across the entire batch
        ix = idx[:, 0, :]  # (B, N)
        iy = idx[:, 1, :]  # (B, N)
        iz = idx[:, 2, :]  # (B, N)
        flat = ix * R * R + iy * R + iz  # (B, N)

        # Expand flat indices to match the feature channel dimension
        flat_expanded = flat.unsqueeze(1).expand(-1, C, -1)  # (B, C, N)

        # Single batched scatter_add (dim=2 is the spatial dimension)
        voxels = torch.zeros(B, C, R * R * R, device=device, dtype=features.dtype)
        voxels.scatter_add_(2, flat_expanded, features)

        voxels = voxels.reshape(B, C, R, R, R)
        return voxels

class TrilinearDevoxelization(nn.Module):
    def __init__(self, resolution):
        super().__init__()
        self.resolution = resolution

    def forward(self, voxels, coords):
        """
        Args:
            voxels : (B, C, R, R, R)
            coords : (B, 3, N) in [-1, 1]
        Returns:
            features : (B, C, N)
        """
        N = coords.shape[2]

        # grid_sample expects grid shape (B, N, 1, 1, 3)
        grid = coords.permute(0, 2, 1).unsqueeze(2).unsqueeze(2)  # (B,N,1,1,3)

        sampled = F.grid_sample(
            voxels,
            grid,
            mode='bilinear',
            padding_mode='border',
            align_corners=True
        )  # (B, C, N, 1, 1)

        return sampled.squeeze(-1).squeeze(-1)  # (B, C, N)


class PVConv(nn.Module):
    def __init__(self, in_channels, out_channels, resolution):
        """
        One PVConv block.
        Args:
            in_channels  : input feature channels
            out_channels : output feature channels
            resolution   : voxel grid resolution R
        """
        super().__init__()
        self.resolution = resolution

        # Voxel branch: 3D conv on the grid
        self.voxel_branch = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
        )

        # Point branch: per-point MLP
        self.point_branch = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, 1),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
        )

        # Fusion MLP after concatenation
        self.fusion = nn.Sequential(
            nn.Conv1d(out_channels * 2, out_channels, 1),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
        )

        self.voxelize   = Voxelization(resolution)
        self.devoxelize = TrilinearDevoxelization(resolution)

    def forward(self, inputs):
        """
        Args:
            inputs : tuple (features (B,C,N), coords (B,3,N))
        Returns:
            fused  : (B, out_channels, N)
            coords : (B, 3, N) unchanged
        """
        features, coords = inputs

        # Voxel path
        voxels      = self.voxelize(features, coords)
        voxels      = self.voxel_branch(voxels)
        voxel_feats = self.devoxelize(voxels, coords)

        # Point path
        point_feats = self.point_branch(features)

        # Fuse
        fused = self.fusion(torch.cat([voxel_feats, point_feats], dim=1))

        return fused, coords