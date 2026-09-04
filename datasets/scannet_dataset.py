# datasets/scannet_dataset.py
import torch
from torch.utils.data import Dataset
from utils.distortions import generate_training_samples, fov_to_focal
from config import config

# ---- DATASET ----
class DepthDataset(Dataset):

    def __init__(self, depth_paths: list, split: str = "train"):
        self.depth_paths   = depth_paths
        self.split         = split
        self.num_points    = config["num_points"]
        self.delta_d_range = config["delta_d_range"]
        self.alpha_f_range = config["alpha_f_range"]

        fov   = config["init_fov_deg"]
        W     = config["image_w"]
        H     = config["image_h"]
        f     = fov_to_focal(fov, W)
        self.fx = f
        self.fy = f
        self.cx = W / 2.0
        self.cy = H / 2.0

    def __len__(self) -> int:
        return len(self.depth_paths)

    def __getitem__(self, idx: int):
        try:
            depth = torch.load(
                self.depth_paths[idx],
                weights_only=True,
            ).float()
        except Exception:
            return None

        # ── Sanitize: reject NaN/Inf/empty depth maps ─────────────────────────
        if torch.isnan(depth).any() or torch.isinf(depth).any():
            return None
        if depth.max() < 1e-5:
            return None

        # ── Normalize to [0, 1] ───────────────────────────────────────────────
        d_min = depth.min()
        d_max = depth.max()
        depth = (depth - d_min) / (d_max - d_min + 1e-8)

        # ── Final safety check after normalization ────────────────────────────
        if torch.isnan(depth).any():
            return None

        # ── Generate corrupted point clouds ───────────────────────────────────
        (pc_shift, delta_d), (pc_focal, alpha_f) = generate_training_samples(
            depth,
            self.fx, self.fy,
            self.cx, self.cy,
            self.delta_d_range,
            self.alpha_f_range,
        )

        # ── Sanitize generated point clouds ───────────────────────────────────
        if torch.isnan(pc_shift).any() or torch.isnan(pc_focal).any():
            return None

        # ── Random subsample to fixed N ───────────────────────────────────────
        N     = self.num_points
        ids_s = torch.randperm(pc_shift.shape[0])[:N]
        ids_f = torch.randperm(pc_focal.shape[0])[:N]

        pc_shift = pc_shift[ids_s]
        pc_focal = pc_focal[ids_f]

        # ── Point dropout augmentation (training only) ────────────────────────
        if self.split == "train":
            pc_focal = _point_dropout(pc_focal, drop_prob=0.3)

        return {
            "pc_shift": pc_shift,
            "delta_d":  torch.tensor(delta_d, dtype=torch.float32),
            "pc_focal": pc_focal,
            "alpha_f":  torch.tensor(alpha_f, dtype=torch.float32),
        }


# ---- AUGMENTATION HELPER ----
def _point_dropout(pc: torch.Tensor, drop_prob: float = 0.3) -> torch.Tensor:
    N    = pc.shape[0]
    mask = torch.rand(N) > drop_prob
    kept = pc[mask]
    if kept.shape[0] < 64:
        return pc
    repeats = (N + kept.shape[0] - 1) // kept.shape[0]
    return kept.repeat(repeats, 1)[:N]


# ---- COLLATE FUNCTION ----
def collate_skip_none(batch: list) -> dict:
    batch = [b for b in batch if b is not None]
    if not batch:
        raise RuntimeError(
            "collate_skip_none: entire batch was None. "
            "Check your depth_paths — all frames in this batch were invalid."
        )
    return torch.utils.data.dataloader.default_collate(batch)