import os
from pathlib import Path


# ── Project paths ─────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(
    os.environ.get("MONODEPTH3D_ROOT",
        Path(__file__).resolve().parent
    )
)

PVCNN_MODEL_REPO = os.environ.get(
    "PVCNN_MODEL_REPO",
    "Govinda-J/MonoDepth3D-Checkpoint"
)

CHECKPOINT = Path(
    os.environ.get("PVCNN_CHECKPOINT",
        PROJECT_ROOT / "checkpoints" / "best_checkpoint.pth"
    )
)

OUTPUT_DIR = Path(
    os.environ.get("PCM_OUTPUT_DIR",
        PROJECT_ROOT / "outputs"
    )
)

# ── ScanNet metric intrinsics ────────────
config = {
    "fx": 577.590698, "fy": 578.729797,
    "cx": 318.905426, "cy": 242.683609,

    # ── Image dimensions (ScanNet standard) ───────────────────────────────────
    "image_w": 640,
    "image_h": 480,

    # ── Paper §2.1: initial FOV guess used at inference → must match training ─
    "init_fov_deg": 60.0,

    # ── PCM training hyper-params ─────────────────────────────────────────────
    "num_points":       8192,
    "delta_d_range":    (-0.25, 0.8),
    "alpha_f_range":    (0.5, 1.5),
    "voxel_resolution": 32,
    "batch_size":       24,
    "lr":               5e-4,
    "lr_decay":         0.1,
    "epochs":           100,
    "visualize_every":  10,

    # ── Paths ─────────────────────────────────────────────────────────────────
    # USED DURING TRAINING - NOT FOR INFERENCE
    # NOT TO BE USED - UNLESS HARDCODED PATHS ARE CORRECTED
    "train_split": os.environ.get(
        "SCANNET_TRAIN_SPLIT",
        r"C:\scannet_data\train_split.json"
    ),

    "test_split": os.environ.get(
        "SCANNET_TEST_SPLIT",
        r"C:\scannet_data\test_split.json"
    ),
 # ── Output path ───────────────────────────────────────────────────────────
    # Can be overridden with PCM_OUTPUT_DIR.
    "cloud_save_dir": str(OUTPUT_DIR),
}