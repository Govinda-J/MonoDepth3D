config = {
    # ── ScanNet metric intrinsics (used only in visualize.py GT cloud) ────────
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
    "train_split":    "C:\\scannet_data\\train_split.json",
    "test_split":     "C:\\scannet_data\\test_split.json",

    # TONIGHT: saving locally (no internet). Tomorrow morning:
    #   1. Move all .pth files from C:\pvcnn checkpoints  →  G:\My Drive\3D_Reconstruction_Project\pvcnn\checkpoints
    #   2. Swap the two lines below (comment tonight's, uncomment tomorrow's)
    "cloud_save_dir": "C:\\pvcnn checkpoints",
    # "cloud_save_dir": "G:\\My Drive\\3D_Reconstruction_Project\\pvcnn\\checkpoints",
}