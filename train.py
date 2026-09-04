# train.py
# Stage-2 PCM training script: trains ShiftPVCNN and FocalPVCNN jointly on
# ScanNet depth maps. Not needed for inference/deployment — training only.

import os, sys, time, json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn

sys.path.insert(0, os.path.dirname(__file__))

from config import config
from models.pvcnn import ShiftPVCNN, FocalPVCNN
from datasets.scannet_dataset import DepthDataset, collate_skip_none

# ── HYPERPARAMETERS  ──────────────────────────────────────────────────────────────
BATCH_SIZE  = config["batch_size"]
ACCUM_STEPS = 4
EPOCHS      = config["epochs"]
LR          = config["lr"]
VOXEL_RES   = config["voxel_resolution"]
CKPT_DIR    = config["cloud_save_dir"]

def _unwrap(m):
    return getattr(m, "_orig_mod", m)

def main():
    os.makedirs(CKPT_DIR, exist_ok=True)

    # ---- GPU SETUP ----
    cudnn.benchmark     = True
    cudnn.deterministic = False

    # Enable TF32 for maximum RTX 4050 (Ampere) throughput
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device : {device}")
    if device.type == "cuda":
        print(f"GPU    : {torch.cuda.get_device_name(0)}")
        print(f"VRAM   : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # ── DATASETS ──────────────────────────────────────────────────────────────
    with open(config["train_split"]) as fh:
        train_paths = json.load(fh)
    with open(config["test_split"]) as fh:
        val_paths = json.load(fh)

    train_ds = DepthDataset(train_paths, split="train")
    val_ds   = DepthDataset(val_paths,   split="test")
    print(f"Train : {len(train_ds)} samples  |  Val : {len(val_ds)} samples")

    # num_workers > 0 on Windows requires if __name__ == '__main__': protection
    train_loader = DataLoader(
        train_ds, 
        batch_size=BATCH_SIZE, 
        shuffle=True,
        num_workers=8,               
        pin_memory=True, 
        drop_last=True,
        persistent_workers=True,     
        collate_fn=collate_skip_none
    )

    val_loader = DataLoader(
        val_ds, 
        batch_size=BATCH_SIZE, 
        shuffle=False,
        num_workers=8,               
        pin_memory=True, 
        drop_last=False,
        persistent_workers=True,     
        collate_fn=collate_skip_none
    )

    # ── MODELS ────────────────────────────────────────────────────────────────
    shift_net = ShiftPVCNN(VOXEL_RES).to(device)
    focal_net = FocalPVCNN(VOXEL_RES).to(device)

    if hasattr(torch, "compile") and sys.platform != "win32":
        print("Compiling with torch.compile() ...")
        shift_net = torch.compile(shift_net, mode="reduce-overhead")
        focal_net = torch.compile(focal_net, mode="reduce-overhead")
    else:
        print("Skipping torch.compile (not supported on Windows/no Triton)")

    # ── OPTIMIZERS & SCHEDULERS ────────────────────────────────────────────────────────────
    opt_shift = torch.optim.AdamW(shift_net.parameters(), lr=LR, weight_decay=1e-4)
    opt_focal = torch.optim.AdamW(focal_net.parameters(), lr=LR, weight_decay=1e-4)

    sched_shift = torch.optim.lr_scheduler.MultiStepLR(
        opt_shift, milestones=[int(EPOCHS*0.5), int(EPOCHS*0.8)], gamma=0.1)
    sched_focal = torch.optim.lr_scheduler.MultiStepLR(
        opt_focal, milestones=[int(EPOCHS*0.5), int(EPOCHS*0.8)], gamma=0.1)

    scaler_shift = torch.amp.GradScaler("cuda")
    scaler_focal = torch.amp.GradScaler("cuda")
    criterion    = nn.L1Loss()

    # ── CHECKPOINT HELPERS ────────────────────────────────────────────────────
    LATEST = os.path.join(CKPT_DIR, "latest_checkpoint.pth")
    BEST   = os.path.join(CKPT_DIR, "best_checkpoint.pth")

    def save_ckpt(path, epoch, best_val):
        torch.save({
            "epoch":      epoch,
            "best_val":   best_val,
            "shift_net":  _unwrap(shift_net).state_dict(),
            "focal_net":  _unwrap(focal_net).state_dict(),
            "opt_shift":  opt_shift.state_dict(),
            "opt_focal":  opt_focal.state_dict(),
            "sched_shift": sched_shift.state_dict(),
            "sched_focal": sched_focal.state_dict(),
        }, path)

    def load_ckpt(path):
        ckpt = torch.load(path, map_location=device, weights_only=True)
        _unwrap(shift_net).load_state_dict(ckpt["shift_net"])
        _unwrap(focal_net).load_state_dict(ckpt["focal_net"])
        if "opt_shift"   in ckpt: opt_shift.load_state_dict(ckpt["opt_shift"])
        if "opt_focal"   in ckpt: opt_focal.load_state_dict(ckpt["opt_focal"])
        if "sched_shift" in ckpt: sched_shift.load_state_dict(ckpt["sched_shift"])
        if "sched_focal" in ckpt: sched_focal.load_state_dict(ckpt["sched_focal"])
        return ckpt.get("epoch", 0), ckpt.get("best_val", float("inf"))

    # ── RESUME FROM CHECKPOINT ────────────────────────────────────────────────────────────────
    start_epoch, best_val = 0, float("inf")
    if os.path.exists(LATEST):
        start_epoch, best_val = load_ckpt(LATEST)
        print(f"Resumed from epoch {start_epoch},  best_val={best_val:.4f}")

    # ── TRAINING LOOP ─────────────────────────────────────────────────────────
    n_steps = len(train_loader)

    for epoch in range(start_epoch, EPOCHS):
        shift_net.train(); focal_net.train()
        t0 = time.time()
        train_s = train_f = 0.0

        opt_shift.zero_grad(set_to_none=True)
        opt_focal.zero_grad(set_to_none=True)

        for step, batch in enumerate(train_loader):
            pc_s = batch["pc_shift"].permute(0,2,1).to(device, non_blocking=True)
            pc_f = batch["pc_focal"].permute(0,2,1).to(device, non_blocking=True)
            gt_s = batch["delta_d"].to(device, non_blocking=True)
            gt_f = batch["alpha_f"].to(device, non_blocking=True)

            with torch.amp.autocast("cuda"):
                loss_s = criterion(shift_net(pc_s).squeeze(-1), gt_s) / ACCUM_STEPS
            if not torch.isnan(loss_s):
                scaler_shift.scale(loss_s).backward()
                train_s += loss_s.item() * ACCUM_STEPS

            with torch.amp.autocast("cuda"):
                loss_f = criterion(focal_net(pc_f).squeeze(-1), gt_f) / ACCUM_STEPS
            if not torch.isnan(loss_f):
                scaler_focal.scale(loss_f).backward()
                train_f += loss_f.item() * ACCUM_STEPS

            is_accum_step = (step + 1) % ACCUM_STEPS == 0
            is_last_step  = (step + 1) == n_steps
            if is_accum_step or is_last_step:
                scaler_shift.unscale_(opt_shift)
                scaler_focal.unscale_(opt_focal)
                nn.utils.clip_grad_norm_(_unwrap(shift_net).parameters(), 10.0)
                nn.utils.clip_grad_norm_(_unwrap(focal_net).parameters(), 10.0)
                scaler_shift.step(opt_shift); scaler_shift.update()
                scaler_focal.step(opt_focal); scaler_focal.update()
                opt_shift.zero_grad(set_to_none=True)
                opt_focal.zero_grad(set_to_none=True)

        train_s /= n_steps; train_f /= n_steps

        # ── VALIDATION ────────────────────────────────────────────────────────
        shift_net.eval(); focal_net.eval()
        val_s = val_f = 0.0
        nv = 0
        with torch.no_grad():
            for batch in val_loader:
                pc_s = batch["pc_shift"].permute(0,2,1).to(device, non_blocking=True)
                pc_f = batch["pc_focal"].permute(0,2,1).to(device, non_blocking=True)
                gt_s = batch["delta_d"].to(device, non_blocking=True)
                gt_f = batch["alpha_f"].to(device, non_blocking=True)
                with torch.amp.autocast("cuda"):
                    ls = criterion(shift_net(pc_s).squeeze(-1), gt_s).item()
                    lf = criterion(focal_net(pc_f).squeeze(-1), gt_f).item()
                # skip any batch that still somehow produces NaN
                if not (ls != ls or lf != lf):
                    val_s += ls
                    val_f += lf
                    nv    += 1

        if nv > 0:
            val_s /= nv; val_f /= nv
            val_total = val_s + val_f
        else:
            val_s = val_f = float("nan")
            val_total = float("inf")
            print("  [warn] all val batches NaN — check dataset")

        sched_shift.step(); sched_focal.step()

        print(f"[{epoch+1:03d}/{EPOCHS}] "
              f"train s={train_s:.4f} f={train_f:.4f}  "
              f"val s={val_s:.4f} f={val_f:.4f}  "
              f"lr={opt_shift.param_groups[0]['lr']:.2e}  "
              f"{time.time()-t0:.1f}s")

        # best_val updated before saving latest so resume reads correct value
        if val_total < best_val:
            best_val = val_total
            save_ckpt(BEST, epoch + 1, best_val)
            print(f"  *** best val={best_val:.4f} saved ***")

        save_ckpt(LATEST, epoch + 1, best_val)

    print(f"\nDone. Best val loss: {best_val:.4f}")

if __name__ == "__main__":
    main()