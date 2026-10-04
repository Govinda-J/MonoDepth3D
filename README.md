# MonoDepth3D

**Correcting the geometry of 3D point clouds reconstructed from a single RGB image, using a Point-Voxel CNN.**

MonoDepth3D (internally called the **Point Cloud Module / PCM**) takes one ordinary photo of an indoor scene, predicts a depth map for it with MiDaS, back-projects that depth map into a 3D point cloud with a pinhole camera model — and then runs that point cloud through two small custom-trained **PVCNN** (Point-Voxel CNN) networks that detect and undo the systematic geometric distortion this process always introduces (bent walls, warped floors, mismatched scale). No LiDAR, no stereo rig, no depth camera — just an RGB image in, a geometrically corrected colored point cloud out.

This README explains **why** the raw reconstruction is wrong in the first place, **how** the correction networks are designed and trained, and **how** all of that is wired together in this repository.

---

## 🤝 Meet the Team

<p align="center">
  Built in collaboration with an incredible team of peers 🚀<br />
  <strong>Meet our Development Squad</strong>
</p>

<table align="center">
  <tr>
    <!--  ME  -->
    <td align="center" width="200px">
      <a href="https://github.com/Govinda-J">
        <img src="https://github.com/Govinda-J.png" width="100" height="100" style="border-radius: 50%; border: 2px solid #58a6ff;" alt="Govinda"/>
        <br />
        <br />
        <strong>Govinda J</strong>
      </a>
      <br />
      <small><a href="https://github.com/Govinda-J">@Govinda-J</a></small>
      <br />
      <br />
      <sub title="Contributions"><kbd>PVCNN Model Design, Inference & Web Design</kbd></sub>
    </td>
    <!-- TEAM MEMBER 1 -->
    <td align="center" width="200px">
      <a href="https://github.com/BhargavRam0307">
        <img src="https://github.com/BhargavRam0307.png" width="100" height="100" style="border-radius: 50%;" alt="Bhargav"/>
        <br />
        <br />
        <strong>MP Bhargav Ram</strong>
      </a>
      <br />
      <small><a href="https://github.com/BhargavRam0307">@BhargavRam0307</a></small>
      <br />
      <br />
      <sub title="Contributions"><kbd>Dataset Preprocessing, PVCNN Model Training & Web Design</kbd></sub>
    </td>
    <!-- TEAM MEMBER 2 -->
    <td align="center" width="200px">
      <a href="https://github.com/08ganesh">
        <img src="https://github.com/08ganesh.png" width="100" height="100" style="border-radius: 50%;" alt="Ganesh"/>
        <br />
        <br />
        <strong>Eggoni Ganesh</strong>
      </a>
      <br />
      <small><a href="https://github.com/08ganesh">@08ganesh</a></small>
      <br />
      <br />
      <sub title="Contributions"><kbd>Depth Module (MiDaS) & Point-cloud Visualization</kbd></sub>
    </td>
  </tr>
</table>

---

## Table of Contents

1. [What This Project Actually Does](#1-what-this-project-actually-does)
2. [Why Monocular Point Clouds Come Out Wrong](#2-why-monocular-point-clouds-come-out-wrong)
3. [System Architecture — The Two-Stage Pipeline](#3-system-architecture--the-two-stage-pipeline)
4. [PVCNN — The Model Doing the Correction](#4-pvcnn--the-model-doing-the-correction)
5. [Training Phase](#5-training-phase)
6. [Post-Processing: Turning Two Numbers Into a Corrected Cloud](#6-post-processing-turning-two-numbers-into-a-corrected-cloud)
7. [Tech Stack](#8-tech-stack)
8. [Project / Code Structure](#9-project--code-structure)
9. [Caveats](#11-limitations--caveats)
10. [Requirements](#12-requirements)
11. [Try it out Yourself - Running This Locally](#13-try-it-out-yourself-running-this-locally)
12. [References](#15-references)

---

## 1. What This Project Actually Does

- **Input**: one RGB photo of an indoor scene (a room, corridor, kitchen, hall, etc.).
- **Stage 1 — Depth Prediction**: a pre-trained **MiDaS DPT-Large** model predicts a relative depth map for the image.
- **Back-projection**: that depth map is unprojected into an initial 3D point cloud using a pinhole camera model and an assumed field of view — this projected cloud is geometrically distorted (bent walls, wrong scale) because monocular depth is only *relative*, not metric.
- **Stage 2 — Point Cloud Module (PCM)**: two independently trained PVCNN networks — **ShiftPVCNN** and **FocalPVCNN** — look at the distorted point cloud and each predict a single scalar correction:
  - `Δd` (depth shift) from ShiftPVCNN
  - `αf` (focal-length scale factor) from FocalPVCNN
- These two scalars drive a spatially-adaptive correction pass over the depth map, which is then re-projected into the final, corrected point cloud.
- The whole thing is exposed as a **FastAPI backend** (`server.py`) and a **Three.js browser viewer** (`index.html` / `app_v2.js` / `viewer.js`).

Everything below is about the **indoor-scene** pipeline specifically — this repo's PCM is trained and evaluated only at the full-scene level (not on isolated objects).

---

## 2. Why Monocular Point Clouds Come Out Wrong

A single photo has no depth channel. Models like MiDaS are trained on a mixture of datasets using **scale- and shift-invariant losses**, which lets them generalize to almost any scene — but as a direct consequence, what they output is only correct *up to an unknown affine transform*: `predicted_depth ≈ true_depth × scale + shift`. An unknown *scale* just makes the whole reconstructed scene uniformly bigger or smaller, which is harmless for shape. An unknown **shift**, however, is not harmless — because the pinhole projection is non-linear in depth, a constant depth offset bends flat surfaces (walls, floors) into curves once they're projected into 3D. On top of that, the camera's true focal length is usually unknown at inference time; using the wrong one distorts how quickly geometry spreads out away from the image center.

So a raw monocular point cloud typically suffers from two decoupled, learnable error modes:

| Error | Symbol | Effect if uncorrected |
|---|---|---|
| Depth shift | `Δd` | Flat walls/floors appear curved; overall depth offset from camera |
| Focal-length error | `αf` | Scene appears stretched or compressed away from the image center |

The whole point of the PCM is to learn to estimate `Δd` and `αf` directly from the shape of the distorted point cloud itself, and it can do this in a **self-supervised** way — see [Section 5](#5-training-phase).

---

## 3. System Architecture — The Two-Stage Pipeline

<img width="1289" height="832" alt="Arch_Diagram" src="https://github.com/user-attachments/assets/14109593-606e-4ee6-aa23-f038464d4ab0" />


### 3.1 Stage 1 — Depth Prediction Module (DPM)

`MiDaS DPT-Large` produces a **depth map / disparity map** (larger value = closer). Because pinhole back-projection needs *depth* (larger value = farther), the raw output is inverted:

1. Normalize raw MiDaS output to `[0, 1]`.
2. Invert: `d = 1 - d`.
3. Re-normalize to `[0, 1]`.

This two-step normalize→invert→normalize sequence is numerically stable, unlike a direct `1/d` inversion which explodes for near-zero disparity pixels.

### 3.2 Pinhole back-projection

Given image coordinates `(u, v)` and normalized depth `d`, with an assumed field of view of 60° (used identically at training and inference so the network never has to "learn around" a mismatch):

```
f  = (W/2) / tan(FOV/2)        # fov_to_focal() in distortions.py
x  = (u - cx) / fx * d
y  = (v - cy) / fy * d
z  = -d
```

The full-resolution cloud is randomly subsampled to a fixed **N = 8,192 points** before being handed to the PCM — a fixed size the PVCNN backbone's voxelization step depends on being roughly consistent in density.

### 3.3 Stage 2 — Point Cloud Module (PCM)

The PCM does **not** try to move every one of the 8,192 points individually. For a whole scene, per-point offsets are unconstrained and easy to overfit into noise; instead, the PCM regresses just **two global scalars** — a depth shift `Δd` and a focal scale `αf` — that are then applied *analytically* back onto the original depth map (Section 6). This keeps the correction geometrically meaningful (it's still a real pinhole re-projection) instead of an arbitrary point cloud deformation.

Two separate PVCNN networks are used rather than one shared network with two output heads, because the two regression targets encode geometrically distinct distortion types and empirically train better independently:

- **ShiftPVCNN** — sees a *depth-shift-corrupted* cloud, predicts `Δd`.
- **FocalPVCNN** — sees a *focal-corrupted* cloud, predicts `αf`.

---

## 4. PVCNN — The Model Doing the Correction

### 4.1 The problem PVCNN solves

There are two traditional ways to feed a neural net a 3D point cloud, and both have a real cost:

- **Voxel-based** (rasterize points onto a dense 3D grid, then run ordinary 3D convolutions): gives great memory locality and lets convolutions see neighborhoods cheaply, but memory and compute scale as `O(R³)` in the grid resolution `R`. Doubling the resolution costs roughly 8× the memory — quickly prohibitive on a single GPU.
- **Point-based** (process the raw `(x, y, z)` list directly, à la PointNet-style models): memory-efficient since no dense grid is allocated, but neighbor lookups require nearest-neighbor search over scattered, non-contiguous memory. In practice, well over half the runtime of many point-based architectures is spent just organizing irregular memory access and computing dynamic convolution kernels — not on the actual feature extraction.

**PVCNN** (Liu, Tang, Lin & Han, *NeurIPS 2019*) resolves this by not choosing one representation — it runs **both, in parallel, every layer**, and fuses the results:

- A **voxel branch** temporarily rasterizes the points onto a coarse grid, convolves them there (cheap, regular, good locality), then interpolates the result back onto each point (devoxelization).
- A **point branch** processes every point independently with a shared per-point MLP, so exact positional precision is never lost to the coarse voxel grid.

<img width="588" height="321" alt="Screenshot 2026-09-08 111043" src="https://github.com/user-attachments/assets/3ef37148-a2c8-4a30-bc18-a3a27d0b390e" />


The two branches' outputs are concatenated and fused per point. This gives point-level precision *and* the contiguous-memory benefit of convolving on a grid, without paying the cubic memory cost of a fully voxel-based model at high resolution — the original paper reports roughly **10× lower memory** than an equivalent voxel-based model and **7× faster inference** than competing point-based models.


### 4.2 Inside one PVConv block


<img width="800" height="515" alt="pvcnn_arch" src="https://github.com/user-attachments/assets/a7ec82ee-13eb-42e5-9066-6699111e9152" />



- **Voxel path.** Point coordinates (already normalized to `[-1, 1]`) are mapped to integer voxel indices:

```
ix = floor((x + 1) * (R - 1) / 2)     # similarly for iy, iz
```

Every point's feature is scatter-added into its voxel cell, ordinary `Conv3D → BatchNorm3D → ReLU` layers run over the resulting dense grid (this is `pvconv.py`'s `Voxelization` → `voxel_branch`), and the result is **trilinearly interpolated** back onto the original point coordinates (`TrilinearDevoxelization`, implemented here via `F.grid_sample`).

- **Point path.** The exact same input features are independently passed through a `Conv1d(kernel_size=1)` MLP — mathematically identical to a per-point fully connected layer, since a 1×1×1 kernel never looks at neighbors. This branch never loses precision to a grid, but also never gains any neighborhood context — that's what the voxel branch is for.

- **Fusion.** The devoxelized voxel features and the point-branch features are concatenated channel-wise and passed through one more `Conv1d + BN + ReLU` to produce the block's output — same point count as the input, richer per-point features.

### 4.3 The backbone used in this repo

Both `ShiftPVCNN` and `FocalPVCNN` (`models/pvcnn.py`) share a common `_BasePVCNN` backbone of three PVConv layers, with **channel width increasing while voxel resolution decreases** — the same "more channels, coarser grid, bigger receptive field" trade shallow-to-deep CNNs use in 2D:

| Layer | In → Out channels | Voxel res. | Role |
|---|---|---|---|
| PVConv 1 | 3 → 64 | 32³ | Coarse local geometry — a large grid captures room-scale structure |
| PVConv 2 | 64 → 128 | 16³ | Mid-level features — halved resolution, doubled channels |
| PVConv 3 | 128 → 256 | 16³ | Deep voxel features — surface orientation, inter-surface relationships |

After the three PVConv layers, a **global max-pool over all points** (`features.max(dim=2).values`) collapses the per-point feature tensor into a single scene-level feature vector — appropriate here because the target is a *global* scalar (`Δd` or `αf`), not a per-point value. That vector is passed into a task-specific regression head:

```python
# Shift head (models/pvcnn.py) — predicts Δd
nn.Sequential(
    nn.Linear(256, 256), nn.LayerNorm(256), nn.SiLU(), nn.Dropout(0.3),
    nn.Linear(256, 1),
)
# bias initialized to 0.275 = mean of the training Δd range (-0.25, 0.8)

# Focal head — predicts αf (deeper, higher dropout — a harder, more ambiguous target)
nn.Sequential(
    nn.Linear(256, 256), nn.LayerNorm(256), nn.ReLU(), nn.Dropout(0.5),
    nn.Linear(256, 128), nn.LayerNorm(128), nn.ReLU(), nn.Dropout(0.5),
    nn.Linear(128, 1),
)
```

Before voxelization, input coordinates are clamped and rescaled into each network's own expected bound box (`bound_min` / `bound_max` in `pvcnn.py`) — ShiftPVCNN and FocalPVCNN see differently-shaped input clouds (Section 5.1), so each gets its own empirically-derived coordinate bounds rather than sharing one normalization.

---

## 5. Training Phase

### 5.1 Self-supervised distortion synthesis

Training a network to *undo* a distortion normally needs paired (distorted, ground-truth) 3D data — expensive to collect at scene scale. This project instead trains entirely on **ScanNet**, a large indoor RGB-D dataset (~0.25M frames, 1,513 scenes) with *known-good* depth and camera intrinsics, and manufactures its own training pairs by deliberately corrupting that known-good geometry (`utils/distortions.py`):

```python
# Shift network input — corrupt depth, keep GT focal (adapted from [2])
z_s = depth_gt + delta_d
x_s = (u - cx) / fx * z_s
y_s = (v - cy) / fy * z_s

# Focal network input — corrupt focal, keep GT depth (adapted from [2])
z_f = depth_gt
x_f = (u - cx) / (fx * alpha_f) * z_f
y_f = (v - cy) / (fy * alpha_f) * z_f
```

with `Δd ~ U(-0.25, 0.8)` and `αf ~ U(0.5, 1.5)` sampled fresh per training sample. This is self-supervised in the sense that no RGB-to-3D-corrected pairs are ever needed — the "ground truth" is simply the known distortion each sample was deliberately given. It works because the domain gap between real and synthetic distortion is far smaller **in 3D point-cloud space** than it would be trying to synthesize distorted *images*.

Both distorted clouds are subsampled to `N = 8,192` points; the focal-distorted cloud additionally gets **30% random point dropout during training only**, so FocalPVCNN is forced to learn the geometric x/z ratio that actually encodes `αf`, rather than memorizing exact point layouts.

**Z-centering for ShiftPVCNN (a repo-specific fix on top of the base recipe).** The raw `z` mean of the shift-distorted cloud is `depth_scene_mean + Δd` — and `depth_scene_mean` varies scene to scene across the 42K-sample training set, which makes the gradient signal for `Δd` noisier than necessary. Subtracting `depth_gt.mean()` from `z` before training removes that confound so that `z.mean() == Δd` exactly, for every scene. This is *not* a data leak — `depth_gt.mean()` is numerically identical to `d_norm.mean()`, the predicted depth's own mean, which is trivially available at inference too (see `infer5.py` / `server.py`, which apply the identical subtraction before calling ShiftPVCNN).

### 5.2 Training configuration

Two independent PVCNN networks, each trained with plain **L1 loss** against its scalar target:

| Setting | Value |
|---|---|
| Optimizer | AdamW, weight decay `1e-4` |
| Learning rate | `5e-4` |
| LR schedule | MultiStepLR, ×0.1 at 50% and 80% of training |
| Batch size | 24 (with gradient accumulation over 4 steps) |
| Epochs | 100 |
| Voxel resolution | 32 |
| Loss | L1 (independently, per network) |

`train.py` mixed-precision-trains both networks in the same loop (`torch.amp.autocast` + separate `GradScaler`s per network), clips gradients to a norm of 10, and checkpoints both `latest_checkpoint.pth` and `best_checkpoint.pth` (by summed validation loss) so training can resume cleanly after an interruption.

### 5.3 What convergence looked like

Both networks converged steadily over 100 epochs with a small, consistent train/validation gap (no meaningful overfitting). ShiftPVCNN reached its best validation performance around epoch 95; FocalPVCNN converged faster early on, owing to its narrower, more constrained target range.

**Zero-shot generalization** (evaluated on 100 unseen NYU Depth V2 images — a dataset the PCM was never trained on):

| Metric | Before → After | Improved samples |
|---|---|---|
| Planarity score | 0.549 → 0.572 (+4.2%) | 92% |
| Normal consistency | 0.596 → 0.613 (+2.3%) | 93% |
| Δd applied | mean −0.037 (range −0.06 to +0.025) | 96% |
| αf applied | mean 0.985 (range 0.97–1.023) | 96% |

Higher Planarity Score and Normal Consistency after correction confirm the intended effect: previously-curved walls/floors become measurably flatter and more uniformly oriented — exactly the failure mode described in Section 2 — without ever having seen an NYU image during training.

---

## 6. Post-Processing: Turning Two Numbers Into a Corrected Cloud

`Δd` and `αf` are single scalars per image — turning them into a good corrected cloud takes more than a uniform shift, because a *uniform* depth offset leaves distance-dependent and edge-dependent bending artifacts uncorrected. `pcm_utils.py` applies a **spatially-adaptive** correction combining three weighted components before re-projecting:

1. **Sigmoid depth weighting** — correction strength is strongest for near-camera pixels and smoothly fades for distant ones (monocular depth errors are typically worse up close):
   `w_depth(u,v) = 1 / (1 + exp(s·(d(u,v) − m)))`, with steepness `s = 8.0`, midpoint `m = 0.4`.
2. **Planarity weighting** — correction is stronger on flat regions (small local depth gradient) and backs off near depth discontinuities, so object edges aren't smeared:
   `w_plane(u,v) = exp(-‖∇d(u,v)‖ / σ)`, `σ = 0.02`.
3. **Near-camera radial correction** — a second pass nudges points near the image border (where perspective bending is most visible) outward/inward based on radial distance from the image center.

Predicted corrections are also clamped to safe, empirically-chosen ranges before use — `Δd ∈ [-0.2, 0.6]`, `αf ∈ [0.85, 1.15]` — so a rare bad prediction can't produce a wildly broken reconstruction. The corrected focal length `f_corr = f_init · αf` and the corrected depth map are then re-projected with the same pinhole model from Section 3.2 to produce the final colored point cloud.

---

## 7. Tech Stack

| Layer | Technology |
|---|---|
| Depth estimation | MiDaS DPT-Large (`torch.hub`) |
| Correction networks | Custom PyTorch: PVCNN / PVConv (built from scratch, not a third-party library) |
| Training data | ScanNet (RGB-D indoor scenes) |
| Zero-shot eval | NYU Depth V2, custom web-sourced indoor images |
| Backend API | FastAPI, Uvicorn |
| Model hosting | Hugging Face Hub (`hf_hub_download`) for the trained checkpoint |
| Browser viewer | Three.js (r128), vanilla JS, HTML/CSS |
| Frontend hosting | Netlify (static) |

---

## 8. Project / Code Structure

```
.
├── config.py                # Central paths + hyperparameters (env-var overridable)
├── models/
│   ├── pvcnn.py              # ShiftPVCNN, FocalPVCNN, shared _BasePVCNN backbone
│   └── pvconv.py              # PVConv block: Voxelization + Conv3D branch + Devoxelization + point MLP
├── datasets/
│   └── scannet_dataset.py     # Loads ScanNet depth frames, generates synthetic distortions per-sample
├── utils/
│   ├── distortions.py          # Synthesizes shift-distorted / focal-distorted training clouds
│   └── pcm_utils.py             # Spatially-adaptive depth correction (shift_combined, shift_near_camera_edges)
├── train.py                    # Stage-2 PCM training loop (both networks, AMP, checkpointing)
├── infer5.py                   # Standalone inference for local system
├── server.py                   # FastAPI backend: /health, /infer — MiDaS + PCM inference over HTTP
└── website/
    ├── index.html               # Upload → config → run → results UI shell
    ├── style.css
    └── js/
        ├── app_v2.js             # Talks to the backend, renders depth map + metrics
        └── viewer.js               # From-scratch Three.js point cloud viewer (orbit controls, export to .xyz)
```

Each major file's job, briefly:

- **`config.py`** — the single source of truth for paths and hyperparameters. Reads overrides from environment variables (`MONODEPTH3D_ROOT`, `PCM_OUTPUT_DIR`, `SCANNET_TRAIN_SPLIT`/`SCANNET_TEST_SPLIT`, `PVCNN_MODEL_REPO`) so the same code runs unmodified on a laptop, a training rig, or inside a container.
- **`distortions.py` / `scannet_dataset.py`** — everything about *how* the self-supervised training pairs are generated (Section 5.1) lives here; nowhere else.
- **`pvcnn.py` / `pvconv.py`** — the model itself (Section 4); no training or inference logic leaks in here.
- **`train.py`** — training only; not needed for anyone who just wants to run inference with the pre-trained checkpoint.
- **`infer5.py`** vs **`server.py`** — two different front doors to the *same* correction logic. `infer5.py` is for a local desktop session; `server.py` is the same pipeline behind a REST endpoint for the browser viewer.

---

## 9. Caveats

- **Affine-invariance ceiling.** MiDaS predicts *relative*, non-linear-normalized disparity — the PCM corrects shift and focal error, but cannot fully undo the residual curvature this normalization leaves on planar surfaces, since that would require true metric depth.
- **No occlusion completion.** Only the visible surface in the depth map is reconstructed — anything the camera can't see (the back of a sofa, the far side of a wall) is simply absent from the point cloud, by design.
- **No metric scale recovery.** The pipeline fixes *shape*, not absolute size — real-world distances/dimensions aren't recoverable without an external reference.
- **Thin structures suffer.** Chair legs, window bars, and similarly thin geometry are hard to preserve at the fixed voxel resolution the PVCNN backbone uses, and can be smoothed away or mis-corrected.
- **Domain sensitivity.** The PCM is trained exclusively on ScanNet indoor scans; performance on scenes far outside that distribution (outdoor scenes, stylized/rendered imagery) is not guaranteed and hasn't been evaluated here.
- **Known open issue — FocalPVCNN convention mismatch.** The training contract in `distortions.py` expects positive, z-centered point clouds; the current inference code paths (`infer5.py`, `server.py`) feed FocalPVCNN clouds with negated y/z and no z-centering. This mismatch is suspected to make FocalPVCNN's raw predictions collapse toward a narrow band (hence the tight `[0.85, 1.15]` inference-time clamp as a safety net) rather than reflecting the network's full, trained expressiveness. This is a correctness issue distinct from packaging/deployment and is flagged here for transparency rather than papered over — it's on the list to fix by aligning the inference-time preprocessing with the training contract.
- **No automated test suite.** Everything has been verified manually end-to-end so far.

---

## 10. Requirements

- Python 3.10.13
- PyTorch (CPU **or** GPU build — pick one at install time; nothing else in the codebase changes)
- OpenCV, NumPy
- FastAPI + Uvicorn (server only)
- HuggingFace Hub

**Installing PyTorch — CPU vs GPU is the only decision point:**

```bash
# CPU-only (no NVIDIA GPU, or you just want it running with zero setup)
pip install torch --index-url https://download.pytorch.org/whl/cpu

# GPU (NVIDIA, CUDA 12.1 example — check pytorch.org for your CUDA version)
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

Everything downstream (`train.py`, `infer5.py`, `server.py`) auto-detects the device with `torch.device("cuda" if torch.cuda.is_available() else "cpu")` — no code changes needed either way.

**Rough inference time per image** (MiDaS DPT-Large + PCM forward pass; the depth model dominates the runtime, actual numbers vary with CPU/GPU model and image size):

| Hardware | Approx. time per image |
|---|---|
| Modern NVIDIA GPU (e.g. RTX 30/40-series) | ~0.5–2 seconds |
| CPU only (modern multi-core laptop/desktop) | ~15–45 seconds |

CPU inference is entirely usable for trying the project out — it's just not snappy for repeated interactive use.

---

## 11. Try it out Yourself - Running This Locally

### Backend

```bash
git clone <this-repo>
cd MonoDepth3D
pip install -r requirements.txt         # or requirements-desktop.txt to also get PyVista for infer5.py
```

Set the checkpoint source (either let the server auto-download from the configured Hugging Face repo, or point `MONODEPTH3D_ROOT` at a local checkout that already has `checkpoints/best_checkpoint.pth`).

```bash
uvicorn server:app --host 0.0.0.0 --port 8000 --reload
```

Confirm it's alive:

```bash
curl http://127.0.0.1:8000/health
# {"status": "ok", "device": "cuda" or "cpu", "model_loaded": true}
```

### Frontend

Open `website/index.html` directly in a browser (or serve the `website/` folder with any static file server), and paste your backend's URL — e.g. `http://127.0.0.1:8000` — into the **API** field at the top of the page. Upload an image and hit **Run Inference**.

---

### Deployment Notes

Only the **frontend** (`website/`) is deployed as a static site on **Netlify**. The **backend is not hosted publicly**; it's meant to be run locally or self-deployed. Whoever wants to use the Netlify-hosted frontend simply runs the backend themselves (locally, or through deployment) and pastes that backend's URL into the frontend's **API** field — as the frontend never assumes a fixed backend address.

---

## 12. References

[1] Liu, Z., Tang, H., Lin, Y., & Han, S. *Point-Voxel CNN for Efficient 3D Deep Learning.* NeurIPS 2019.

[2] Yin, W., Zhang, J., Wang, O., Niklaus, S., Chen, S., Liu, Y., & Shen, C. *Towards Accurate Reconstruction of 3D Scene Shape From A Single Monocular Image.* IEEE TPAMI, 2023.

[3] Asafa, G. F., Ren, S., Mamun, S. S., & Gobena, K. A. *DepthCloud2Point: Depth Maps and Initial Point for 3D Point Cloud Reconstruction from a Single Image.* Electronics (MDPI), 2025.

[4] ScanNet: Dai, A. et al. *ScanNet: Richly-annotated 3D Reconstructions of Indoor Scenes.*

[5] NYU Depth V2: Silberman, N. et al. *Indoor Segmentation and Support Inference from RGBD Images.*
