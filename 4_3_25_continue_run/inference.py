"""
Synthesize the next plasma frame from real conditioning frames,
then visualize the density field in 2D and 3D.
"""
import sys
import os

# Add parent folder to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
import h5py
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# ── Your project imports ──────────────────────────────────────────────────────
from base_model import ModelBase
from diffusion import DiffusionScheduler, DiffusionWrapper, ddpm_sample_inference
from dataloader import compute_normalization_stats

from tqdm import tqdm

@torch.no_grad()
def ddim_sample_inference(
    diffusion,
    shape,
    condition,
    cfg_scale  = 1.0,
    steps      = 50,
    eta        = 0.0,       # 0=deterministic, 1=stochastic like DDPM
    device     = 'cuda',
):
    """
    DDIM inference — better quality than DDPM at same step count.
    steps=50 gives similar quality to DDPM steps=500+
    eta=0.0 is fully deterministic — same condition always same output
    eta=1.0 recovers stochastic DDPM behavior
    """
    schedule  = diffusion.schedule
    B         = shape[0]
    x         = torch.randn(shape, device=device)

    # Evenly spaced timesteps from T-1 down to 0
    timesteps = torch.linspace(
        schedule.noise_steps - 1, 0, steps, dtype=torch.long
    ).tolist()

    for i, t_val in enumerate(tqdm(timesteps, desc=f'DDIM ({steps} steps)')):
        t_val   = int(t_val)
        t       = torch.full((B,), t_val, dtype=torch.long, device=device)
        ab      = schedule.alpha_hat[t_val]
        ab_prev = schedule.alpha_hat[int(timesteps[i+1])] \
                  if i < len(timesteps) - 1 \
                  else torch.tensor(1.0, device=device)

        # Predict noise
        eps_pred = diffusion.predict(x, t, condition, cfg_scale)

        # Reconstruct x0 estimate
        x0_pred  = (x - (1 - ab).sqrt() * eps_pred) / ab.sqrt()
        x0_pred  = x0_pred.clamp(-1, 1)

        # DDIM update
        sigma = eta * ((1 - ab_prev) / (1 - ab) * (1 - ab / ab_prev)).clamp(min=0).sqrt()
        noise = torch.randn_like(x) if eta > 0 else torch.zeros_like(x)

        x = (
            ab_prev.sqrt()                            * x0_pred +
            (1 - ab_prev - sigma**2).clamp(min=0).sqrt() * eps_pred +
            sigma                                     * noise
        )

    return x.clamp(-1, 1)


# ── Config (must match training) ──────────────────────────────────────────────
device     = "cuda" if torch.cuda.is_available() else "cpu"
n_cond     = 3
in_ch      = 4
resolution = 64
checkpoint = r"C:\Users\noahm\PlasmaDiffusionModel\4_3_25_continue_run\plasma_full_best.pt"         # <-- your .pt file
data_file  = r"C:/mhd_data/data/valid/MHD_Ma_0.7_Ms_0.5.hdf5"  # <-- any HDF5 file

# ── 1. Rebuild model ─────────────────────────────────────────────────────────
base_model = ModelBase(
    ch            = 32,
    ch_mult       = (1, 2, 4),
    node_dim      = 64,           # match training
    edge_dim      = 32,           # match training
    in_channels   = 4,
    out_channels  = 4,
    resolution    = 64,
    k             = 16,
    num_mp_layers = 2,            # match training
).to(device)

schedule  = DiffusionScheduler(noise_steps=1000).to(device)
diffusion = DiffusionWrapper(
    base_model=base_model, schedule=schedule,
    in_channels=4, n_cond=3, cfg_dropout=0.1,
).to(device)

# ── 2. Load checkpoint (using EMA weights — better quality) ──────────────────
ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
diffusion.load_state_dict(ckpt["ema_state"])
diffusion.eval()
print(f"Loaded checkpoint: epoch {ckpt['epoch']}, val_loss={ckpt['val_loss']:.4f}")

# ── 3. Load real conditioning frames from HDF5 ──────────────────────────────
#    We need n_cond consecutive frames to condition on.
#    Frames are: density (64,64,64) + magnetic_field (64,64,64,3) = 4 channels

# Compute stats (same as training) for normalization
stats = compute_normalization_stats(
    local_path=data_file, use_hf=False,
    n_sample_trajs=2, n_sample_times=5,
)
field_mean = np.array(stats["mean"], dtype=np.float32)  # (4,)
field_std  = np.array(stats["std"],  dtype=np.float32)  # (4,)

traj_idx = 0       # which trajectory
start_t  = 55      # starting time index — frames [10, 11, 12] condition → predict 13

with h5py.File(data_file, "r") as f:
    cond_frames = []
    for t in range(start_t, start_t + n_cond):
        rho = np.array(f["t0_fields"]["density"][traj_idx, t],       dtype=np.float32)
        mag = np.array(f["t1_fields"]["magnetic_field"][traj_idx, t], dtype=np.float32)
        rho = rho[..., None]                              # (64,64,64,1)
        frame = np.concatenate([rho, mag], axis=-1)       # (64,64,64,4)

        # Normalize (same as training)
        frame = (frame - field_mean) / (field_std + 1e-8)

        # (64,64,64,4) → (4,64,64,64) for PyTorch
        frame = np.transpose(frame, (3, 0, 1, 2))
        cond_frames.append(frame)

    # Also load the REAL next frame (frame 13) for comparison
    rho_gt = np.array(f["t0_fields"]["density"][traj_idx, start_t + n_cond], dtype=np.float32)

# Stack conditioning: (n_cond, 4, 64, 64, 64) → (1, n_cond*4, 64, 64, 64)
condition = np.concatenate(cond_frames, axis=0)            # (12, 64, 64, 64)
condition = torch.tensor(condition, dtype=torch.float32).unsqueeze(0).to(device)

print(f"Condition shape: {condition.shape}")  # (1, 12, 64, 64, 64)

# ── 4. Synthesize next frame ─────────────────────────────────────────────────
'''
print("Running DDPM sampling (1000 steps)...")
with torch.no_grad():
    sample = ddpm_sample_inference(
        diffusion=diffusion,
        shape=(1, in_ch, resolution, resolution, resolution),
        condition=condition,
        cfg_scale=3.0,        # try 1.0–3.0
        steps= 600,
        device=device,
    ) '''

print("Running with DDIM...")
with torch.no_grad():
    sample = ddim_sample_inference(
    diffusion = diffusion,
    shape     = (1, 4, 64, 64, 64),
    condition = condition,
    cfg_scale = 3.0,
    steps     = 25,      # try 50, 100, 200 and compare
    eta       = 0.0,     # deterministic
    device    = device,)

# Unnormalize the predicted density (channel 0)
sample_np = sample[0].cpu().numpy()                        # (4, 64, 64, 64)
pred_density = sample_np[0] * (field_std[0] + 1e-8) + field_mean[0]  # (64, 64, 64)

print(f"Predicted density range: [{pred_density.min():.4f}, {pred_density.max():.4f}]")
print(f"Ground truth range:      [{rho_gt.min():.4f}, {rho_gt.max():.4f}]")

mid = resolution // 2

fig, axes = plt.subplots(2, 3, figsize=(15, 9))

# Predicted slices
for ax, axis, label in zip(axes[0], [0, 1, 2], ["XY (z=mid)", "XZ (y=mid)", "YZ (x=mid)"]):
    slc = np.take(pred_density, mid, axis=axis)
    im = ax.imshow(slc, cmap="inferno", origin="lower")
    ax.set_title(f"Predicted — {label}")
    ax.axis("off")
    plt.colorbar(im, ax=ax, shrink=0.8)

# Ground truth slices
for ax, axis, label in zip(axes[1], [0, 1, 2], ["XY (z=mid)", "XZ (y=mid)", "YZ (x=mid)"]):
    slc = np.take(rho_gt, mid, axis=axis)
    im = ax.imshow(slc, cmap="inferno", origin="lower")
    ax.set_title(f"Ground Truth — {label}")
    ax.axis("off")
    plt.colorbar(im, ax=ax, shrink=0.8)

plt.suptitle("Density: Predicted vs Ground Truth", fontsize=14)
plt.tight_layout()
plt.savefig("density_2d_comparison.png", dpi=150)
plt.show()

# ── 3D isosurface — shared threshold and color scale ──────────────────────
fig = plt.figure(figsize=(14, 6))

# Shared threshold and color scale for 3D too
threshold = np.percentile(np.concatenate([pred_density.flatten(),
                                          rho_gt.flatten()]), 30)
vmin_3d   = min(pred_density[pred_density > threshold].min(),
                rho_gt[rho_gt > threshold].min())
vmax_3d   = max(pred_density[pred_density > threshold].max(),
                rho_gt[rho_gt > threshold].max())

for idx, (data, title) in enumerate([(pred_density, "Predicted"),
                                     (rho_gt,       "Ground Truth")]):
    ax = fig.add_subplot(1, 2, idx + 1, projection="3d")

    z, y, x = np.where(data > threshold)
    vals    = data[data > threshold]

    sc = ax.scatter(
        x, y, z,
        c     = vals,
        cmap  = "inferno",
        vmin  = vmin_3d,
        vmax  = vmax_3d,
        s     = 1,
        alpha = 0.3,
    )
    ax.set_title(f"{title} Density (>{np.percentile(data,30):.2f})")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    plt.colorbar(sc, ax=ax, shrink=0.6)

plt.tight_layout()
plt.savefig("density_3d_comparison.png", dpi=150)
plt.show()

print("Done! Saved density_2d_comparison.png and density_3d_comparison.png")


# Run all three and compare
real_condition  = condition[:1]                      # real prior frames
null_condition  = torch.zeros_like(real_condition)   # no prior frames
half_condition  = real_condition * 0.5               # weakened condition

sample_cond   = ddim_sample_inference( diffusion = diffusion,
    shape = (1, 4, 64, 64, 64), condition=real_condition,  cfg_scale=1.0)
sample_uncond = ddim_sample_inference(diffusion = diffusion,
    shape     = (1, 4, 64, 64, 64), condition=null_condition,  cfg_scale=1.0)
sample_guided = ddim_sample_inference(diffusion = diffusion,
    shape     = (1, 4, 64, 64, 64), condition=real_condition,  cfg_scale=3.0)

# Plot all three side by side
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
for ax, s, title in zip(axes,
    [sample_uncond, sample_cond, sample_guided],
    ['Unconditional', 'Conditional', 'CFG guided (scale=3)']):
    ax.imshow(s[0, 0, 32].cpu().numpy(), cmap='inferno')
    ax.set_title(title)
    ax.axis('off')
plt.savefig('condition_comparison.png', dpi=150)
plt.show()
