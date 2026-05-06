from import_libraries import *

# base_model.py
from base_model import ModelBase

# dataloader.py
from dataloader import MHDWindowDataset, make_temporal_dataloader, compute_normalization_stats

# diffusion.py
from diffusion import (
    DiffusionScheduler,
    DiffusionWrapper,
    EMA,
    ddpm_sample,
    train,
)

import os
import torch
import glob
from torch.utils.data import DataLoader, ConcatDataset


def main():

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")

    # ── Config ────────────────────────────────────────────────────────────────
    # Memory optimized for RTX 5070 8GB VRAM
    ch       = 32           # reduced from 32
    ch_mult  = (1, 2, 4)    # 3 levels, bottleneck at 16^3
    node_dim = 64           # reduced from 128
    n_cond   = 3
    run_name = 'plasma_full'

    # ── Data ──────────────────────────────────────────────────────────────────
    local_train = sorted(glob.glob(r"C:/mhd_data/data/train/*.hdf5"))
    local_val   = sorted(glob.glob(r"C:/mhd_data/data/valid/*.hdf5"))

    print(f"Train files: {len(local_train)}")
    print(f"Val files:   {len(local_val)}")

    # ── Normalization stats ───────────────────────────────────────────────────
    # After first run paste printed stats here to skip recomputation:
    # stats = {'mean': [...], 'std': [...]}
    stats = compute_normalization_stats(
        local_path     = local_train[0],
        use_hf         = False,
        n_sample_trajs = 2,
        n_sample_times = 5,
    )
    print(f"Stats: {stats}")

    # ── Full training set — all files, all trajectories ───────────────────────
    train_loader = DataLoader(
        ConcatDataset([
            MHDWindowDataset(
                hf_path          = path,
                n_cond           = n_cond,
                stats            = stats,
                max_trajectories = 8,       # all trajectories per file
                verbose          = False,   # suppress per-file prints
            )
            for path in local_train         # all train files
        ]),
        batch_size  = 2,                    # small batch — saves VRAM
        shuffle     = True,
        num_workers = 0,
        pin_memory  = True,
        drop_last   = True,
    )

    # ── Full validation set ───────────────────────────────────────────────────
    val_loader = DataLoader(
        ConcatDataset([
            MHDWindowDataset(
                hf_path          = path,
                n_cond           = n_cond,
                stats            = stats,
                max_trajectories = 2,       # fewer val trajectories
                verbose          = False,
            )
            for path in local_val           # all val files
        ]),
        batch_size  = 2,
        shuffle     = False,
        num_workers = 0,
        pin_memory  = True,
        drop_last   = True,
    )

    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches:   {len(val_loader)}")

    # ── Model — memory optimized ───────────────────────────────────────────────
    base_model = ModelBase(
        ch            = ch,
        ch_mult       = ch_mult,
        node_dim      = node_dim,
        edge_dim      = 32,         # reduced from 64
        in_channels   = 4,
        out_channels  = 4,
        resolution    = 64,
        k             = 16,
        num_mp_layers = 2,          # reduced from 4
    ).to(device)

    # ── Resume — check for existing checkpoint ────────────────────────────────
    latest_path = f"{run_name}_latest.pt"
    start_epoch = 0
    resume_ckpt = None

    if os.path.exists(latest_path):
        print(f"\nCheckpoint found: {latest_path}")
        ckpt        = torch.load(latest_path, map_location=device)
        start_epoch = ckpt['epoch'] + 1
        resume_ckpt = latest_path
        print(f"Will resume from epoch {start_epoch}\n")
    else:
        print(f"\nNo checkpoint found — starting from scratch\n")

    # ── Train ─────────────────────────────────────────────────────────────────
    diffusion, ema_model = train(
        base_model   = base_model,
        train_loader = train_loader,
        val_loader   = val_loader,
        n_epochs     = 300,
        lr           = 1e-4,
        device       = device,
        run_name     = run_name,
        save_every   = 25,
        sample_every = 100,
        start_epoch  = start_epoch,    # resume from correct epoch
        resume_ckpt  = resume_ckpt,    # load weights + optimizer state
    )

    print("Training done")


if __name__ == '__main__':
    main()