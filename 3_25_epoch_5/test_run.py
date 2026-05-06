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

import torch
import glob
from torch.utils.data import DataLoader

def main():

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # device = torch.device("cpu")   instead of "cuda"
    print(f"Device: {device}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")

    # ── Config ────────────────────────────────────────────────────────────────
    ch       = 32
    ch_mult  = (1, 2, 4, 8)
    node_dim = 256
    n_cond   = 3

    #---Data--------------------------------------------------------------------

    local_train = sorted(glob.glob(r"C:/mhd_data/data/train/*.hdf5"))
    local_val   = sorted(glob.glob(r"C:/mhd_data/data/valid/*.hdf5"))

    # ── Normalization stats ───────────────────────────────────────────────────
    # Compute from first file
    stats = compute_normalization_stats(
        local_path        = local_train[0], use_hf=False,
        n_sample_trajs = 2,
        n_sample_times = 5,
    )
    print(f"Stats: {stats}")

    # ── Data — small subset ───────────────────────────────────────────────────
    # 1 file, 2 trajectories, ~97 windows each = ~194 training samples
    train_loader = DataLoader(
        MHDWindowDataset(
            hf_path          = local_train[0],
            n_cond           = n_cond,
            stats            = stats,
            max_trajectories = 2,
        ),
        batch_size  = 4,         
        shuffle     = True,
        num_workers = 0,
        pin_memory  = True,
        drop_last   = True,
    )

    val_loader = DataLoader(
        MHDWindowDataset(
            hf_path          = local_val[0],
            n_cond           = n_cond,
            stats            = stats,
            max_trajectories = 1,
        ),
        batch_size  = 4,
        shuffle     = False,
        num_workers = 0,
        pin_memory  = True,
        drop_last   = True,
    )

    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches:   {len(val_loader)}")

    # ── Model ─────────────────────────────────────────────────────────────────
    base_model = ModelBase(
        ch           = ch,
        ch_mult      = ch_mult,
        node_dim     = node_dim,
        edge_dim     = 64,
        in_channels  = 4,
        out_channels = 4,
        resolution   = 64,
        k            = 16,
    ).to(device)

    # ── Train 5 epochs ────────────────────────────────────────────────────────
    diffusion, ema_model = train(
        base_model   = base_model,
        train_loader = train_loader,
        val_loader   = val_loader,
        n_epochs     = 300,
        lr           = 1e-4,
        device       = device,
        run_name     = 'plasma_test',
        save_every   = 500,
        sample_every = 999,    # skip sampling during training for speed
    )

    print("Training done")

if __name__ == '__main__':

    main()