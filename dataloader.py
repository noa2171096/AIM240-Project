# Data loader w/ Scanning Window for Conditional Samples

import os
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from huggingface_hub import HfFileSystem
import glob

class MHDWindowDataset(Dataset):
    """
    Sliding window dataset for temporal conditioning.

    Each sample returns:
        condition: frames [t-n_cond, ..., t-1] stacked along channels
                   shape (n_cond * in_channels, 64, 64, 64)
                   e.g. (12, 64, 64, 64) for n_cond=3, in_channels=4
        target:    frame [t]
                   shape (in_channels, 64, 64, 64)
                   e.g. (4, 64, 64, 64)

    Only the target frame is noised during training.
    Condition frames are always clean.

    Args:
        hf_path:          HuggingFace HDF5 file path
        n_cond:           number of conditioning frames (default 3)
        in_channels:      physical fields per frame (default 4)
        normalize:        apply per-channel z-score normalization
        stats:            {'mean': [...], 'std': [...]} per channel
        max_trajectories: cap trajectories for Colab memory
    """
    def __init__(
        self,
        hf_path:          str,
        n_cond:           int   = 3,
        in_channels:      int   = 4,
        normalize:        bool  = True,
        stats:            dict  = None,
        max_trajectories: int   = None,
        verbose: bool = False,
    ):
        self.hf_path     = hf_path
        self.n_cond      = n_cond
        self.in_channels = in_channels
        self.normalize   = normalize
        self.stats       = stats

        # Probe metadata — no data loaded
        if os.path.exists(hf_path):
            # Local file — open directly
            with h5py.File(hf_path, "r") as h5:
                shape = h5["t0_fields"]["density"].shape
        else:
            # Remote HuggingFace path
            fs = HfFileSystem()
            with fs.open(hf_path, "rb") as f:
                with h5py.File(f, "r") as h5:
                    shape = h5["t0_fields"]["density"].shape

        n_traj, n_time = shape[0], shape[1]
        if max_trajectories is not None:
            n_traj = min(n_traj, max_trajectories)

        # Target frame starts at index n_cond
        # (need n_cond frames before it)
        self.samples = [
            (traj_i, t_target)
            for traj_i  in range(n_traj)
            for t_target in range(n_cond, n_time)
        ]

        if verbose:
            print(f"MHDWindowDataset:")
            print(f"  {n_traj} trajectories × {n_time - n_cond} windows")
            print(f"  = {len(self.samples)} total samples")
            print(f"  condition: {n_cond} frames × {in_channels} ch = {n_cond*in_channels} ch")
            print(f"  target:    1 frame  × {in_channels} ch = {in_channels} ch")
            
    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        traj_i, t_target = self.samples[idx]

        # Load target frame
        target = self._load_frame(traj_i, t_target)           # (4, 64, 64, 64)

        # Load n_cond preceding frames — [t-n_cond, ..., t-1]
        cond_frames = [
            self._load_frame(traj_i, t_target - self.n_cond + i)
            for i in range(self.n_cond)
        ]
        condition = torch.cat(cond_frames, dim=0)             # (12, 64, 64, 64)

        # Normalize
        if self.normalize and self.stats is not None:
            target    = self._normalize(target)
            condition = self._normalize_stacked(condition)

        return condition, target

    def _load_frame(self, traj_i: int, t_i: int) -> torch.Tensor:
        """Load a single (trajectory, timestep) frame. Opens/closes HDF5 per call."""

        if os.path.exists(self.hf_path):
            with h5py.File(self.hf_path, "r") as h5:
                rho = h5["t0_fields"]["density"][traj_i, t_i]
                mag = h5["t1_fields"]["magnetic_field"][traj_i, t_i]
        else:
            fs = HfFileSystem()
            with fs.open(self.hf_path, "rb") as f:
                with h5py.File(f, "r") as h5:
                    rho = h5["t0_fields"]["density"][traj_i, t_i]
                    mag = h5["t1_fields"]["magnetic_field"][traj_i, t_i]

        rho = torch.from_numpy(np.array(rho, dtype=np.float32)).unsqueeze(0)
        mag = torch.from_numpy(np.array(mag, dtype=np.float32)).permute(3, 0, 1, 2)
        return torch.cat([rho, mag], dim=0)                    # (4, 64, 64, 64)

    def _normalize(self, x: torch.Tensor) -> torch.Tensor:
        """Per-channel z-score normalization for a single frame."""
        mean = torch.tensor(self.stats["mean"], dtype=x.dtype)
        std  = torch.tensor(self.stats["std"],  dtype=x.dtype)
        return (x - mean[:, None, None, None]) / (std[:, None, None, None] + 1e-8)

    def _normalize_stacked(self, x: torch.Tensor) -> torch.Tensor:
        """
        Per-channel normalization for n_cond stacked frames.
        Stats tile across stacked frames — same normalization per field type
        regardless of which frame it comes from.
        """
        n_repeat = x.shape[0] // self.in_channels
        mean = torch.tensor(self.stats["mean"], dtype=x.dtype).repeat(n_repeat)
        std  = torch.tensor(self.stats["std"],  dtype=x.dtype).repeat(n_repeat)
        return (x - mean[:, None, None, None]) / (std[:, None, None, None] + 1e-8)


def make_temporal_dataloader(
    hf_paths:         list,
    batch_size:       int  = 2,
    n_cond:           int  = 3,
    in_channels:      int  = 4,
    max_traj_per_file: int = 4,
    stats:            dict = None,
    num_workers:      int  = 2,
    shuffle:          bool = True,
) -> DataLoader:
    """
    Build DataLoader for temporal conditioning.
    Concatenates multiple HDF5 files.
    """
    from torch.utils.data import ConcatDataset

    datasets = [
        MHDWindowDataset(
            hf_path          = path,
            n_cond           = n_cond,
            in_channels      = in_channels,
            normalize        = (stats is not None),
            stats            = stats,
            max_trajectories = max_traj_per_file,
        )
        for path in hf_paths
    ]

    dataset   = ConcatDataset(datasets)
    use_cuda  = torch.cuda.is_available()

    return DataLoader(
        dataset,
        batch_size        = batch_size,
        shuffle           = shuffle,
        num_workers       = num_workers,
        pin_memory        = use_cuda,
        prefetch_factor   = 2 if num_workers > 0 else None,
        persistent_workers = (num_workers > 0),
        drop_last         = True,
    )

def compute_normalization_stats(
    hf_path=None, local_path=None, use_hf=True,
    n_sample_trajs: int = 5,
    n_sample_times: int = 10,
) -> dict:
    """
    Estimate per-channel mean and std from a small random sample.
    Run once, then hardcode the result.

    Returns:
        {'mean': [rho_mean, Bx_mean, By_mean, Bz_mean],
         'std':  [rho_std,  Bx_std,  By_std,  Bz_std]}
    """
    import gc

    samples = []

    if use_hf:
        print(f"Computing stats from {hf_path}...")
        fs = HfFileSystem()
        f = fs.open(hf_path, "rb")
        h5 = h5py.File(f, "r")
    else:
        print(f"Computing stats from {local_path}...")
        f = None
        h5 = h5py.File(local_path, "r")

    try:
        n_traj = h5["t0_fields"]["density"].shape[0]
        n_time = h5["t0_fields"]["density"].shape[1]

        traj_idx = np.random.choice(n_traj, min(n_sample_trajs, n_traj), replace=False)
        time_idx = np.random.choice(n_time, min(n_sample_times, n_time), replace=False)

        for ti in traj_idx:
            for t in time_idx:
                rho = np.array(h5["t0_fields"]["density"][ti, t],         dtype=np.float32)
                mag = np.array(h5["t1_fields"]["magnetic_field"][ti, t],   dtype=np.float32)
                rho = rho[..., None]
                combined = np.concatenate([rho, mag], axis=-1)
                samples.append(combined)
    finally:
        h5.close()
        if f is not None:
            f.close()

    samples = np.stack(samples, axis=0)    # (N, 64, 64, 64, 4)

    # Mean and std over all samples + all spatial locations, per channel
    axes = (0, 1, 2, 3)                   # average over N, D, H, W
    mean = samples.mean(axis=axes).tolist()
    std  = samples.std(axis=axes).tolist()

    del samples
    gc.collect()

    print(f"  mean: {[f'{m:.4f}' for m in mean]}")
    print(f"  std:  {[f'{s:.4f}' for s in std]}")
    return {"mean": mean, "std": std}


'''
#### IGNORE #####
print("Hello world")

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
local_train = sorted(glob.glob(r"C:/mhd_data/data/train/*.hdf5"))[:1]
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

def get_memory_bytes(x):

    return x.element_size()*x.nelement()/(1024**3)


condition, target= next(iter(train_loader))

print(condition.shape, target.shape)

print(get_memory_bytes(condition), get_memory_bytes(target))

'''