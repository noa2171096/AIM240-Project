import copy
import logging
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm
import math
import os


# ─────────────────────────────────────────────────────────────────────────────
# 1. EMA
# ─────────────────────────────────────────────────────────────────────────────

class EMA:
    """
    Exponential Moving Average of model weights.
    Maintains a shadow copy of the model with smoothed weights.
    EMA model produces better samples than raw trained model.

    Usage:
        ema       = EMA(beta=0.995)
        ema_model = copy.deepcopy(model).eval().requires_grad_(False)

        # After each optimizer step:
        ema.step_ema(ema_model, model)

        # For sampling — use ema_model not model
    """
    def __init__(self, beta: float = 0.995):
        self.beta = beta
        self.step = 0

    def update_model_average(self, ema_model: nn.Module, model: nn.Module):
        for ema_param, param in zip(ema_model.parameters(), model.parameters()):
            ema_param.data = self.beta * ema_param.data + (1 - self.beta) * param.data

    def step_ema(
        self,
        ema_model:       nn.Module,
        model:           nn.Module,
        step_start_ema:  int = 2000,
    ):
        """
        Update EMA weights.
        For first step_start_ema steps, just copy model weights — EMA
        needs a warm start before it's meaningful.
        """
        if self.step < step_start_ema:
            ema_model.load_state_dict(model.state_dict())
            self.step += 1
            return
        self.update_model_average(ema_model, model)
        self.step += 1


# ─────────────────────────────────────────────────────────────────────────────
# 2. Noise Schedule
# ─────────────────────────────────────────────────────────────────────────────

class DiffusionScheduler(nn.Module):
    """
    Linear beta noise schedule + forward diffusion utilities.

    Buffers registered so tensors move to GPU with .to(device).

    Args:
        noise_steps: total diffusion timesteps T
        beta_start:  starting beta value
        beta_end:    ending beta value
    """
    def __init__(
        self,
        noise_steps: int   = 1000,
        beta_start:  float = 1e-4,
        beta_end:    float = 0.02,
    ):
        super().__init__()
        self.noise_steps = noise_steps

        beta      = torch.linspace(beta_start, beta_end, noise_steps)
        alpha     = 1.0 - beta
        alpha_hat = torch.cumprod(alpha, dim=0)

        # Register as buffers — auto GPU, saved in state_dict
        self.register_buffer('beta',      beta)
        self.register_buffer('alpha',     alpha)
        self.register_buffer('alpha_hat', alpha_hat)

    def noise_images(
        self,
        x:     torch.Tensor,          # (B, C, D, H, W) clean field
        t:     torch.Tensor,          # (B,) integer timesteps
        noise: torch.Tensor = None,
    ) -> tuple:
        """
        Forward diffusion: add noise to x at timestep t.
        Returns (x_noisy, noise).
        """
        if noise is None:
            noise = torch.randn_like(x)

        sqrt_alpha_hat           = torch.sqrt(self.alpha_hat[t])[:, None, None, None, None]
        sqrt_one_minus_alpha_hat = torch.sqrt(1 - self.alpha_hat[t])[:, None, None, None, None]

        x_noisy = sqrt_alpha_hat * x + sqrt_one_minus_alpha_hat * noise
        return x_noisy, noise

    def sample_timesteps(self, n: int) -> torch.Tensor:
        """Sample n random timesteps uniformly from [1, noise_steps)."""
        return torch.randint(low=1, high=self.noise_steps, size=(n,))

    def snr(self, t: torch.Tensor) -> torch.Tensor:
        """
        Signal-to-noise ratio at timestep t.
        Used for Min-SNR-gamma loss weighting.
        """
        ab = self.alpha_hat[t]
        return ab / (1 - ab + 1e-8)                       # FIXED: was nested in sample()


# ─────────────────────────────────────────────────────────────────────────────
# 3. Diffusion Wrapper
# ─────────────────────────────────────────────────────────────────────────────

class DiffusionWrapper(nn.Module):
    """
    Wraps base model with temporal conditioning for diffusion training.

    Conditioning: n_cond prior frames stacked along channel dim.
    condition: (B, n_cond*in_ch, D, H, W) — e.g. (B, 12, 64, 64, 64) for n_cond=3
    target:    (B, in_ch,        D, H, W) — e.g. (B,  4, 64, 64, 64)

    Only the target frame is noised. Condition is always clean.

    Args:
        base_model:   PlasmaGraphUNet — never modified
        schedule:     DiffusionScheduler
        in_channels:  physical fields per frame (e.g. 4)
        n_cond:       conditioning frames (e.g. 3)
        cfg_dropout:  condition dropout rate for classifier-free guidance
    """
    def __init__(
        self,
        base_model:  nn.Module,
        schedule:    DiffusionScheduler,
        in_channels: int   = 4,
        n_cond:      int   = 3,
        cfg_dropout: float = 0.1,
    ):
        super().__init__()                                 # FIXED: was missing nn.Module
        self.model       = base_model
        self.schedule    = schedule
        self.in_channels = in_channels
        self.n_cond      = n_cond
        self.cfg_dropout = cfg_dropout

        cond_channels = n_cond * in_channels               # 3 * 4 = 12

        # 1x1x1 conv: project [noisy_target | condition] → in_channels
        self.cond_proj = nn.Conv3d(
            in_channels + cond_channels,                   # 4 + 12 = 16
            in_channels,                                   # 4
            kernel_size=1,
        )

        # Null condition for CFG — zeros, broadcasts over B and spatial dims
        self.register_buffer(
            'null_cond',
            torch.zeros(1, cond_channels, 1, 1, 1),
        )

    def forward(
        self,
        x0:         torch.Tensor,    # (B, 4,  64, 64, 64) clean target frame
        condition:  torch.Tensor,    # (B, 12, 64, 64, 64) prior frames
    ) -> torch.Tensor:
        """Training forward — returns SNR-weighted loss."""
        B = x0.shape[0]
        t = self.schedule.sample_timesteps(B).to(x0.device)

        # Noise TARGET frame only — condition stays clean
        x_noisy, noise = self.schedule.noise_images(x0, t)

        # Prepare conditioned input with CFG dropout
        model_input = self._prepare_input(x_noisy, condition, training=True)

        # Predict noise
        pred_noise = self.model(model_input, t)

        return self._snr_loss(pred_noise, noise, t)

    def predict(
        self,
        x_noisy:        torch.Tensor,    # (B, 4, 64, 64, 64)
        t:              torch.Tensor,    # (B,)
        condition:      torch.Tensor,    # (B, 12, 64, 64, 64)
        cfg_scale:      float = 1.0,
    ) -> torch.Tensor:
        """Single denoising step — used inside sampling loop."""
        if cfg_scale <= 1.0:
            model_input = self._prepare_input(x_noisy, condition, training=False)
            return self.model(model_input, t)

        # Classifier-free guidance — two forward passes
        B, _, D, H, W = x_noisy.shape

        # Conditional
        cond_input = self._prepare_input(x_noisy, condition, training=False)
        eps_cond    = self.model(cond_input, t)

        # Unconditional — null condition
        null        = self.null_cond.expand(B, -1, D, H, W)
        uncond_input = self._prepare_input(x_noisy, null, training=False)
        eps_uncond   = self.model(uncond_input, t)

        # Interpolate — cfg_scale=1 → pure conditional, >1 → stronger conditioning
        eps_guided = torch.lerp(eps_uncond, eps_cond, cfg_scale)

        return eps_guided

    # ── Helpers ───────────────────────────────────────────────────────────

    def _prepare_input(
        self,
        x_noisy:   torch.Tensor,    # (B, 4,  D, H, W)
        condition: torch.Tensor,    # (B, 12, D, H, W)
        training:  bool,
    ) -> torch.Tensor:              # (B, 4,  D, H, W)
        """
        Concatenate condition to noisy field then project back to in_channels.
        CFG dropout: randomly replace condition with zeros during training.
        """
        B, _, D, H, W = x_noisy.shape

        # CFG dropout
        if training and torch.rand(1).item() < self.cfg_dropout:
            condition = self.null_cond.expand(B, -1, D, H, W)

        x_cat = torch.cat([x_noisy, condition], dim=1)    # (B, 16, D, H, W)
        return self.cond_proj(x_cat)                       # (B, 4,  D, H, W)
                                                           # FIXED: cond_proj applied here

    def _snr_loss(
        self,
        pred:   torch.Tensor,
        target: torch.Tensor,
        t:      torch.Tensor,
        gamma:  float = 5.0,
    ) -> torch.Tensor:
        """Min-SNR-gamma weighted MSE loss."""
        snr    = self.schedule.snr(t)
        weight = torch.minimum(snr, torch.full_like(snr, gamma)) / snr
        weight = weight[:, None, None, None, None]
        return (weight * (pred - target) ** 2).mean()


# ─────────────────────────────────────────────────────────────────────────────
# 4. DDPM Sampling
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def ddpm_sample(
    diffusion:   DiffusionWrapper,
    shape:       tuple,              # (B, C, D, H, W)
    condition:   torch.Tensor,       # (B, n_cond*C, D, H, W)
    cfg_scale:   float = 1.0,
    device:      str   = 'cuda',
) -> torch.Tensor:
    """Full DDPM reverse diffusion — T denoising steps."""
    schedule = diffusion.schedule
    B        = shape[0]
    x        = torch.randn(shape, device=device)

    for i in tqdm(reversed(range(1, schedule.noise_steps)),
                  desc='DDPM sampling', total=schedule.noise_steps - 1):

        t         = torch.full((B,), i, dtype=torch.long, device=device)
        alpha     = schedule.alpha[i]
        alpha_hat = schedule.alpha_hat[i]
        beta      = schedule.beta[i]

        # Predict noise
        predicted_noise = diffusion.predict(
            x, t, condition, cfg_scale
        )

        # Reconstruct x0 estimate then compute posterior mean
        # Standard DDPM update formula
        x = (1 / alpha.sqrt()) * (
            x - ((1 - alpha) / (1 - alpha_hat).sqrt()) * predicted_noise
        )

        # Add noise — except at final step
        if i > 1:
            x = x + beta.sqrt() * torch.randn_like(x)

    # Clamp to valid range
    x = x.clamp(-1, 1)
    return x


# ─────────────────────────────────────────────────────────────────────────────
# 5. Training Loop
# ─────────────────────────────────────────────────────────────────────────────
def get_lr_scheduler(
    optimizer:    torch.optim.Optimizer,
    warmup_steps: int,
    total_steps:  int,
):
    """
    Linear warmup then cosine decay.

    warmup_steps: loss is unstable early — ramp lr up gradually
    total_steps:  lr decays to ~0 by end of training

    Example with n_epochs=300, 48 batches/epoch:
        total_steps  = 14,400
        warmup_steps = 720  (5% warmup)
    """
    def lr_lambda(step):
        if step < warmup_steps:
            # Linear warmup: 0 → 1
            return step / max(warmup_steps, 1)
        # Cosine decay: 1 → ~0
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.5 * (1 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

def train(
    base_model:   nn.Module,
    train_loader,
    val_loader,
    n_epochs:     int   = 300,
    lr:           float = 1e-4,
    cfg_scale:    float = 1.0,
    device:       str   = 'cuda',
    run_name:     str   = 'plasma',
    save_every:   int   = 10,
    sample_every: int   = 20,
    start_epoch =0,
    resume_ckpt = None,
):
    """
    Training loop for conditional plasma diffusion model.

    Args:
        base_model:   ModelBase — graph handled internally
        train_loader: returns (condition, target)
        val_loader:   returns (condition, target)
        n_epochs:     training epochs
        lr:           base learning rate
        cfg_scale:    guidance scale for periodic sampling
        device:       'cuda' or 'cpu'
        run_name:     checkpoint filename prefix
        save_every:   save numbered checkpoint every N epochs
        sample_every: run ddpm_sample every N epochs (expensive — keep high)
    """
    # ── Build diffusion model ──────────────────────────────────────────────
    schedule  = DiffusionScheduler(noise_steps=1000).to(device)
    diffusion = DiffusionWrapper(
        base_model  = base_model,
        schedule    = schedule,
        in_channels = 4,
        n_cond      = 3,
        cfg_dropout = 0.1,
    ).to(device)

    optimizer = optim.AdamW(
        diffusion.parameters(),
        lr           = 1e-4,
        weight_decay = 1e-4,
    ) #lr = lr

    total_steps  = n_epochs * len(train_loader)
    warmup_steps = total_steps // 20
    scheduler    = get_lr_scheduler(optimizer, warmup_steps, total_steps)
    scaler = torch.amp.GradScaler('cuda')

    ema       = EMA(beta=0.995)
    ema_model = copy.deepcopy(diffusion).eval().requires_grad_(False).to(device)

    best_val_loss = float('inf')

    #--- Resume Args -------------------------------------------------
    if resume_ckpt and os.path.exists(resume_ckpt):
        ckpt = torch.load(resume_ckpt, map_location=device)
        diffusion.load_state_dict(ckpt['model_state'])
        ema_model.load_state_dict(ckpt['ema_state'])
        optimizer.load_state_dict(ckpt['optimizer_state'])
        scheduler.load_state_dict(ckpt['scheduler_state'])
        print(f"Loaded checkpoint — epoch {start_epoch}")

    # ── Training loop ──────────────────────────────────────────────────────
    for epoch in range(start_epoch, n_epochs):

        # ── Train ─────────────────────────────────────────────────────────
        diffusion.train()
        train_loss = 0.0
        pbar       = tqdm(train_loader, desc=f"Epoch {epoch}/{n_epochs}")

        for condition, target in pbar:
            condition = condition.to(device, non_blocking=True)
            target    = target.to(device,    non_blocking=True)

            optimizer.zero_grad()

            with torch.amp.autocast('cuda'):
                loss = diffusion(target, condition)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(diffusion.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            ema.step_ema(ema_model, diffusion)

            train_loss += loss.item()
            pbar.set_postfix(
                loss = f"{loss.item():.4f}",
                lr   = f"{scheduler.get_last_lr()[0]:.2e}",
            )

        avg_train = train_loss / len(train_loader)

        # ── Validate ──────────────────────────────────────────────────────
        diffusion.eval()
        val_loss = 0.0

        with torch.no_grad():
            for condition, target in val_loader:
                condition = condition.to(device, non_blocking=True)
                target    = target.to(device,    non_blocking=True)
                with torch.cuda.amp.autocast():
                    val_loss += diffusion(target, condition).item()

        avg_val = val_loss / len(val_loader)

        print(f"Epoch {epoch}: train={avg_train:.4f}  "
              f"val={avg_val:.4f}  "
              f"lr={scheduler.get_last_lr()[0]:.2e}")

        # ── Checkpoint ────────────────────────────────────────────────────
        is_best = avg_val < best_val_loss
        if is_best:
            best_val_loss = avg_val

        ckpt = {
            'epoch':           epoch,
            'train_loss':      avg_train,
            'val_loss':        avg_val,
            'best_val_loss':   best_val_loss,
            'model_state':     diffusion.state_dict(),
            'ema_state':       ema_model.state_dict(),
            'optimizer_state': optimizer.state_dict(),
            'scheduler_state': scheduler.state_dict(),
        }

        # Always save latest — safe resume point
        torch.save(ckpt, f"{run_name}_latest.pt")

        if is_best:
            torch.save(ckpt, f"{run_name}_best.pt")
            print(f"  ✓ Best model saved (val={best_val_loss:.4f})")

        if epoch % save_every == 0:
            torch.save(ckpt, f"{run_name}_epoch{epoch}.pt")

        # ── Periodic sampling ─────────────────────────────────────────────
        if epoch % sample_every == 0 and epoch > 0:
            diffusion.eval()
            with torch.no_grad():
                sample = ddpm_sample(
                    diffusion = ema_model,
                    shape     = (1, 4, 64, 64, 64),
                    condition = condition[:1],
                    cfg_scale = cfg_scale,
                    device    = device,
                )
            print(f"  Sample range: [{sample.min():.3f}, {sample.max():.3f}]  "
                  f"std: {sample.std():.3f}")

    print("Training complete.")
    return diffusion, ema_model


@torch.no_grad()
def ddpm_sample_inference(
    diffusion:  DiffusionWrapper,
    shape:      tuple,
    condition:  torch.Tensor,
    cfg_scale:  float = 1.0,
    device:     str   = 'cuda',
    steps:      int   = 200,       # default fewer steps for inference
) -> torch.Tensor:
    """
    DDPM sampling for inference — adjustable steps.
    Identical to ddpm_sample but steps freely configurable.
    Training model unchanged.
    """
    schedule  = diffusion.schedule
    B         = shape[0]
    x         = torch.randn(shape, device=device)

    timesteps = torch.linspace(
        schedule.noise_steps - 1, 1, steps, dtype=torch.long
    ).tolist()

    for i in tqdm(timesteps, desc=f'DDPM inference ({steps} steps)'):
        t         = torch.full((B,), i, dtype=torch.long, device=device)
        alpha     = schedule.alpha[i]
        alpha_hat = schedule.alpha_hat[i]
        beta      = schedule.beta[i]

        predicted_noise = diffusion.predict(x, t, condition, cfg_scale)

        x = (1 / alpha.sqrt()) * (
            x - ((1 - alpha) / (1 - alpha_hat).sqrt()) * predicted_noise
        )

        if i > 1:
            x = x + beta.sqrt() * torch.randn_like(x)

    return x.clamp(-1, 1)