import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────────────────────
# Utilities
# ─────────────────────────────────────────────────────────────────────────────

def nonlinearity(x):
    return x * torch.sigmoid(x)    # swish / SiLU


def Normalize(in_channels):
    return nn.GroupNorm(
        num_groups=min(32, in_channels),
        num_channels=in_channels,
        eps=1e-6,
        affine=True,
    )


def get_timestep_embedding(timesteps: torch.Tensor, embedding_dim: int) -> torch.Tensor:
    """
    Sinusoidal timestep embedding.
    Matches DDPM implementation.
    timesteps: (B,) integer timesteps
    returns:   (B, embedding_dim)
    """
    assert len(timesteps.shape) == 1

    half_dim = embedding_dim // 2
    emb = math.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float32) * -emb)
    emb = emb.to(device=timesteps.device)
    emb = timesteps.float()[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)

    if embedding_dim % 2 == 1:
        emb = F.pad(emb, (0, 1, 0, 0))

    return emb


# ─────────────────────────────────────────────────────────────────────────────
# DownSample / UpSample
# ─────────────────────────────────────────────────────────────────────────────

class DownSample(nn.Module):
    """
    2x spatial downsampling via strided Conv3d.
    Asymmetric padding to handle even spatial dimensions cleanly.
    in_channels passed explicitly — no global scope dependency.
    """
    def __init__(self, in_channels: int, with_conv: bool = True):
        super().__init__()
        self.with_conv = with_conv
        if with_conv:
            self.conv = nn.Conv3d(
                in_channels, in_channels,
                kernel_size=3, stride=2, padding=0,
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.with_conv:
            # Pad to handle odd spatial dims: (left, right, top, bottom, front, back)
            x = F.pad(x, (0, 1, 0, 1, 0, 1), mode="constant", value=0)
            x = self.conv(x)
        else:
            x = F.avg_pool3d(x, kernel_size=2, stride=2)
        return x


class UpSample(nn.Module):
    """
    2x spatial upsampling via ConvTranspose3d.
    in_channels passed explicitly.
    """
    def __init__(self, in_channels: int, with_conv: bool = True):
        super().__init__()
        self.with_conv = with_conv
        if with_conv:
            self.conv = nn.ConvTranspose3d(
                in_channels, in_channels,
                kernel_size=4, stride=2, padding=1,
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.with_conv:
            x = self.conv(x)
        else:
            x = F.interpolate(x, scale_factor=2.0, mode='nearest')
        return x


# ─────────────────────────────────────────────────────────────────────────────
# ResNet Block
# ─────────────────────────────────────────────────────────────────────────────

class ResnetBlock(nn.Module):
    """
    3D ResNet block with timestep conditioning via linear projection.

    Structure:
        Norm → Nonlin → Conv3d
             ↓ (+ temb projection)
        Norm → Nonlin → Dropout → Conv3d
             ↓ (+ residual shortcut)

    temb is projected and added BETWEEN the two conv layers — standard
    DDPM pattern. Critically fixed for 3D: projection needs [:,:,None,None,None]
    not [:,:,None,None] — Conv3d output has 3 spatial dims not 2.
    """
    def __init__(
        self,
        *,
        in_channels: int,
        out_channels: int = None,
        conv_shortcut: bool = False,
        dropout: float,
        temb_channels: int = 512,
    ):
        super().__init__()
        self.in_channels  = in_channels
        out_channels      = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut

        self.norm1     = Normalize(in_channels)
        self.conv1     = nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.temb_proj = nn.Linear(temb_channels, out_channels)
        self.norm2     = Normalize(out_channels)
        self.dropout   = nn.Dropout(dropout)
        self.conv2     = nn.Conv3d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)

        if in_channels != out_channels:
            if conv_shortcut:
                self.shortcut = nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
            else:
                self.shortcut = nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x: torch.Tensor, temb: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x)
        h = nonlinearity(h)
        h = self.conv1(h)

        # Timestep conditioning — inject between the two conv layers
        # [:, :, None, None, None] broadcasts (B, C) → (B, C, 1, 1, 1)
        # then adds to (B, C, D, H, W) via broadcasting
        # FIXED: original had only 2 None dims, wrong for 3D Conv output
        h = h + self.temb_proj(nonlinearity(temb))[:, :, None, None, None]

        h = self.norm2(h)
        h = nonlinearity(h)
        h = self.dropout(h)
        h = self.conv2(h)

        if self.in_channels != self.out_channels:
            x = self.shortcut(x)

        return x + h


# ─────────────────────────────────────────────────────────────────────────────
# Attention Block
# ─────────────────────────────────────────────────────────────────────────────

class AttnBlock(nn.Module):
    """
    Self-attention block for 3D volumes.

    FIXED from original: original assumed 2D (b,c,h,w).
    3D volumes have shape (B, C, D, H, W) — spatial dims must be
    flattened as D*H*W not just H*W.

    Uses 1x1x1 convolutions to compute Q, K, V projections —
    equivalent to linear projections applied pointwise across spatial locations.

    Complexity: O((D*H*W)²) — expensive at full resolution.
    Only apply at coarse resolutions (e.g. 8³ or 16³) via attn_resolutions.
    """
    def __init__(self, in_channels: int):
        super().__init__()
        self.in_channels = in_channels
        self.norm     = Normalize(in_channels)
        self.q        = nn.Conv3d(in_channels, in_channels, kernel_size=1)
        self.k        = nn.Conv3d(in_channels, in_channels, kernel_size=1)
        self.v        = nn.Conv3d(in_channels, in_channels, kernel_size=1)
        self.proj_out = nn.Conv3d(in_channels, in_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h_ = self.norm(x)
        q  = self.q(h_)
        k  = self.k(h_)
        v  = self.v(h_)

        # Reshape for attention: (B, C, D, H, W) → (B, C, D*H*W)
        B, C, D, H, W = q.shape
        N = D * H * W                              # total spatial locations

        q = q.reshape(B, C, N).permute(0, 2, 1)   # (B, N, C)
        k = k.reshape(B, C, N)                     # (B, C, N)
        v = v.reshape(B, C, N)                     # (B, C, N)

        # Scaled dot-product attention
        attn = torch.bmm(q, k) * (C ** -0.5)      # (B, N, N)
        attn = F.softmax(attn, dim=-1)

        # Attend to values
        h_ = torch.bmm(v, attn.permute(0, 2, 1))  # (B, C, N)
        h_ = h_.reshape(B, C, D, H, W)            # (B, C, D, H, W)

        h_ = self.proj_out(h_)
        return x + h_


# ─────────────────────────────────────────────────────────────────────────────
# Positional Encoder
# ─────────────────────────────────────────────────────────────────────────────

class PositionalEncoder(nn.Module):
    """
    Maps normalized 3D coordinates [-1, 1] → spatial embeddings.
    Data-independent: same output for the same position regardless of
    the input field. Gives graph nodes explicit spatial context.
    """
    def __init__(self, out_dim: int = 128):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(3, 64),
            nn.SiLU(),
            nn.Linear(64, 128),
            nn.SiLU(),
            nn.Linear(128, out_dim),
        )

    def forward(self, pos: torch.Tensor) -> torch.Tensor:
        # pos: (N, 3) in [-1, 1]
        return self.mlp(pos)    # (N, out_dim)


# ─────────────────────────────────────────────────────────────────────────────
# Mesh2Grid Encoder
# ─────────────────────────────────────────────────────────────────────────────

class Mesh2GridEncoder(nn.Module):
    """
    3D Conv UNet Encoder with timestep conditioning and conv→graph interface.

    Encodes (B, in_channels, resolution³) plasma field down to a set of
    graph node features at the bottleneck resolution.

    Args:
        ch:               base channel count
        out_ch:           output channels (unused in encoder, kept for symmetry)
        ch_mult:          channel multipliers per resolution level
                          e.g. (1,2,4,8) → ch, 2ch, 4ch, 8ch
        num_res_blocks:   ResNet blocks per resolution level
        attn_resolutions: spatial resolutions at which to apply attention
                          e.g. [8] → only apply attention at 8³
                          keep small — attention is O(N²) in spatial locations
        dropout:          dropout rate in ResNet blocks
        resamp_with_conv: use learned conv for downsampling (True) or avg pool (False)
        in_channels:      number of input physical fields (e.g. 4 for rho,Bx,By,Bz)
        resolution:       input spatial resolution (assumed cubic, e.g. 64)
        node_dim:         graph node feature dimension
        node_pos:         (N, 3) precomputed node positions in [-1,1]
                          registered as buffer in parent model, passed here
    """
    def __init__(
        self,
        *,
        ch: int,
        out_ch: int,
        ch_mult: tuple       = (1, 2, 4, 8),
        num_res_blocks: int  = 2,
        attn_resolutions: list,
        dropout: float       = 0.0,
        resamp_with_conv: bool = True,
        in_channels: int,
        resolution: int,
        node_dim: int        = 256,
    ):
        super().__init__()
        self.ch               = ch
        self.temb_ch          = ch * 4
        self.num_resolutions  = len(ch_mult)
        self.num_res_blocks   = num_res_blocks
        self.resolution       = resolution
        self.in_channels      = in_channels

        # ── Timestep embedding MLP ─────────────────────────────────────────
        self.temb_dense = nn.ModuleList([
            nn.Linear(ch, self.temb_ch),
            nn.Linear(self.temb_ch, self.temb_ch),
        ])

        # ── Input projection ───────────────────────────────────────────────
        self.conv_in = nn.Conv3d(in_channels, ch, kernel_size=3, stride=1, padding=1)

        # ── Downsampling levels ────────────────────────────────────────────
        curr_res    = resolution
        in_ch_mult  = (1,) + tuple(ch_mult)
        self.down   = nn.ModuleList()

        for i_level in range(self.num_resolutions):
            block     = nn.ModuleList()
            attn      = nn.ModuleList()
            block_in  = ch * in_ch_mult[i_level]
            block_out = ch * ch_mult[i_level]

            for i_block in range(num_res_blocks):
                block.append(ResnetBlock(
                    in_channels=block_in,
                    out_channels=block_out,
                    temb_channels=self.temb_ch,
                    dropout=dropout,
                ))
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(AttnBlock(block_in))

            level = nn.Module()
            level.block = block
            level.attn  = attn

            if i_level != self.num_resolutions - 1:
                level.downsample = DownSample(block_in, resamp_with_conv)
                curr_res = curr_res // 2

            self.down.append(level)

        # curr_res is now the bottleneck resolution
        # block_in is now the bottleneck channel count
        self.bottleneck_ch  = block_in
        self.bottleneck_res = curr_res

        # ── Conv → Graph interface ─────────────────────────────────────────
        # Projects bottleneck conv features (data-dependent) to node_dim
        self.conv_to_node_proj = nn.Linear(self.bottleneck_ch, node_dim)

        # Positional encoder for spatial context (data-independent)
        self.pos_encoder       = PositionalEncoder(out_dim=node_dim)

        # Fuses conv features + positional encoding → node features
        self.node_input_proj   = nn.Linear(node_dim * 2, node_dim)

    def forward(
        self,
        x: torch.Tensor,          # (B, in_channels, resolution, resolution, resolution)
        t: torch.Tensor,          # (B,) diffusion timesteps
        node_pos: torch.Tensor,   # (N, 3) node positions in [-1,1] — from parent buffer
    ):
        """
        Returns:
            node_feats: (B*N, node_dim) — graph node features ready for GATv2
            hs:         list of skip connection tensors for decoder
        """
        B = x.shape[0]
        assert x.shape[2] == x.shape[3] == x.shape[4] == self.resolution, \
            f"Expected resolution {self.resolution}, got {x.shape[2:]}"

        # ── Timestep embedding ─────────────────────────────────────────────
        temb = get_timestep_embedding(t, self.ch)          # (B, ch)
        temb = self.temb_dense[0](temb)
        temb = nonlinearity(temb)
        temb = self.temb_dense[1](temb)                    # (B, temb_ch)

        # ── Downsampling ───────────────────────────────────────────────────
        hs = [self.conv_in(x)]                             # first skip connection

        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks):
                h = self.down[i_level].block[i_block](hs[-1], temb)
                if len(self.down[i_level].attn) > 0:
                    h = self.down[i_level].attn[i_block](h)
                hs.append(h)

            # Downsample between levels (not after last level)
            if i_level != self.num_resolutions - 1:
                hs.append(self.down[i_level].downsample(hs[-1]))

        # hs[-1] is now the bottleneck: (B, bottleneck_ch, bottleneck_res³)

        # ── Conv → Graph ───────────────────────────────────────────────────
        feature_map = hs[-1]                               # (B, C, D, H, W)

        # Construct batch_idx: maps each node to its batch element
        # node_pos is shared across batch — tile it B times
        N         = node_pos.shape[0]
        batch_idx = torch.arange(B, device=x.device).repeat_interleave(N)  # (B*N,)
        node_pos_b = node_pos.repeat(B, 1)                 # (B*N, 3)

        # Sample conv features at node positions via trilinear interpolation
        # Differentiable w.r.t. feature_map values — gradients flow to encoder
        feats = []
        for b in range(B):
            mask  = (batch_idx == b)
            pos_b = node_pos_b[mask]                       # (N, 3)
            n_b   = pos_b.shape[0]

            grid  = pos_b.view(1, 1, 1, n_b, 3)
            fm_b  = feature_map[b].unsqueeze(0)            # (1, C, D, H, W)

            interp = F.grid_sample(
                fm_b, grid,
                mode='bilinear',
                align_corners=True,
                padding_mode='border',
            )                                              # (1, C, 1, 1, N)

            interp = interp.squeeze(0).squeeze(1).squeeze(1).T  # (N, C)
            feats.append(interp)

        feats = torch.cat(feats, dim=0)                    # (B*N, bottleneck_ch)

        # Project conv features to node_dim
        conv_feats = self.conv_to_node_proj(feats)         # (B*N, node_dim)

        # Positional encoding — where is each node in the domain?
        pos_feats  = self.pos_encoder(node_pos_b)          # (B*N, node_dim)

        # Fuse data-dependent + data-independent features
        node_feats = self.node_input_proj(
            torch.cat([conv_feats, pos_feats], dim=-1)     # (B*N, node_dim*2)
        )                                                   # (B*N, node_dim)

        return node_feats, hs    # hs carries skip connections for decoder

    @staticmethod
    def build_graph(pos: torch.Tensor, k: int):
        """
        Pure PyTorch KNN graph + raw edge features.
        Returns edge_index (2, E) and edge_raw (E, 4).
        """
        # ── KNN connectivity ──────────────────────────────────────────
        diff = pos.unsqueeze(0) - pos.unsqueeze(1)    # (N, N, 3)
        dist = diff.norm(dim=-1)                       # (N, N)
        dist.fill_diagonal_(float('inf'))              # exclude self
        _, idx = dist.topk(k, dim=-1, largest=False)  # (N, k)

        N          = pos.shape[0]
        src        = torch.arange(N, device=pos.device).unsqueeze(1).expand(-1, k)
        edge_index = torch.stack([src.reshape(-1), idx.reshape(-1)], dim=0)  # (2, N*k)

        # ── Raw edge features ─────────────────────────────────────────
        src_idx, dst_idx = edge_index
        rel_pos  = pos[dst_idx] - pos[src_idx]        # (E, 3)
        edge_dist = rel_pos.norm(dim=-1, keepdim=True) # (E, 1)
        edge_raw  = torch.cat([rel_pos, edge_dist], dim=-1)  # (E, 4)

        return edge_index, edge_raw


def batch_graph(
    edge_index: torch.Tensor,    # (2, E) — single graph edges
    edge_attr: torch.Tensor,     # (E, edge_dim) — single graph edge features
    B: int,                      # batch size
    N_per: int,                  # nodes per graph (512 for 8³)
    device: torch.device,
) -> tuple:
    """
    Returns:
        edge_index_b: (2, B*E) — batched edge connectivity
        edge_attr_b:  (B*E, edge_dim) — tiled edge features
    """
    offsets = torch.arange(B, device=device) * N_per    # (B,) — [0, 512, 1024, ...]

    ei_list = [edge_index + offsets[b] for b in range(B)]
    ea_list = [edge_attr  for _        in range(B)]

    edge_index_b = torch.cat(ei_list, dim=1)    # (2, B*E)
    edge_attr_b  = torch.cat(ea_list, dim=0)    # (B*E, edge_dim)

    return edge_index_b, edge_attr_b
#------------------------------------------------------------------------------
# Bottle Neck GAT Layer
#------------------------------------------------------------------------------

#GAT Layer

from torch_geometric.nn import GATv2Conv as GATConv

class GAT(torch.nn.Module):
    def __init__(self, node_dim: int, edge_dim: int, heads: int=4 ):
        super().__init__()

        self.conv = GATConv(
            in_channels=node_dim,
            out_channels=node_dim // heads,
            heads=heads,
            edge_dim=edge_dim,
            concat=True,
            add_self_loops=False,
            bias=True,
        )

        self.norm = nn.LayerNorm(node_dim)

    def forward(self, x, edge_index, edge_attr):

        x_upd = self.conv(x, edge_index, edge_attr) #apply convolution

        return self.norm(x_upd+x) #normalize conv + residual

class GraphProcessor(nn.Module):
    """
    Stack of GAT layers with per-layer FFN.
    Replaces single GAT layer in ModelBase.

    Each layer:
        GATv2 attention  — aggregates neighbor features
        FFN              — pointwise transformation per node
        residual + norm  — gradient flow + stability

    num_layers guidelines:
        4:  basic long-range propagation for 8³ grid
        8:  full domain coverage, recommended default
        12: richer representations, diminishing returns beyond this
    """
    def __init__(
        self,
        node_dim:   int,
        edge_dim:   int,
        num_layers: int = 8, #num of bottle neck layers
        heads:      int = 4,
    ):
        super().__init__()

        self.attn_layers = nn.ModuleList([
            GAT(node_dim=node_dim, edge_dim=edge_dim, heads=heads)
            for _ in range(num_layers)
        ])

        self.ffn_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(node_dim, node_dim * 2),
                nn.SiLU(),
                nn.Linear(node_dim * 2, node_dim),
                nn.LayerNorm(node_dim),
            )
            for _ in range(num_layers)
        ])

    def forward(
        self,
        x:          torch.Tensor,    # (B*N, node_dim)
        edge_index: torch.Tensor,    # (2, B*E)
        edge_attr:  torch.Tensor,    # (B*E, edge_dim)
    ) -> torch.Tensor:               # (B*N, node_dim)

        for attn, ffn in zip(self.attn_layers, self.ffn_layers):
            x = attn(x, edge_index, edge_attr)    # attention + residual + norm
            x = x + ffn(x)                        # FFN residual
        return x

#------------------------------------------------------------------------------
# Decoding Layer
#------------------------------------------------------------------------------

class GraphToConv(nn.Module):

  """
  Procedure:

  node_feats: (B*N, node_dim)

  -> proj -> (B*N, conv ch) -> reshape -> (B, conv ch, D , H, W)

  """


  def __init__(self, node_ch: int, conv_ch: int, grid_size: tuple, target_size:tuple = None):
    super().__init__()

    self.D, self.H, self.W = grid_size

    self.proj = nn.Linear(node_ch, conv_ch)


  def forward(self, node_feats: torch.Tensor) -> torch.Tensor:

    D, H, W = self.D, self.H, self.W

    feats = self.proj(node_feats)

    C = feats.shape[1]

    # Calculate batch size B. node_feats is (B*N, node_dim) where N = D*H*W
    B = node_feats.shape[0] // (D * H * W)

    out = feats.view(B, D, H, W, C).permute(0, 4, 1, 2, 3)

    out = out.contiguous()

    return out

class Decoder3D(nn.Module):
    """
    3D ResNet decoder — mirrors the encoder in reverse.

    Starts at bottleneck resolution (8³) and upsamples back to input
    resolution (64³) through num_resolutions levels.

    No skip connections in this version — graph output feeds in directly
    via GraphToConv + F.interpolate. Add skip connections later if needed.

    Channel progression (reversed from encoder):
        bottleneck:  ch * ch_mult[-1]  e.g. 256
        level 0:     ch * ch_mult[-1]  e.g. 256  (bottleneck level)
        level 1:     ch * ch_mult[-2]  e.g. 128
        level 2:     ch * ch_mult[-3]  e.g. 64
        level 3:     ch * ch_mult[-4]  e.g. 32   (output level)

    Args:
        ch:               base channel count (same as encoder)
        ch_mult:          channel multipliers (same as encoder)
        num_res_blocks:   ResNet blocks per level (same as encoder)
        attn_resolutions: resolutions to apply attention (same as encoder)
        dropout:          dropout rate
        resamp_with_conv: use learned ConvTranspose3d for upsampling
        resolution:       target output resolution (e.g. 64)
        out_channels:     number of output physical fields
        temb_ch:          timestep embedding dimension
    """
    def __init__(
        self,
        *,
        ch: int,
        ch_mult: tuple          = (1, 2, 4, 8),
        num_res_blocks: int     = 2,
        attn_resolutions: list  = [8],
        dropout: float          = 0.0,
        resamp_with_conv: bool  = True,
        resolution: int         = 64,
        out_channels: int       = 4,
        temb_ch: int            = 128,
    ):
        super().__init__()
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks  = num_res_blocks

        # Bottleneck resolution — where decoder starts
        bottleneck_res = resolution // (2 ** (self.num_resolutions - 1))
        # e.g. 64 // 2^3 = 8

        # ── Upsampling levels ──────────────────────────────────────────────
        # Start at bottleneck, work UP to full resolution
        # reversed(range(4)) = [3, 2, 1, 0]
        # i_level=3: 8³  → process at bottleneck
        # i_level=2: 8³  → 16³
        # i_level=1: 16³ → 32³
        # i_level=0: 32³ → 64³ (output)

        in_ch_mult = (1,) + tuple(ch_mult)
        curr_res  = bottleneck_res
        self.up   = nn.ModuleList()

        block_in = ch * ch_mult[-1]

        for i_level in reversed(range(self.num_resolutions)):
            block     = nn.ModuleList()
            attn      = nn.ModuleList()
            block_out = ch * ch_mult[i_level]   # target channels at this level
            skip_ch   = ch * ch_mult[i_level]   # same as block out

            for i_block in range(num_res_blocks+1):

 # Last block uses shallower skip (in_ch_mult instead of ch_mult)
                if i_block == self.num_res_blocks:
                    skip_ch = ch * in_ch_mult[i_level]

                block.append(ResnetBlock(
                    in_channels   = block_in + skip_ch,    # what actually comes in
                    out_channels  = block_out,   # what we want to output
                    temb_channels = temb_ch,
                    dropout       = dropout,
                ))
                block_in = block_out    # subsequent blocks in same level get block_out
                if curr_res in attn_resolutions:
                    attn.append(AttnBlock(block_in))

            level       = nn.Module()
            level.block = block
            level.attn  = attn

            if i_level != 0:
                level.upsample = UpSample(block_out, resamp_with_conv)
                curr_res = curr_res * 2

            self.up.append(level)

        # Output head
        self.norm_out = Normalize(ch)
        self.conv_out = nn.Conv3d(ch, out_channels, kernel_size=3, padding=1)

    def forward(self, h: torch.Tensor, temb: torch.Tensor, skips: list) -> torch.Tensor:
        """
        h:    (B, bottleneck_ch, 8, 8, 8)  — from GraphToConv
        temb: (B, temb_ch)                 — timestep embedding
        """
        # self.up was appended in reversed order [level3, level2, level1, level0]
        # so self.up[0] = level3 (bottleneck), self.up[-1] = level0 (output)
        for i, level in enumerate(self.up):
            for i_block in range(self.num_res_blocks+1):

                # Concatenate skip connection from encoder
                skip = skips.pop()                  # corresp  encoder feature
                h    = torch.cat([h, skip], dim=1)  # concat along channels

                h = level.block[i_block](h, temb)
                if len(level.attn) > 0:
                    h = level.attn[i_block](h)
            if hasattr(level, 'upsample'):
                h = level.upsample(h)

        return self.conv_out(nonlinearity(self.norm_out(h)))
    

class ModelBase(nn.Module):
    """
    Full plasma CFD graph-diffusion model.

    Pipeline:
        Input (B, 4, 64, 64, 64)
            ↓ Mesh2GridEncoder  — conv encoder + conv→graph
        node_feats (B*N, node_dim)
            ↓ GAT               — graph message passing
        node_feats (B*N, node_dim)
            ↓ GraphToConv       — reshape to 3D grid
        (B, bottleneck_ch, 8, 8, 8)
            ↓ Decoder3D         — upsample back to 64³
        Output (B, 4, 64, 64, 64)

    Args:
        ch:            base channel count
        ch_mult:       channel multipliers per encoder level
        node_dim:      graph node feature dimension
        edge_dim:      graph edge feature dimension
        in_channels:   input physical fields
        out_channels:  output physical fields
        resolution:    input spatial resolution
        k:             KNN graph connectivity
    """
    def __init__(
        self,
        ch:           int   = 32,
        ch_mult:      tuple = (1, 2, 4, 8),
        node_dim:     int   = 256,
        edge_dim:     int   = 64,
        in_channels:  int   = 4,
        out_channels: int   = 4,
        resolution:   int   = 64,
        k:            int   = 16, #graph connectivity
        num_mp_layers: int  = 8, #graph layers
    ):
        super().__init__()

        # ── Derived dims ──────────────────────────────────────────────────
        self.ch           = ch
        self.temb_ch      = ch * 4
        self.node_dim     = node_dim
        bottleneck_ch     = ch * ch_mult[-1]       # 32 * 8 = 256
        bottleneck_res    = resolution // (2 ** (len(ch_mult) - 1))  # 64 // 8 = 8
        D = H = W         = bottleneck_res
        self.bottleneck_res = bottleneck_res

        # ── Graph topology — registered as buffers ────────────────────────
        # Buffers move to GPU automatically with model.to(device)
        # Not saved in state_dict (recomputed cheaply on load)
        gz, gy, gx = torch.meshgrid(
            torch.arange(D), torch.arange(H), torch.arange(W), indexing='ij'
        )
        coords   = torch.stack([gz, gy, gx], dim=-1).view(-1, 3).float()
        node_pos = (coords / torch.tensor([D-1, H-1, W-1], dtype=torch.float)) * 2 - 1

        edge_index, edge_raw = Mesh2GridEncoder.build_graph(node_pos, k)

        self.register_buffer('node_pos',   node_pos)    # (N, 3)
        self.register_buffer('edge_index', edge_index)  # (2, E)
        self.register_buffer('edge_raw',   edge_raw)    # (E, 4)

        # ── Modules ───────────────────────────────────────────────────────
        self.encoder = Mesh2GridEncoder(
            ch               = ch,
            out_ch           = ch,
            ch_mult          = ch_mult,
            num_res_blocks   = 2,
            attn_resolutions = [bottleneck_res],
            dropout          = 0.0,
            resamp_with_conv = True,
            in_channels      = in_channels,
            resolution       = resolution,
            node_dim         = node_dim,
        )

        self.edge_proj = nn.Sequential(
            nn.Linear(4, edge_dim),
            nn.SiLU(),
            nn.Linear(edge_dim, edge_dim),
        )

        self.graphlayer = GraphProcessor(
          node_dim   = node_dim,
          edge_dim   = edge_dim,
          num_layers = num_mp_layers,
          heads      = 4,
          )

        self.graph2conv = GraphToConv(
            node_ch   = node_dim,
            conv_ch   = bottleneck_ch,
            grid_size = (D, H, W),
        )

        self.decoder = Decoder3D(
            ch               = ch,
            ch_mult          = ch_mult,
            num_res_blocks   = 2,
            attn_resolutions = [bottleneck_res],
            dropout          = 0.0,
            resamp_with_conv = True,
            resolution       = resolution,
            out_channels     = out_channels,
            temb_ch          = self.temb_ch,
        )

        total = sum(p.numel() for p in self.parameters())
        print(f"ModelBase: {total:,} parameters")

    def forward(
        self,
        x: torch.Tensor,    # (B, in_channels, 64, 64, 64)
        t: torch.Tensor,    # (B,) integer timesteps
    ) -> torch.Tensor:      # (B, out_channels, 64, 64, 64)

        B = x.shape[0]
        N = self.node_pos.shape[0]    # 512

        # ── 1. Encode ─────────────────────────────────────────────────────
        node_feats, hs = self.encoder(x, t, self.node_pos)
        # node_feats: (B*N, node_dim)
        # hs:         skip connections for decoder

        # ── 2. Project edge geometry → edge features ──────────────────────
        edge_attr = self.edge_proj(self.edge_raw)          # (E, edge_dim)

        # ── 3. Batch graph across batch elements ──────────────────────────
        edge_index_b, edge_attr_b = batch_graph(
            self.edge_index,
            edge_attr,
            B    = B,
            N_per = N,
            device = x.device,
        )

        # ── 4. GAT message passing ────────────────────────────────────────
        node_feats = self.graphlayer(node_feats, edge_index_b, edge_attr_b)
        # (B*N, node_dim)

        # ── 5. Graph → Conv grid ──────────────────────────────────────────
        h = self.graph2conv(node_feats)
        # (B, bottleneck_ch, 8, 8, 8)

        # ── 6. Build timestep embedding for decoder ───────────────────────
        temb = get_timestep_embedding(t, self.ch)          # (B, ch)
        temb = self.encoder.temb_dense[0](temb)
        temb = nonlinearity(temb)
        temb = self.encoder.temb_dense[1](temb)            # (B, temb_ch)

        # ── 7. Decode ─────────────────────────────────────────────────────
        out = self.decoder(h, temb, skips =hs)                        # (B, out_channels, 64, 64, 64)

        return out
