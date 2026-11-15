"""FiLM-conditioned Residual U-Net for 250m weather downscaling."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .build_patches import ALL_CHANNELS, TIME_CHANNELS

SPATIAL_CHANNELS = [c for c in ALL_CHANNELS if c not in TIME_CHANNELS]
SPATIAL_INDICES = [ALL_CHANNELS.index(c) for c in SPATIAL_CHANNELS]
WN3_EXCLUDED = {"sp_hpa", "snow_depth_m"}
WN3_SPATIAL_CHANNELS = [c for c in SPATIAL_CHANNELS if c not in WN3_EXCLUDED]
WN3_SPATIAL_INDICES = [ALL_CHANNELS.index(c) for c in WN3_SPATIAL_CHANNELS]
TIME_INDICES = [ALL_CHANNELS.index(c) for c in TIME_CHANNELS]
SSRD_INDEX = ALL_CHANNELS.index("ssrd_wm2")


def split_spatial_and_cond(
    x: torch.Tensor,
    wn3_compat: bool = False,
    solar: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Split (B, 27, H, W) input tensor into spatial grid + scalar time/solar conditioning.

    Returns:
      x_spatial: (B, 23 or 21, H, W) dynamic ERA5-Land + static 250m terrain/land-cover channels
      cond:      (B, 5 or 10)       [sin_hour, cos_hour, sin_doy, cos_doy, patch_mean_ssrd,
                                      (+ solar_film: sin_az, cos_az, sin_alt, sin_noon_alt, lat_norm)]
    """
    idx = WN3_SPATIAL_INDICES if wn3_compat else SPATIAL_INDICES
    x_spatial = x[:, idx, :, :]
    time_vec = x[:, TIME_INDICES, 0, 0]  # constant across patch
    mean_ssrd = x[:, SSRD_INDEX, :, :].mean(dim=(1, 2)).unsqueeze(1)
    parts = [time_vec, mean_ssrd]
    if solar is not None:
        parts.append(solar)
    cond = torch.cat(parts, dim=1)
    return x_spatial, cond


def _gn_groups(ch: int, max_groups: int = 8) -> int:
    g = min(max_groups, ch)
    while ch % g != 0:
        g -= 1
    return g


class FiLMResBlock(nn.Module):
    """Residual 2D conv block with GroupNorm + FiLM modulation (gamma, beta)."""

    def __init__(self, in_ch: int, out_ch: int, cond_emb_dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False)
        self.norm1 = nn.GroupNorm(_gn_groups(out_ch), out_ch)
        self.film = nn.Linear(cond_emb_dim, 2 * out_ch)
        # Initialize FiLM near identity: (1 + 0) * h + 0
        nn.init.zeros_(self.film.weight)
        nn.init.zeros_(self.film.bias)

        self.act = nn.SiLU()
        self.drop = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False)
        self.norm2 = nn.GroupNorm(_gn_groups(out_ch), out_ch)
        self.skip = nn.Conv2d(in_ch, out_ch, kernel_size=1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor, cond_emb: torch.Tensor) -> torch.Tensor:
        h = self.norm1(self.conv1(x))
        gamma, beta = self.film(cond_emb).chunk(2, dim=1)
        h = h * (1.0 + gamma[:, :, None, None]) + beta[:, :, None, None]
        h = self.act(h)
        h = self.drop(h)
        h = self.norm2(self.conv2(h))
        return self.act(h + self.skip(x))


class FiLMUNet(nn.Module):
    """4-level Residual FiLM U-Net predicting [dT2m, du10, dv10] added to coarse ERA5-Land."""

    def __init__(
        self,
        in_spatial_ch: int | None = None,
        cond_in_dim: int | None = None,
        out_ch: int = 3,
        base_ch: int = 64,
        cond_emb_dim: int = 128,
        dropout: float = 0.0,
        wn3_compat: bool = False,
        use_solar_film: bool = False,
    ) -> None:
        super().__init__()
        self.wn3_compat = wn3_compat
        self.use_solar_film = use_solar_film
        if in_spatial_ch is None:
            in_spatial_ch = len(WN3_SPATIAL_CHANNELS) if wn3_compat else len(SPATIAL_CHANNELS)
        if cond_in_dim is None:
            cond_in_dim = 10 if use_solar_film else 5
        c1, c2, c3, c4 = base_ch, base_ch * 2, base_ch * 4, base_ch * 8

        self.cond_mlp = nn.Sequential(
            nn.Linear(cond_in_dim, cond_emb_dim),
            nn.SiLU(),
            nn.Linear(cond_emb_dim, cond_emb_dim),
            nn.SiLU(),
        )

        # Encoder
        self.enc1 = FiLMResBlock(in_spatial_ch, c1, cond_emb_dim)
        self.enc2 = FiLMResBlock(c1, c2, cond_emb_dim)
        self.enc3 = FiLMResBlock(c2, c3, cond_emb_dim, dropout=dropout)
        self.enc4 = FiLMResBlock(c3, c4, cond_emb_dim, dropout=dropout)

        # Bottleneck (16x16 for 256x256 input)
        self.bot = FiLMResBlock(c4, c4, cond_emb_dim, dropout=dropout)

        # Decoder (bilinear upsampling + skip concat)
        self.dec4 = FiLMResBlock(c4 + c4, c3, cond_emb_dim, dropout=dropout)
        self.dec3 = FiLMResBlock(c3 + c3, c2, cond_emb_dim)
        self.dec2 = FiLMResBlock(c2 + c2, c1, cond_emb_dim)
        self.dec1 = FiLMResBlock(c1 + c1, c1, cond_emb_dim)

        self.head = nn.Conv2d(c1, out_ch, kernel_size=1)
        # Start at coarse ERA5-Land baseline (delta = 0)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(
        self,
        x: torch.Tensor,
        coarse: torch.Tensor | None = None,
        solar: torch.Tensor | None = None,
    ) -> torch.Tensor:
        sol = solar if self.use_solar_film else None
        x_sp, cond = split_spatial_and_cond(x, wn3_compat=self.wn3_compat, solar=sol)
        cemb = self.cond_mlp(cond)

        e1 = self.enc1(x_sp, cemb)                          # 256
        e2 = self.enc2(F.avg_pool2d(e1, 2), cemb)           # 128
        e3 = self.enc3(F.avg_pool2d(e2, 2), cemb)           # 64
        e4 = self.enc4(F.avg_pool2d(e3, 2), cemb)           # 32

        b = self.bot(F.avg_pool2d(e4, 2), cemb)             # 16

        d4 = self.dec4(torch.cat([F.interpolate(b, scale_factor=2, mode="bilinear", align_corners=False), e4], dim=1), cemb)
        d3 = self.dec3(torch.cat([F.interpolate(d4, scale_factor=2, mode="bilinear", align_corners=False), e3], dim=1), cemb)
        d2 = self.dec2(torch.cat([F.interpolate(d3, scale_factor=2, mode="bilinear", align_corners=False), e2], dim=1), cemb)
        d1 = self.dec1(torch.cat([F.interpolate(d2, scale_factor=2, mode="bilinear", align_corners=False), e1], dim=1), cemb)

        delta = self.head(d1)
        return coarse + delta if coarse is not None else delta
