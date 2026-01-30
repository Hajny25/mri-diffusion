import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """(Conv => GroupNorm => SiLU) * 2, with optional FiLM conditioning."""
    def __init__(self, in_ch, out_ch, time_emb_dim=0):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1)
        self.norm1 = nn.GroupNorm(8, out_ch)
        self.norm2 = nn.GroupNorm(8, out_ch)
        self.act = nn.SiLU()

        # if time/dir embedding is used, produce scale & shift per conv layer
        if time_emb_dim > 0:
            self.film1 = nn.Linear(time_emb_dim, out_ch * 2)
            self.film2 = nn.Linear(time_emb_dim, out_ch * 2)
        else:
            self.film1 = self.film2 = None

    def apply_film(self, h, emb, film_layer):
        """Apply FiLM (scale & shift) using embedding."""
        if film_layer is None or emb is None:
            return h
        # emb: [B, time_emb_dim] -> [B, 2*C]
        scale_shift = film_layer(emb)  # [B, 2*C]
        scale, shift = scale_shift.chunk(2, dim=1)  # [B, C], [B, C]
        # reshape for broadcasting
        scale = scale[:, :, None, None]
        shift = shift[:, :, None, None]
        return h * (1 + scale) + shift

    def forward(self, x, emb=None):
        h = self.conv1(x)
        h = self.norm1(h)
        h = self.apply_film(h, emb, self.film1)
        h = self.act(h)

        h = self.conv2(h)
        h = self.norm2(h)
        h = self.apply_film(h, emb, self.film2)
        h = self.act(h)
        return h


class Down(nn.Module):
    def __init__(self, in_ch, out_ch, time_emb_dim=0):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.conv = DoubleConv(in_ch, out_ch, time_emb_dim)

    def forward(self, x, emb=None):
        x = self.pool(x)
        x = self.conv(x, emb)
        return x


class Up(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch, time_emb_dim=0):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, in_ch // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_ch // 2 + skip_ch, out_ch, time_emb_dim)

    def forward(self, x, skip, emb=None):
        x = self.up(x)
        # pad if needed (odd sizes)
        diffY = skip.size(2) - x.size(2)
        diffX = skip.size(3) - x.size(3)
        if diffY != 0 or diffX != 0:
            x = F.pad(x, [diffX // 2, diffX - diffX // 2,
                          diffY // 2, diffY - diffY // 2])

        x = torch.cat([skip, x], dim=1)
        x = self.conv(x, emb)
        return x


class TimeDirSliceEmbedding(nn.Module):
    """
    Combine diffusion timestep, direction, and normalized slice index into a single embedding.
    - t: diffusion timestep
    - direction: -1 or +1
    - slice_pos: normalized slice index in [0, 1]
    """
    def __init__(self, emb_dim):
        super().__init__()
        self.emb_dim = emb_dim
        self.time_mlp = nn.Sequential(
            nn.Linear(emb_dim, emb_dim),
            nn.SiLU(),
            nn.Linear(emb_dim, emb_dim),
        )
        self.dir_mlp = nn.Sequential(
            nn.Linear(1, emb_dim),
            nn.SiLU(),
            nn.Linear(emb_dim, emb_dim),
        )
        self.slice_mlp = nn.Sequential(
            nn.Linear(1, emb_dim),
            nn.SiLU(),
            nn.Linear(emb_dim, emb_dim),
        )

    def sinusoidal_embedding(self, t):
        if t.dim() == 1:
            t = t[:, None]
        half_dim = self.emb_dim // 2
        scale = -math.log(10000) / (half_dim - 1)
        freqs = torch.exp(
            torch.arange(half_dim, device=t.device, dtype=torch.float32)
            * scale
        )
        args = t * freqs[None, :]
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        return emb

    def forward(self, t, direction, slice_pos):
        """
        t: [B]
        direction: [B] in {-1, +1}
        slice_pos: [B] in [0,1], e.g. center_idx / (num_slices-1)
        """
        t = t.float()
        direction = direction.float().view(-1, 1)
        slice_pos = slice_pos.float().view(-1, 1)

        t_emb = self.sinusoidal_embedding(t)
        t_emb = self.time_mlp(t_emb)

        d_emb = self.dir_mlp(direction)
        s_emb = self.slice_mlp(slice_pos)

        return t_emb + d_emb + s_emb  # [B, emb_dim]

class UNetSlicePredictor(nn.Module):
    """
    UNet that predicts noise (or target slice) for the neighbor, conditioned on:
      - noisy neighbor slice x_t
      - center slice (condition)
      - timestep t
      - direction (-1 or +1)
    """

    def __init__(self, in_channels=2, out_channels=1, base_channels=64, time_emb_dim=128):
        """
        in_channels:
            2 if you concatenate [noisy_neighbor, center_slice]
        out_channels:
            1 if you predict a single-channel slice or noise
        """
        super().__init__()

        self.time_dir_slice_emb = TimeDirSliceEmbedding(time_emb_dim)

        self.inc = DoubleConv(in_channels, base_channels, time_emb_dim)
        self.down1 = Down(base_channels, base_channels * 2, time_emb_dim)
        self.down2 = Down(base_channels * 2, base_channels * 4, time_emb_dim)
        self.down3 = Down(base_channels * 4, base_channels * 4, time_emb_dim)

        self.bot = DoubleConv(base_channels * 4, base_channels * 4, time_emb_dim)

        self.up1 = Up(base_channels * 4, base_channels * 4, base_channels * 4, time_emb_dim)
        self.up2 = Up(base_channels * 4, base_channels * 2, base_channels * 2, time_emb_dim)
        self.up3 = Up(base_channels * 2, base_channels, base_channels, time_emb_dim)

        self.outc = nn.Conv2d(base_channels, out_channels, kernel_size=1)

    def forward(self, x, t, direction, slice_pos):
        """
        x: [B, 2, H, W]
           channel 0: noisy neighbor slice (x_t)
           channel 1: conditioning slice
        t: [B] diffusion timesteps
        direction: [B] in {-1, +1}
        """
        emb = self.time_dir_slice_emb(t, direction, slice_pos)  # [B, time_emb_dim]

        x1 = self.inc(x, emb)
        x2 = self.down1(x1, emb)
        x3 = self.down2(x2, emb)
        x4 = self.down3(x3, emb)

        xb = self.bot(x4, emb)

        x = self.up1(xb, x3, emb)
        x = self.up2(x, x2, emb)
        x = self.up3(x, x1, emb)
        out = self.outc(x)

        return out