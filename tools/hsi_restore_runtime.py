#!/usr/bin/env python3
"""Минимальный inference-runtime HyperRestormer, перенесённый из HSIRestore."""

from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from einops import rearrange


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict):
        raise ValueError("YAML config must be a dictionary at top-level")
    return config


def _ensure_list(value: int | Iterable[int], count: int, default: int) -> list[int]:
    if isinstance(value, int):
        return [value for _ in range(count)]
    result = list(value)
    if len(result) < count:
        fill = result[-1] if result else default
        result.extend([fill] * (count - len(result)))
    return result[:count]


def _gaussian_kernel2d(
    kernel_size: int,
    sigma: float,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    kernel_size = kernel_size if kernel_size % 2 == 1 else kernel_size + 1
    coords = torch.arange(kernel_size, device=device, dtype=dtype) - kernel_size // 2
    gaussian = torch.exp(-(coords**2) / (2 * sigma * sigma + 1e-12))
    gaussian = gaussian / gaussian.sum()
    kernel = torch.outer(gaussian, gaussian)
    return kernel / kernel.sum()


def _gaussian_blur(x: torch.Tensor, kernel_size: int, sigma: float) -> torch.Tensor:
    if x.dim() != 4:
        raise ValueError(f"Expected 4D tensor NCHW, got shape={tuple(x.shape)}")
    kernel_size = kernel_size if kernel_size % 2 == 1 else kernel_size + 1
    if kernel_size <= 1:
        return x
    channels = x.shape[1]
    kernel_2d = _gaussian_kernel2d(kernel_size, sigma, x.device, x.dtype)
    kernel = kernel_2d.view(1, 1, kernel_size, kernel_size).repeat(channels, 1, 1, 1)
    padding = kernel_size // 2
    padded = F.pad(x, (padding, padding, padding, padding), mode="reflect")
    return F.conv2d(padded, kernel, groups=channels)


def _highpass(x: torch.Tensor, kernel_size: int, sigma: float) -> torch.Tensor:
    return x - _gaussian_blur(x, kernel_size=kernel_size, sigma=sigma)


class LayerNorm2d(nn.Module):
    def __init__(self, num_channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1, num_channels, 1, 1))
        self.bias = nn.Parameter(torch.zeros(1, num_channels, 1, 1))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=1, keepdim=True)
        variance = (x - mean).pow(2).mean(dim=1, keepdim=True)
        normalized = (x - mean) / torch.sqrt(variance + self.eps)
        return normalized * self.weight + self.bias


class SpatialWindowAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, window_size: int, dropout: float) -> None:
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"dim ({dim}) must be divisible by num_heads ({num_heads})")
        self.num_heads = num_heads
        self.window_size = window_size
        self.scale = (dim // num_heads) ** -0.5
        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=False)
        self.proj = nn.Conv2d(dim, dim, kernel_size=1, bias=False)
        self.attn_drop = nn.Dropout(dropout)
        self.proj_drop = nn.Dropout(dropout)

    def _pad(self, x: torch.Tensor) -> tuple[torch.Tensor, int, int]:
        height, width = x.shape[-2:]
        pad_h = (self.window_size - (height % self.window_size)) % self.window_size
        pad_w = (self.window_size - (width % self.window_size)) % self.window_size
        if pad_h > 0 or pad_w > 0:
            mode = "replicate" if pad_h >= height or pad_w >= width else "reflect"
            x = F.pad(x, (0, pad_w, 0, pad_h), mode=mode)
        return x, pad_h, pad_w

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, _channels, height, width = x.shape
        padded, pad_h, pad_w = self._pad(x)
        _batch, _channels, padded_h, padded_w = padded.shape
        q, k, v = self.qkv(padded).chunk(3, dim=1)
        window = self.window_size
        windows_h = padded_h // window
        windows_w = padded_w // window
        pattern = "n (head d) (nh ws1) (nw ws2) -> (n nh nw) head (ws1 ws2) d"
        kwargs = {
            "head": self.num_heads,
            "nh": windows_h,
            "nw": windows_w,
            "ws1": window,
            "ws2": window,
        }
        q = rearrange(q, pattern, **kwargs)
        k = rearrange(k, pattern, **kwargs)
        v = rearrange(v, pattern, **kwargs)
        attention = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attention = self.attn_drop(attention.softmax(dim=-1))
        out = torch.matmul(attention, v)
        out = rearrange(
            out,
            "(n nh nw) head (ws1 ws2) d -> n (head d) (nh ws1) (nw ws2)",
            n=batch,
            nh=windows_h,
            nw=windows_w,
            ws1=window,
            ws2=window,
        )
        out = self.proj_drop(self.proj(out))
        return out[:, :, :height, :width] if pad_h > 0 or pad_w > 0 else out


class SpectralChannelAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, dropout: float) -> None:
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"dim ({dim}) must be divisible by num_heads ({num_heads})")
        self.num_heads = num_heads
        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=False)
        self.qkv_dw = nn.Conv2d(
            dim * 3,
            dim * 3,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=dim * 3,
            bias=False,
        )
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.proj = nn.Conv2d(dim, dim, kernel_size=1, bias=False)
        self.attn_drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _batch, _channels, height, width = x.shape
        q, k, v = self.qkv_dw(self.qkv(x)).chunk(3, dim=1)
        pattern = "n (head d) h w -> n head d (h w)"
        q = rearrange(q, pattern, head=self.num_heads)
        k = rearrange(k, pattern, head=self.num_heads)
        v = rearrange(v, pattern, head=self.num_heads)
        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)
        attention = torch.matmul(q, k.transpose(-2, -1)) * self.temperature
        attention = self.attn_drop(attention.softmax(dim=-1))
        out = torch.matmul(attention, v)
        out = rearrange(
            out,
            "n head d (h w) -> n (head d) h w",
            head=self.num_heads,
            h=height,
            w=width,
        )
        return self.proj(out)


class GatedFeedForward(nn.Module):
    def __init__(self, dim: int, expansion: float, dropout: float) -> None:
        super().__init__()
        hidden = int(dim * expansion)
        self.project_in = nn.Conv2d(dim, hidden * 2, kernel_size=1, bias=False)
        self.dwconv = nn.Conv2d(
            hidden * 2,
            hidden * 2,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=hidden * 2,
            bias=False,
        )
        self.project_out = nn.Conv2d(hidden, dim, kernel_size=1, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.dwconv(self.project_in(x))
        left, right = x.chunk(2, dim=1)
        return self.dropout(self.project_out(F.gelu(left) * right))


class SpectralSpatialBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        window_size: int,
        ff_mult: float,
        dropout: float,
    ) -> None:
        super().__init__()
        self.norm1 = LayerNorm2d(dim)
        self.spatial_attn = SpatialWindowAttention(dim, num_heads, window_size, dropout)
        self.norm2 = LayerNorm2d(dim)
        self.spectral_attn = SpectralChannelAttention(dim, num_heads, dropout)
        self.norm3 = LayerNorm2d(dim)
        self.ffn = GatedFeedForward(dim, ff_mult, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.spatial_attn(self.norm1(x))
        x = x + self.spectral_attn(self.norm2(x))
        return x + self.ffn(self.norm3(x))


class Downsample(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.body = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.body(x)


class Upsample(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.body = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.body(x)


class HyperRestormerLite(nn.Module):
    def __init__(
        self,
        in_channels: int = 44,
        dim: int = 48,
        num_heads: int = 4,
        window_size: int = 8,
        enc_blocks: list[int] | tuple[int, ...] = (2, 2),
        latent_blocks: int = 4,
        dec_blocks: list[int] | tuple[int, ...] = (2, 2),
        ff_mult: float = 2.0,
        dropout: float = 0.0,
        hf_kernel: int = 5,
        hf_sigma: float = 1.0,
    ) -> None:
        super().__init__()
        encoder_counts = _ensure_list(enc_blocks, count=2, default=2)
        decoder_counts = _ensure_list(dec_blocks, count=2, default=2)
        self.in_channels = in_channels
        self.hf_kernel = hf_kernel
        self.hf_sigma = hf_sigma
        self.input_proj = nn.Conv2d(in_channels, dim, kernel_size=3, stride=1, padding=1)
        self.enc1 = nn.Sequential(
            *[
                SpectralSpatialBlock(dim, num_heads, window_size, ff_mult, dropout)
                for _ in range(encoder_counts[0])
            ]
        )
        self.down1 = Downsample(dim, dim * 2)
        self.enc2 = nn.Sequential(
            *[
                SpectralSpatialBlock(dim * 2, num_heads, window_size, ff_mult, dropout)
                for _ in range(encoder_counts[1])
            ]
        )
        self.down2 = Downsample(dim * 2, dim * 4)
        self.latent = nn.Sequential(
            *[
                SpectralSpatialBlock(dim * 4, num_heads, window_size, ff_mult, dropout)
                for _ in range(latent_blocks)
            ]
        )
        self.up1 = Upsample(dim * 4, dim * 2)
        self.fuse2 = nn.Conv2d(dim * 4, dim * 2, kernel_size=1, stride=1)
        self.dec2 = nn.Sequential(
            *[
                SpectralSpatialBlock(dim * 2, num_heads, window_size, ff_mult, dropout)
                for _ in range(decoder_counts[0])
            ]
        )
        self.up2 = Upsample(dim * 2, dim)
        self.fuse1 = nn.Conv2d(dim * 2, dim, kernel_size=1, stride=1)
        self.dec1 = nn.Sequential(
            *[
                SpectralSpatialBlock(dim, num_heads, window_size, ff_mult, dropout)
                for _ in range(decoder_counts[1])
            ]
        )
        self.output_proj = nn.Conv2d(dim, in_channels, kernel_size=3, stride=1, padding=1)
        gate_mid = max(16, dim // 2)
        self.gate_head = nn.Sequential(
            nn.Conv2d(in_channels, gate_mid, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(gate_mid, in_channels, kernel_size=3, padding=1),
        )
        self.detail_proj = nn.Sequential(
            nn.Conv2d(1, gate_mid, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(gate_mid, in_channels, kernel_size=3, padding=1),
        )
        self.detail_scale = nn.Parameter(torch.tensor(0.1, dtype=torch.float32))

    @staticmethod
    def _pad_to_multiple(x: torch.Tensor, multiple: int) -> tuple[torch.Tensor, int, int]:
        height, width = x.shape[-2:]
        pad_h = (multiple - (height % multiple)) % multiple
        pad_w = (multiple - (width % multiple)) % multiple
        if pad_h == 0 and pad_w == 0:
            return x, height, width
        mode = "replicate" if pad_h >= height or pad_w >= width else "reflect"
        return F.pad(x, (0, pad_w, 0, pad_h), mode=mode), height, width

    def forward(self, source: torch.Tensor) -> dict[str, torch.Tensor]:
        if source.shape[1] != self.in_channels:
            raise ValueError(f"Expected {self.in_channels} channels, got {source.shape[1]}")
        source_pad, original_h, original_w = self._pad_to_multiple(source, multiple=4)
        encoded1 = self.enc1(self.input_proj(source_pad))
        encoded2 = self.enc2(self.down1(encoded1))
        latent = self.latent(self.down2(encoded2))
        decoded2 = self.up1(latent)
        if decoded2.shape[-2:] != encoded2.shape[-2:]:
            decoded2 = F.interpolate(decoded2, size=encoded2.shape[-2:], mode="bilinear", align_corners=False)
        decoded2 = self.dec2(self.fuse2(torch.cat([decoded2, encoded2], dim=1)))
        decoded1 = self.up2(decoded2)
        if decoded1.shape[-2:] != encoded1.shape[-2:]:
            decoded1 = F.interpolate(decoded1, size=encoded1.shape[-2:], mode="bilinear", align_corners=False)
        decoded1 = self.dec1(self.fuse1(torch.cat([decoded1, encoded1], dim=1)))
        base = self.output_proj(decoded1)
        gate = torch.softmax(self.gate_head(source_pad), dim=1)
        source_highpass = _highpass(source_pad, kernel_size=self.hf_kernel, sigma=self.hf_sigma)
        detail_map = (gate * source_highpass).sum(dim=1, keepdim=True)
        detail_inject = self.detail_proj(detail_map)
        prediction = source_pad + base + self.detail_scale * detail_inject
        return {
            "pred": prediction[:, :, :original_h, :original_w].clamp(0.0, 1.0),
            "gate": gate[:, :, :original_h, :original_w],
            "detail_map": detail_map[:, :, :original_h, :original_w],
        }


def choose_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def _to_hwc(cube: np.ndarray, expected_channels: int | None) -> tuple[np.ndarray, int]:
    if cube.ndim != 3:
        raise ValueError(f"Expected 3D cube, got shape={cube.shape}")
    band_axis = None
    if expected_channels is not None:
        matches = [index for index, size in enumerate(cube.shape) if int(size) == int(expected_channels)]
        if len(matches) == 1:
            band_axis = matches[0]
    if band_axis is None:
        band_axis = int(np.argmin(cube.shape))
    if band_axis == 2:
        cube_hwc = cube
    elif band_axis == 0:
        cube_hwc = np.transpose(cube, (1, 2, 0))
    elif band_axis == 1:
        cube_hwc = np.transpose(cube, (0, 2, 1))
    else:  # pragma: no cover
        raise ValueError(f"Unable to infer band axis for shape={cube.shape}")
    return np.clip(cube_hwc.astype(np.float32, copy=False), 0.0, 1.0), band_axis


def _from_hwc(cube_hwc: np.ndarray, band_axis: int) -> np.ndarray:
    if band_axis == 2:
        result = cube_hwc
    elif band_axis == 0:
        result = np.transpose(cube_hwc, (2, 0, 1))
    elif band_axis == 1:
        result = np.transpose(cube_hwc, (0, 2, 1))
    else:
        raise ValueError(f"Unsupported band_axis={band_axis}")
    return np.clip(result.astype(np.float32, copy=False), 0.0, 1.0)


def load_cube(path: Path, expected_channels: int | None = None) -> tuple[np.ndarray, int]:
    return _to_hwc(np.load(str(path)), expected_channels=expected_channels)


def save_cube(path: Path, cube_hwc: np.ndarray, band_axis: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(str(path), _from_hwc(cube_hwc, band_axis))


def load_model(config: dict[str, Any], checkpoint_path: Path, device: torch.device) -> nn.Module:
    model_config = config["model"]
    loss_config = config.get("loss", {})
    model = HyperRestormerLite(
        in_channels=int(model_config.get("in_channels", 44)),
        dim=int(model_config.get("dim", 48)),
        num_heads=int(model_config.get("num_heads", 4)),
        window_size=int(model_config.get("window_size", 8)),
        enc_blocks=model_config.get("enc_blocks", [2, 2]),
        latent_blocks=int(model_config.get("latent_blocks", 4)),
        dec_blocks=model_config.get("dec_blocks", [2, 2]),
        ff_mult=float(model_config.get("ff_mult", 2.0)),
        dropout=float(model_config.get("dropout", 0.0)),
        hf_kernel=int(loss_config.get("hf_kernel", 5)),
        hf_sigma=float(loss_config.get("hf_sigma", 1.0)),
    )
    checkpoint = torch.load(str(checkpoint_path), map_location="cpu", weights_only=True)
    state_dict = checkpoint.get("state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    if not isinstance(state_dict, dict):
        raise RuntimeError(f"Invalid checkpoint state from: {checkpoint_path}")
    try:
        model.load_state_dict(state_dict, strict=True)
    except RuntimeError:
        model_state = {
            key[len("model.") :]: value
            for key, value in state_dict.items()
            if isinstance(key, str) and key.startswith("model.")
        }
        if not model_state:
            raise RuntimeError(f"Cannot load checkpoint weights from: {checkpoint_path}")
        model.load_state_dict(model_state, strict=True)
    model = model.to(device)
    model.eval()
    return model


def _positions(length: int, tile: int, step: int) -> list[int]:
    if length <= tile:
        return [0]
    positions = list(range(0, length - tile + 1, step))
    last = length - tile
    if positions[-1] != last:
        positions.append(last)
    return positions


def _window_2d(tile: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    window = torch.hann_window(tile, periodic=False, device=device, dtype=dtype)
    window = torch.clamp(window, min=1e-3)
    return (window[:, None] * window[None, :]).unsqueeze(0).unsqueeze(0)


def predict_cube(
    model: nn.Module,
    source_hwc: np.ndarray,
    device: torch.device,
    tile_size: int,
    tile_overlap: int,
    use_fp16: bool,
    return_gate: bool,
) -> tuple[np.ndarray, np.ndarray | None]:
    source = torch.from_numpy(np.transpose(source_hwc, (2, 0, 1))).unsqueeze(0)
    source = source.to(device=device, dtype=torch.float32)
    use_amp = bool(use_fp16 and device.type == "cuda")
    autocast = (
        torch.autocast(device_type="cuda", dtype=torch.float16, enabled=True)
        if use_amp
        else nullcontext()
    )
    with torch.inference_mode():
        if tile_size <= 0 or (source.shape[-2] <= tile_size and source.shape[-1] <= tile_size):
            with autocast:
                output = model(source)
            prediction = output["pred"]
            gate = output.get("gate") if return_gate else None
        else:
            if tile_overlap < 0:
                raise ValueError("tile-overlap must be >= 0")
            if tile_overlap >= tile_size:
                raise ValueError("tile-overlap must be smaller than tile-size")
            _batch, channels, original_h, original_w = source.shape
            pad_h = max(0, tile_size - original_h)
            pad_w = max(0, tile_size - original_w)
            if pad_h > 0 or pad_w > 0:
                mode = "replicate" if pad_h >= original_h or pad_w >= original_w else "reflect"
                source = F.pad(source, (0, pad_w, 0, pad_h), mode=mode)
            _batch, _channels, height, width = source.shape
            step = tile_size - tile_overlap
            ys = _positions(height, tile_size, step)
            xs = _positions(width, tile_size, step)
            window = _window_2d(tile_size, device=source.device, dtype=source.dtype)
            prediction_sum = torch.zeros((1, channels, height, width), device=source.device)
            weight_sum = torch.zeros((1, 1, height, width), device=source.device)
            gate_sum = torch.zeros_like(prediction_sum) if return_gate else None
            for y in ys:
                for x in xs:
                    patch = source[:, :, y : y + tile_size, x : x + tile_size]
                    with autocast:
                        output = model(patch)
                    prediction_sum[:, :, y : y + tile_size, x : x + tile_size] += (
                        output["pred"].to(source.dtype) * window
                    )
                    weight_sum[:, :, y : y + tile_size, x : x + tile_size] += window
                    if gate_sum is not None:
                        gate_sum[:, :, y : y + tile_size, x : x + tile_size] += (
                            output["gate"].to(source.dtype) * window
                        )
            prediction = prediction_sum / weight_sum.clamp_min(1e-6)
            gate = gate_sum / weight_sum.clamp_min(1e-6) if gate_sum is not None else None
            prediction = prediction[:, :, :original_h, :original_w]
            if gate is not None:
                gate = gate[:, :, :original_h, :original_w]
    prediction_hwc = prediction.squeeze(0).permute(1, 2, 0).float().cpu().numpy()
    prediction_hwc = np.clip(prediction_hwc, 0.0, 1.0).astype(np.float32)
    gate_hwc = None
    if gate is not None:
        gate_hwc = gate.squeeze(0).permute(1, 2, 0).float().cpu().numpy().astype(np.float32)
    return prediction_hwc, gate_hwc
