#!/usr/bin/env python3
"""RAW buffer decode and lightweight preview helpers."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

try:
    from PIL import Image

    PIL_AVAILABLE = True
except Exception:  # pragma: no cover - optional dependency in runtime env
    Image = None  # type: ignore[assignment]
    PIL_AVAILABLE = False


PFNC_NAME_TO_INT: dict[str, int] = {
    "MONO8": 0x01080001,
    "BAYERRG8": 0x01080009,
    "MONO12": 0x01100005,
    "BAYERRG12": 0x01100011,
    # Some devices expose BayerGB variants.
    "BAYERGB8": 0x0108000A,
    "BAYERGB12": 0x01100015,
}

PFNC_INT_TO_NAME: dict[int, str] = {v: k for k, v in PFNC_NAME_TO_INT.items()}

UNSUPPORTED_PACKED: set[str] = {
    "MONO10PACKED",
    "MONO12PACKED",
    "BAYERGB10",
}


@dataclass
class DecodedFrame:
    array: np.ndarray
    pixel_format_name: str


class RawDecodeError(RuntimeError):
    """Raised when RAW decoding cannot continue."""


def pixel_format_to_name(pixel_format: int | str) -> str:
    if isinstance(pixel_format, str):
        return pixel_format.strip().upper()
    return PFNC_INT_TO_NAME.get(int(pixel_format), f"PFNC_0x{int(pixel_format):08X}")


def _require_length(raw: bytes, expected: int, pixel_format_name: str) -> bytes:
    if len(raw) < expected:
        raise RawDecodeError(
            f"RAW payload is too short for {pixel_format_name}: got {len(raw)} bytes, expected >= {expected}"
        )
    return raw[:expected]


def decode_buffer_to_ndarray(
    raw: bytes,
    width: int,
    height: int,
    pixel_format: int | str,
) -> np.ndarray:
    if width <= 0 or height <= 0:
        raise RawDecodeError(f"Invalid shape: width={width}, height={height}")

    fmt = pixel_format_to_name(pixel_format)
    if fmt in UNSUPPORTED_PACKED:
        raise RawDecodeError(
            f"Packed format {fmt} is not supported in scientific RAW path yet. Use unpacked variant."
        )

    n_px = width * height
    if fmt in {"MONO8", "BAYERRG8", "BAYERGB8"}:
        payload = _require_length(raw, n_px, fmt)
        return np.frombuffer(payload, dtype=np.uint8, count=n_px).reshape((height, width))

    if fmt in {"MONO12", "BAYERRG12", "BAYERGB12"}:
        payload = _require_length(raw, n_px * 2, fmt)
        # Unpacked 12-bit formats are transported in 16-bit containers.
        return np.frombuffer(payload, dtype="<u2", count=n_px).reshape((height, width))

    # Best-effort fallback: infer from raw size.
    if len(raw) >= n_px * 2:
        payload = raw[: n_px * 2]
        return np.frombuffer(payload, dtype="<u2", count=n_px).reshape((height, width))
    if len(raw) >= n_px:
        payload = raw[:n_px]
        return np.frombuffer(payload, dtype=np.uint8, count=n_px).reshape((height, width))

    raise RawDecodeError(f"Unsupported pixel format {fmt} with payload length {len(raw)}")


def _to_u8_for_preview(raw_array: np.ndarray, pixel_format_name: str) -> np.ndarray:
    if raw_array.dtype == np.uint8:
        return raw_array
    fmt = pixel_format_name.upper()
    if "12" in fmt:
        return np.clip(raw_array.astype(np.uint16) >> 4, 0, 255).astype(np.uint8)
    if "10" in fmt:
        return np.clip(raw_array.astype(np.uint16) >> 2, 0, 255).astype(np.uint8)
    # Generic 16-bit fallback.
    return np.clip(raw_array.astype(np.uint16) >> 8, 0, 255).astype(np.uint8)


def _bayer_quick_to_rgb_u8(raw_u8: np.ndarray, pattern: str) -> np.ndarray:
    h, w = raw_u8.shape
    if h < 2 or w < 2:
        return np.repeat(raw_u8[:, :, None], 3, axis=2)

    if pattern == "RGGB":
        r = raw_u8[0::2, 0::2]
        g1 = raw_u8[0::2, 1::2].astype(np.uint16)
        g2 = raw_u8[1::2, 0::2].astype(np.uint16)
        b = raw_u8[1::2, 1::2]
    elif pattern == "GBRG":
        g1 = raw_u8[0::2, 0::2].astype(np.uint16)
        b = raw_u8[0::2, 1::2]
        r = raw_u8[1::2, 0::2]
        g2 = raw_u8[1::2, 1::2].astype(np.uint16)
    else:
        return np.repeat(raw_u8[:, :, None], 3, axis=2)

    hh = min(r.shape[0], g1.shape[0], g2.shape[0], b.shape[0])
    ww = min(r.shape[1], g1.shape[1], g2.shape[1], b.shape[1])
    if hh <= 0 or ww <= 0:
        return np.repeat(raw_u8[:, :, None], 3, axis=2)

    rgb = np.empty((hh, ww, 3), dtype=np.uint8)
    rgb[:, :, 0] = r[:hh, :ww]
    rgb[:, :, 1] = ((g1[:hh, :ww] + g2[:hh, :ww]) >> 1).astype(np.uint8)
    rgb[:, :, 2] = b[:hh, :ww]
    return rgb


def _downsample_rgb_stride(rgb: np.ndarray, max_w: int, max_h: int) -> np.ndarray:
    if rgb.shape[1] <= max_w and rgb.shape[0] <= max_h:
        return rgb
    step = max(1, int(math.ceil(max(rgb.shape[1] / max_w, rgb.shape[0] / max_h))))
    return rgb[::step, ::step]


def make_preview_from_raw(
    raw_array: np.ndarray,
    pixel_format: int | str,
    max_w: int = 960,
    max_h: int = 640,
) -> np.ndarray:
    fmt = pixel_format_to_name(pixel_format)
    raw_u8 = _to_u8_for_preview(raw_array, fmt)

    if fmt.startswith("BAYERRG"):
        rgb = _bayer_quick_to_rgb_u8(raw_u8, "RGGB")
    elif fmt.startswith("BAYERGB"):
        rgb = _bayer_quick_to_rgb_u8(raw_u8, "GBRG")
    else:
        rgb = np.repeat(raw_u8[:, :, None], 3, axis=2)

    return _downsample_rgb_stride(rgb, max_w=max_w, max_h=max_h)


def save_preview_png(path: Path, rgb_u8: np.ndarray) -> None:
    if not PIL_AVAILABLE:
        raise RawDecodeError("Pillow is required for preview PNG export.")
    img = Image.fromarray(rgb_u8, mode="RGB")
    img.save(path)

