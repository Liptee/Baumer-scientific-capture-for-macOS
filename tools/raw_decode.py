#!/usr/bin/env python3
"""Декодирование RAW-буферов, используемое Hydra GUI."""

from __future__ import annotations

import numpy as np


PFNC_NAME_TO_INT: dict[str, int] = {
    "MONO8": 0x01080001,
    "BAYERRG8": 0x01080009,
    "MONO12": 0x01100005,
    "BAYERRG12": 0x01100011,
    "BAYERGB8": 0x0108000A,
    "BAYERGB12": 0x01100015,
}
PFNC_INT_TO_NAME: dict[int, str] = {value: name for name, value in PFNC_NAME_TO_INT.items()}
UNSUPPORTED_PACKED = {"MONO10PACKED", "MONO12PACKED", "BAYERGB10"}


class RawDecodeError(RuntimeError):
    """RAW-буфер нельзя декодировать без потери корректности."""


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


def decode_buffer_to_ndarray(raw: bytes, width: int, height: int, pixel_format: int | str) -> np.ndarray:
    if width <= 0 or height <= 0:
        raise RawDecodeError(f"Invalid shape: width={width}, height={height}")

    pixel_format_name = pixel_format_to_name(pixel_format)
    if pixel_format_name in UNSUPPORTED_PACKED:
        raise RawDecodeError(
            f"Packed format {pixel_format_name} is not supported. Use an unpacked pixel format."
        )

    pixel_count = width * height
    if pixel_format_name in {"MONO8", "BAYERRG8", "BAYERGB8"}:
        payload = _require_length(raw, pixel_count, pixel_format_name)
        return np.frombuffer(payload, dtype=np.uint8, count=pixel_count).reshape((height, width))
    if pixel_format_name in {"MONO12", "BAYERRG12", "BAYERGB12"}:
        payload = _require_length(raw, pixel_count * 2, pixel_format_name)
        return np.frombuffer(payload, dtype="<u2", count=pixel_count).reshape((height, width))

    if len(raw) >= pixel_count * 2:
        return np.frombuffer(raw[: pixel_count * 2], dtype="<u2", count=pixel_count).reshape((height, width))
    if len(raw) >= pixel_count:
        return np.frombuffer(raw[:pixel_count], dtype=np.uint8, count=pixel_count).reshape((height, width))
    raise RawDecodeError(
        f"Unsupported pixel format {pixel_format_name} with payload length {len(raw)}"
    )
