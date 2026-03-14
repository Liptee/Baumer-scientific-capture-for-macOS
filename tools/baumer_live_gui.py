#!/usr/bin/env python3
"""
Live viewer for Baumer GigE camera on macOS.

Features:
- Live preview
- Gain / Exposure Time controls
- Snapshot saving
"""

from __future__ import annotations

import argparse
import base64
import datetime as dt
import fcntl
import json
import math
import queue
import socket
import struct
import subprocess
import threading
import time
import tkinter as tk
import zlib
from dataclasses import dataclass
from pathlib import Path
from tkinter import ttk

SIOCGIFADDR = 0xC0206921
DEFAULT_PACKET_SIZE = 1440
DEFAULT_PACKET_DELAY = 1000
DEFAULT_PREVIEW_FPS = 25.0
DEFAULT_UI_POLL_MS = 10

try:
    from baumer_force_ip import send_force_ip as gvcp_force_ip
    from baumer_gvcp_explorer import discover as gvcp_discover

    AUTO_FIX_AVAILABLE = True
except Exception:
    gvcp_force_ip = None  # type: ignore[assignment]
    gvcp_discover = None  # type: ignore[assignment]
    AUTO_FIX_AVAILABLE = False

try:
    import numpy as np
    from PIL import Image, ImageTk

    FAST_PREVIEW_AVAILABLE = True
except Exception:
    np = None  # type: ignore[assignment]
    Image = None  # type: ignore[assignment]
    ImageTk = None  # type: ignore[assignment]
    FAST_PREVIEW_AVAILABLE = False

try:
    from scipy.ndimage import convolve, convolve1d

    SCIPY_DEMOSAIC_AVAILABLE = True
except Exception:
    convolve = None  # type: ignore[assignment]
    convolve1d = None  # type: ignore[assignment]
    SCIPY_DEMOSAIC_AVAILABLE = False

try:
    from camera_control import configure_scientific_mode, read_buffer_metadata, read_camera_runtime_metadata
    from capture_profiles import (
        CaptureProfile,
        get_capture_profile,
        list_profile_names,
        profile_from_dict,
        profile_to_dict,
        with_overrides,
    )
    from capture_session import SessionWriter
    from raw_decode import decode_buffer_to_ndarray, make_preview_from_raw, pixel_format_to_name

    SCIENTIFIC_CAPTURE_AVAILABLE = True
except Exception:
    configure_scientific_mode = None  # type: ignore[assignment]
    read_buffer_metadata = None  # type: ignore[assignment]
    read_camera_runtime_metadata = None  # type: ignore[assignment]
    CaptureProfile = None  # type: ignore[assignment]
    get_capture_profile = None  # type: ignore[assignment]
    list_profile_names = None  # type: ignore[assignment]
    profile_from_dict = None  # type: ignore[assignment]
    profile_to_dict = None  # type: ignore[assignment]
    with_overrides = None  # type: ignore[assignment]
    SessionWriter = None  # type: ignore[assignment]
    decode_buffer_to_ndarray = None  # type: ignore[assignment]
    make_preview_from_raw = None  # type: ignore[assignment]
    pixel_format_to_name = None  # type: ignore[assignment]
    SCIENTIFIC_CAPTURE_AVAILABLE = False


@dataclass
class FramePacket:
    width: int
    height: int
    pixel_format: int
    raw: bytes
    timestamp: float
    meta: dict[str, object] | None = None


def get_interface_ipv4(interface: str) -> str | None:
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        packed = struct.pack("256s", interface[:15].encode("ascii", errors="ignore"))
        data = fcntl.ioctl(sock.fileno(), SIOCGIFADDR, packed)
        return socket.inet_ntoa(data[20:24])
    except OSError:
        return None
    finally:
        sock.close()


def get_interface_netmask(interface: str) -> str | None:
    try:
        out = subprocess.check_output(["ifconfig", interface], text=True, stderr=subprocess.DEVNULL)
    except Exception:
        return None
    for line in out.splitlines():
        parts = line.strip().split()
        if len(parts) >= 4 and parts[0] == "inet" and parts[2] == "netmask":
            nm = parts[3]
            if nm.startswith("0x"):
                try:
                    val = int(nm, 16)
                    return socket.inet_ntoa(struct.pack(">I", val))
                except Exception:
                    return None
            return nm
    return None


def ip_to_int(ip: str) -> int:
    return int.from_bytes(socket.inet_aton(ip), "big", signed=False)


def same_subnet(ip_a: str, ip_b: str, mask: str) -> bool:
    try:
        ma = ip_to_int(mask)
        return (ip_to_int(ip_a) & ma) == (ip_to_int(ip_b) & ma)
    except Exception:
        return False


def suggest_camera_ip(host_ip: str) -> str:
    parts = host_ip.split(".")
    if len(parts) != 4:
        return "192.168.77.2"
    try:
        octets = [int(x) for x in parts]
    except ValueError:
        return "192.168.77.2"
    base = octets[:3]
    candidate = 2
    if octets[3] == candidate:
        candidate = 3
    return f"{base[0]}.{base[1]}.{base[2]}.{candidate}"


def ipv4_to_u32(value: str) -> int:
    return int.from_bytes(socket.inet_aton(value), "big", signed=False)


def downsample_mono(raw: bytes, width: int, height: int, max_w: int, max_h: int) -> tuple[int, int, bytes]:
    if width <= 0 or height <= 0:
        return 0, 0, b""
    step = max(1, math.ceil(max(width / max_w, height / max_h)))
    out_w = (width + step - 1) // step
    out_h = (height + step - 1) // step
    out = bytearray(out_w * out_h)
    idx = 0
    for y in range(0, height, step):
        row = raw[y * width : (y + 1) * width]
        sampled = row[::step]
        ln = len(sampled)
        out[idx : idx + ln] = sampled
        idx += ln
    return out_w, out_h, bytes(out[: idx])


def downsample_bayer_rg8_to_rgb(
    raw: bytes, width: int, height: int, max_w: int, max_h: int
) -> tuple[int, int, bytes]:
    if width <= 0 or height <= 0:
        return 0, 0, b""
    step = max(1, math.ceil(max(width / max_w, height / max_h)))
    out_w = (width + step - 1) // step
    out_h = (height + step - 1) // step
    src = memoryview(raw)
    w_last = width - 1
    h_last = height - 1
    out = bytearray(out_w * out_h * 3)
    j = 0

    def pix(xx: int, yy: int) -> int:
        if xx < 0:
            xx = 0
        elif xx > w_last:
            xx = w_last
        if yy < 0:
            yy = 0
        elif yy > h_last:
            yy = h_last
        return src[yy * width + xx]

    for oy in range(out_h):
        y = oy * step
        y_even = (y & 1) == 0
        for ox in range(out_w):
            x = ox * step
            x_even = (x & 1) == 0
            c = pix(x, y)
            if y_even and x_even:
                # R
                r = c
                g = (pix(x - 1, y) + pix(x + 1, y) + pix(x, y - 1) + pix(x, y + 1)) // 4
                b = (pix(x - 1, y - 1) + pix(x + 1, y - 1) + pix(x - 1, y + 1) + pix(x + 1, y + 1)) // 4
            elif (not y_even) and (not x_even):
                # B
                b = c
                g = (pix(x - 1, y) + pix(x + 1, y) + pix(x, y - 1) + pix(x, y + 1)) // 4
                r = (pix(x - 1, y - 1) + pix(x + 1, y - 1) + pix(x - 1, y + 1) + pix(x + 1, y + 1)) // 4
            elif y_even and (not x_even):
                # G on R row
                g = c
                r = (pix(x - 1, y) + pix(x + 1, y)) // 2
                b = (pix(x, y - 1) + pix(x, y + 1)) // 2
            else:
                # G on B row
                g = c
                r = (pix(x, y - 1) + pix(x, y + 1)) // 2
                b = (pix(x - 1, y) + pix(x + 1, y)) // 2
            out[j] = r
            out[j + 1] = g
            out[j + 2] = b
            j += 3
    return out_w, out_h, bytes(out)


def masks_cfa_bayer(shape: tuple[int, int], pattern: str) -> tuple["np.ndarray", "np.ndarray", "np.ndarray"]:
    if np is None:
        raise RuntimeError("NumPy is required for Bayer masks")
    h, w = shape
    y, x = np.indices((h, w))
    y_even = (y % 2) == 0
    x_even = (x % 2) == 0
    p = pattern.upper()
    if p == "RGGB":
        r = y_even & x_even
        b = (~y_even) & (~x_even)
    elif p == "BGGR":
        b = y_even & x_even
        r = (~y_even) & (~x_even)
    elif p == "GRBG":
        r = y_even & (~x_even)
        b = (~y_even) & x_even
    elif p == "GBRG":
        r = (~y_even) & x_even
        b = y_even & (~x_even)
    else:
        raise ValueError(f"Unsupported Bayer pattern: {pattern}")
    g = ~(r | b)
    return r.astype(np.float32), g.astype(np.float32), b.astype(np.float32)


def _cnv_h(x: "np.ndarray", kernel: "np.ndarray") -> "np.ndarray":
    if convolve1d is None:
        raise RuntimeError("scipy.ndimage.convolve1d is unavailable")
    return convolve1d(x, kernel, mode="mirror")


def _cnv_v(x: "np.ndarray", kernel: "np.ndarray") -> "np.ndarray":
    if convolve1d is None:
        raise RuntimeError("scipy.ndimage.convolve1d is unavailable")
    return convolve1d(x, kernel, mode="mirror", axis=0)


def demosaic_bayer_menon2007(cfa: "np.ndarray", pattern: str = "RGGB") -> "np.ndarray":
    if np is None or convolve is None or convolve1d is None:
        raise RuntimeError("NumPy/SciPy demosaicing dependencies are unavailable")

    cfa = np.asarray(cfa, dtype=np.float32)
    r_m, g_m, b_m = masks_cfa_bayer(cfa.shape, pattern)

    h_0 = np.asarray([0.0, 0.5, 0.0, 0.5, 0.0], dtype=np.float32)
    h_1 = np.asarray([-0.25, 0.0, 0.5, 0.0, -0.25], dtype=np.float32)

    r = cfa * r_m
    g = cfa * g_m
    b = cfa * b_m

    g_h = np.where(g_m == 0, _cnv_h(cfa, h_0) + _cnv_h(cfa, h_1), g)
    g_v = np.where(g_m == 0, _cnv_v(cfa, h_0) + _cnv_v(cfa, h_1), g)

    c_h = np.where(r_m == 1, r - g_h, 0)
    c_h = np.where(b_m == 1, b - g_h, c_h)

    c_v = np.where(r_m == 1, r - g_v, 0)
    c_v = np.where(b_m == 1, b - g_v, c_v)

    d_h = np.abs(c_h - np.pad(c_h, ((0, 0), (0, 2)), mode="reflect")[:, 2:])
    d_v = np.abs(c_v - np.pad(c_v, ((0, 2), (0, 0)), mode="reflect")[2:, :])

    k = np.asarray(
        [
            [0.0, 0.0, 1.0, 0.0, 1.0],
            [0.0, 0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 3.0, 0.0, 3.0],
            [0.0, 0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )

    d_h = convolve(d_h, k, mode="constant")
    d_v = convolve(d_v, np.transpose(k), mode="constant")

    directional_mask = d_v >= d_h
    g = np.where(directional_mask, g_h, g_v)
    m = np.where(directional_mask, 1.0, 0.0)

    # Red rows / blue rows masks.
    r_r = np.transpose(np.any(r_m == 1, axis=1)[None]) * np.ones(r.shape, dtype=np.float32)
    b_r = np.transpose(np.any(b_m == 1, axis=1)[None]) * np.ones(b.shape, dtype=np.float32)

    k_b = np.asarray([0.5, 0.0, 0.5], dtype=np.float32)

    r = np.where(
        np.logical_and(g_m == 1, r_r == 1),
        g + _cnv_h(r, k_b) - _cnv_h(g, k_b),
        r,
    )
    r = np.where(
        np.logical_and(g_m == 1, b_r == 1),
        g + _cnv_v(r, k_b) - _cnv_v(g, k_b),
        r,
    )

    b = np.where(
        np.logical_and(g_m == 1, b_r == 1),
        g + _cnv_h(b, k_b) - _cnv_h(g, k_b),
        b,
    )
    b = np.where(
        np.logical_and(g_m == 1, r_r == 1),
        g + _cnv_v(b, k_b) - _cnv_v(g, k_b),
        b,
    )

    r = np.where(
        np.logical_and(b_r == 1, b_m == 1),
        np.where(
            m == 1,
            b + _cnv_h(r, k_b) - _cnv_h(b, k_b),
            b + _cnv_v(r, k_b) - _cnv_v(b, k_b),
        ),
        r,
    )
    b = np.where(
        np.logical_and(r_r == 1, r_m == 1),
        np.where(
            m == 1,
            r + _cnv_h(b, k_b) - _cnv_h(r, k_b),
            r + _cnv_v(b, k_b) - _cnv_v(r, k_b),
        ),
        b,
    )

    return np.stack([r, g, b], axis=-1)


def bayer_rg8_to_rgb(raw: bytes, width: int, height: int) -> bytes:
    if width <= 0 or height <= 0:
        return b""
    expected = width * height
    if len(raw) < expected:
        return b""

    if np is not None and SCIPY_DEMOSAIC_AVAILABLE:
        try:
            cfa = np.frombuffer(raw, dtype=np.uint8, count=expected).reshape((height, width))
            rgb_f = demosaic_bayer_menon2007(cfa, "RGGB")
            rgb_u8 = np.clip(np.rint(rgb_f), 0, 255).astype(np.uint8)
            return rgb_u8.tobytes()
        except Exception:
            # Fall back to bilinear demosaicing if Menon path fails.
            pass

    out_w, out_h, rgb = downsample_bayer_rg8_to_rgb(raw, width, height, width, height)
    if out_w != width or out_h != height:
        return b""
    return rgb


def save_pgm(path: Path, width: int, height: int, image_bytes: bytes) -> None:
    header = f"P5\n{width} {height}\n255\n".encode("ascii")
    path.write_bytes(header + image_bytes)


def _png_chunk(chunk_type: bytes, payload: bytes) -> bytes:
    length = struct.pack(">I", len(payload))
    crc = zlib.crc32(chunk_type + payload) & 0xFFFFFFFF
    return length + chunk_type + payload + struct.pack(">I", crc)


def mono_to_png_bytes(mono: bytes, width: int, height: int, level: int = 3) -> bytes:
    # 8-bit grayscale PNG (color type 0), one byte per pixel.
    if width <= 0 or height <= 0:
        return b""
    expected = width * height
    if len(mono) < expected:
        return b""
    scan = bytearray()
    row_stride = width
    for y in range(height):
        scan.append(0)  # filter type 0
        start = y * row_stride
        scan.extend(mono[start : start + row_stride])
    compressed = zlib.compress(bytes(scan), level=level)
    png_sig = b"\x89PNG\r\n\x1a\n"
    ihdr = struct.pack(">IIBBBBB", width, height, 8, 0, 0, 0, 0)
    return png_sig + _png_chunk(b"IHDR", ihdr) + _png_chunk(b"IDAT", compressed) + _png_chunk(b"IEND", b"")


def rgb_to_png_bytes(rgb: bytes, width: int, height: int, level: int = 3) -> bytes:
    if width <= 0 or height <= 0:
        return b""
    expected = width * height * 3
    if len(rgb) < expected:
        return b""
    scan = bytearray()
    row_stride = width * 3
    for y in range(height):
        scan.append(0)  # filter type 0
        start = y * row_stride
        scan.extend(rgb[start : start + row_stride])
    compressed = zlib.compress(bytes(scan), level=level)
    png_sig = b"\x89PNG\r\n\x1a\n"
    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)  # color type 2 = RGB
    return png_sig + _png_chunk(b"IHDR", ihdr) + _png_chunk(b"IDAT", compressed) + _png_chunk(b"IEND", b"")


def is_bayer_rg8(pixel_format: int) -> bool:
    # PFNC BayerRG8
    return pixel_format == 0x01080009


def preview_bayer_rg8_fast(raw: bytes, width: int, height: int, max_w: int, max_h: int) -> tuple[int, int, bytes]:
    # Fast preview path:
    # Sample Bayer RGGB in coarse 2x2 blocks:
    # R = (x,y), G = avg((x+1,y),(x,y+1)), B = (x+1,y+1)
    if width < 2 or height < 2:
        return 0, 0, b""
    src = memoryview(raw)
    scale = max(1.0, max((width / 2) / max_w, (height / 2) / max_h))
    block = max(1, int(math.ceil(scale)))
    step = block * 2
    out_w = ((width - 2) // step) + 1
    out_h = ((height - 2) // step) + 1
    out = bytearray(out_w * out_h * 3)
    j = 0
    for y in range(0, height - 1, step):
        row0 = y * width
        row1 = (y + 1) * width
        for x in range(0, width - 1, step):
            r = src[row0 + x]
            g = (src[row0 + x + 1] + src[row1 + x]) >> 1
            b = src[row1 + x + 1]
            out[j] = r
            out[j + 1] = g
            out[j + 2] = b
            j += 3
    return out_w, out_h, bytes(out)


def mono_to_rgb_bytes(mono: bytes) -> bytes:
    out = bytearray(len(mono) * 3)
    j = 0
    for v in mono:
        out[j] = v
        out[j + 1] = v
        out[j + 2] = v
        j += 3
    return bytes(out)


def rotate_rgb(rgb: bytes, width: int, height: int, degrees: int) -> tuple[int, int, bytes]:
    deg = degrees % 360
    if deg == 0:
        return width, height, rgb
    src = memoryview(rgb)
    if deg == 180:
        out = bytearray(width * height * 3)
        j = 0
        for yd in range(height):
            sy = height - 1 - yd
            row = sy * width * 3
            for xd in range(width):
                sx = width - 1 - xd
                si = row + sx * 3
                out[j] = src[si]
                out[j + 1] = src[si + 1]
                out[j + 2] = src[si + 2]
                j += 3
        return width, height, bytes(out)
    if deg == 90:
        new_w, new_h = height, width
        out = bytearray(new_w * new_h * 3)
        j = 0
        for yd in range(new_h):
            for xd in range(new_w):
                sx = yd
                sy = height - 1 - xd
                si = (sy * width + sx) * 3
                out[j] = src[si]
                out[j + 1] = src[si + 1]
                out[j + 2] = src[si + 2]
                j += 3
        return new_w, new_h, bytes(out)
    if deg == 270:
        new_w, new_h = height, width
        out = bytearray(new_w * new_h * 3)
        j = 0
        for yd in range(new_h):
            for xd in range(new_w):
                sx = width - 1 - yd
                sy = xd
                si = (sy * width + sx) * 3
                out[j] = src[si]
                out[j + 1] = src[si + 1]
                out[j + 2] = src[si + 2]
                j += 3
        return new_w, new_h, bytes(out)
    return width, height, rgb


def resize_rgb_nearest(rgb: bytes, width: int, height: int, zoom: float) -> tuple[int, int, bytes]:
    if zoom <= 0:
        zoom = 1.0
    if abs(zoom - 1.0) < 1e-6:
        return width, height, rgb
    new_w = max(1, int(round(width * zoom)))
    new_h = max(1, int(round(height * zoom)))
    src = memoryview(rgb)
    x_map = [min(width - 1, int(x / zoom)) for x in range(new_w)]
    y_map = [min(height - 1, int(y / zoom)) for y in range(new_h)]
    out = bytearray(new_w * new_h * 3)
    j = 0
    for sy in y_map:
        row = sy * width * 3
        for sx in x_map:
            si = row + sx * 3
            out[j] = src[si]
            out[j + 1] = src[si + 1]
            out[j + 2] = src[si + 2]
            j += 3
    return new_w, new_h, bytes(out)


def _rotate_pil_for_deg(image, degrees: int):
    deg = degrees % 360
    if deg == 90:
        # Clockwise 90.
        return image.transpose(Image.Transpose.ROTATE_270)
    if deg == 180:
        return image.transpose(Image.Transpose.ROTATE_180)
    if deg == 270:
        # Clockwise 270 == CCW 90.
        return image.transpose(Image.Transpose.ROTATE_90)
    return image


def build_preview_image_fast(
    raw: bytes,
    width: int,
    height: int,
    pixel_format: int,
    rotation_deg: int,
    zoom: float,
    max_w: int = 960,
    max_h: int = 640,
):
    if not FAST_PREVIEW_AVAILABLE or width <= 0 or height <= 0:
        return None, 0, 0, "Mono"
    expected = width * height
    if len(raw) < expected:
        return None, 0, 0, "Mono"

    arr = np.frombuffer(raw, dtype=np.uint8, count=expected).reshape((height, width))

    if is_bayer_rg8(pixel_format) and width >= 2 and height >= 2:
        r = arr[0::2, 0::2]
        g1 = arr[0::2, 1::2].astype(np.uint16)
        g2 = arr[1::2, 0::2].astype(np.uint16)
        b = arr[1::2, 1::2]
        hh = min(r.shape[0], g1.shape[0], g2.shape[0], b.shape[0])
        ww = min(r.shape[1], g1.shape[1], g2.shape[1], b.shape[1])
        if hh <= 0 or ww <= 0:
            return None, 0, 0, "RGB"
        rgb = np.empty((hh, ww, 3), dtype=np.uint8)
        rgb[:, :, 0] = r[:hh, :ww]
        rgb[:, :, 1] = ((g1[:hh, :ww] + g2[:hh, :ww]) >> 1).astype(np.uint8)
        rgb[:, :, 2] = b[:hh, :ww]
        mode_label = "RGB"
    else:
        rgb = np.repeat(arr[:, :, None], 3, axis=2)
        mode_label = "Mono"

    # Use stride decimation for fit-to-window (faster than per-frame PIL resize).
    ds = max(1, int(math.ceil(max(rgb.shape[1] / max_w, rgb.shape[0] / max_h))))
    if ds > 1:
        rgb = rgb[::ds, ::ds]

    rot = rotation_deg % 360
    if rot in (90, 180, 270):
        # np.rot90 uses CCW turns: 90CW=3, 180=2, 270CW=1.
        k = 0
        if rot == 90:
            k = 3
        elif rot == 180:
            k = 2
        elif rot == 270:
            k = 1
        rgb = np.rot90(rgb, k)

    img = Image.fromarray(rgb, mode="RGB")

    zoom = max(0.25, min(4.0, float(zoom)))
    if abs(zoom - 1.0) > 1e-6:
        zw = max(1, int(round(img.width * zoom)))
        zh = max(1, int(round(img.height * zoom)))
        img = img.resize((zw, zh), resample=Image.Resampling.NEAREST)

    return img, int(img.width), int(img.height), mode_label


class CameraWorker(threading.Thread):
    def __init__(
        self,
        interface: str,
        camera_ip: str,
        event_q: queue.Queue,
        cmd_q: queue.Queue,
        packet_size: int,
        packet_delay: int,
        buffers: int,
    ) -> None:
        super().__init__(daemon=True)
        self.interface = interface
        self.camera_ip = camera_ip
        self.event_q = event_q
        self.cmd_q = cmd_q
        self.packet_size = packet_size
        self.packet_delay = packet_delay
        self.buffers = buffers
        self.stop_event = threading.Event()
        self.stream_stall_timeout_s = 2.5
        self.stream_restart_cooldown_s = 1.2
        self._last_good_frame_ts = 0.0
        self._last_restart_ts = 0.0
        self._last_bad_status_log_ts = 0.0

    def stop(self) -> None:
        self.stop_event.set()

    def _emit(self, kind: str, payload: object | None = None) -> None:
        if kind == "frame":
            if self.event_q.qsize() > 3:
                return
        try:
            self.event_q.put_nowait((kind, payload))
        except queue.Full:
            pass

    def _read_controls(self, camera) -> dict[str, float] | None:
        out: dict[str, float] = {}
        try:
            g_min, g_max = camera.get_gain_bounds()
            out["gain_min"] = float(g_min)
            out["gain_max"] = float(g_max)
            out["gain"] = float(camera.get_gain())
        except Exception:
            pass
        try:
            e_min, e_max = camera.get_exposure_time_bounds()
            out["exposure_min"] = float(e_min)
            out["exposure_max"] = float(e_max)
            out["exposure"] = float(camera.get_exposure_time())
        except Exception:
            pass
        return out if out else None

    @staticmethod
    def _is_access_denied_error(exc: Exception) -> bool:
        text = str(exc).lower()
        return "access-denied" in text or "(17)" in text or "access denied" in text

    def _set_auto_controls_off(self, camera) -> None:
        try:
            camera.set_string("GainAuto", "Off")
        except Exception:
            pass
        try:
            camera.set_string("ExposureAuto", "Off")
        except Exception:
            pass
        try:
            camera.set_gain_auto(0)
        except Exception:
            pass
        try:
            camera.set_exposure_time_auto(0)
        except Exception:
            pass
        try:
            camera.set_string("ExposureMode", "Timed")
        except Exception:
            pass
        try:
            camera.set_exposure_mode(1)
        except Exception:
            pass
        try:
            camera.set_string("TriggerMode", "Off")
        except Exception:
            pass

    def _try_take_control(self, camera) -> None:
        try:
            device = camera.get_device()
        except Exception:
            device = None
        if device is None:
            return
        if not hasattr(device, "is_controller") or not hasattr(device, "take_control"):
            return
        try:
            if not bool(device.is_controller()):
                self._emit("status", "Camera control lost, requesting control...")
                device.take_control()
                time.sleep(0.05)
        except Exception:
            pass

    def _recover_after_access_denied(self, camera, label: str) -> bool:
        paused = False
        self._emit("status", f"{label}: access denied, recovering control...")
        try:
            camera.stop_acquisition()
            paused = True
        except Exception:
            paused = False
        self._try_take_control(camera)
        self._set_auto_controls_off(camera)
        return paused

    def _write_with_recovery(self, camera, label: str, writer) -> None:
        paused = False
        try:
            for attempt in (1, 2):
                try:
                    writer()
                    return
                except Exception as exc:
                    if attempt == 1 and self._is_access_denied_error(exc):
                        paused = self._recover_after_access_denied(camera, label)
                        continue
                    raise
        finally:
            if paused:
                try:
                    camera.start_acquisition()
                except Exception as exc:
                    self._emit("status", f"{label}: failed to resume acquisition: {exc}")

    def _maybe_restart_acquisition(self, camera, reason: str, now_mono: float) -> None:
        if (now_mono - self._last_good_frame_ts) < self.stream_stall_timeout_s:
            return
        if (now_mono - self._last_restart_ts) < self.stream_restart_cooldown_s:
            return
        self._last_restart_ts = now_mono
        self._emit("status", f"Stream stalled ({reason}), restarting acquisition...")
        try:
            camera.stop_acquisition()
        except Exception:
            pass
        time.sleep(0.03)
        try:
            camera.start_acquisition()
            self._last_good_frame_ts = time.monotonic()
            self._emit("status", "Acquisition restarted")
        except Exception as exc:
            self._emit("status", f"Acquisition restart failed: {exc}")

    def _apply_command(self, camera, cmd: str, value: object | None) -> None:
        if cmd == "set_gain" and value is not None:
            gain = float(value)
            self._write_with_recovery(camera, "set_gain", lambda: camera.set_gain(gain))
            self._emit("status", f"Gain set to {value:.2f}")
        elif cmd == "set_exposure" and value is not None:
            exposure = float(value)
            self._write_with_recovery(camera, "set_exposure", lambda: camera.set_exposure_time(exposure))
            self._emit("status", f"Exposure set to {value:.1f} us")
        elif cmd == "configure_profile" and value is not None:
            if not SCIENTIFIC_CAPTURE_AVAILABLE or profile_from_dict is None or configure_scientific_mode is None:
                self._emit("status", "Scientific mode configuration is unavailable")
                return
            if not isinstance(value, dict):
                self._emit("status", "Scientific mode config error: invalid profile payload")
                return
            try:
                profile = profile_from_dict(value)
                result = configure_scientific_mode(camera, profile, logger=lambda m: self._emit("status", m))
                self._emit("camera_config", result)
                self._emit("status", f"Scientific mode ready: PixelFormat={result.get('selected_pixel_format')}")
            except Exception as exc:
                self._emit("status", f"Scientific mode config failed: {exc}")
                return
        elif cmd == "refresh_controls":
            pass
        else:
            return
        controls = self._read_controls(camera)
        if controls:
            self._emit("controls", controls)

    def run(self) -> None:
        camera = None
        stream = None
        try:
            import gi

            gi.require_version("Aravis", "0.8")
            from gi.repository import Aravis  # type: ignore

            Aravis.GvInterface.set_discovery_interface_name(self.interface)
            camera = Aravis.Camera.new(self.camera_ip)
            if camera is None:
                raise RuntimeError(f"Camera not found at {self.camera_ip}")
            self._try_take_control(camera)

            camera.gv_set_stream_options(Aravis.GvStreamOption.PACKET_SOCKET_DISABLED)
            camera.gv_set_packet_size_adjustment(Aravis.GvPacketSizeAdjustment.NEVER)
            if self.packet_size > 0:
                camera.gv_set_packet_size(self.packet_size)

            stream = camera.create_stream(None, None)
            if stream is None:
                raise RuntimeError("Failed to create stream")

            if_ip = get_interface_ipv4(self.interface)
            if if_ip:
                try:
                    if hasattr(stream, "get_port"):
                        camera.set_integer("GevSCPHostPort", int(stream.get_port()))
                except Exception:
                    pass
                try:
                    camera.set_integer("GevSCDA", ipv4_to_u32(if_ip))
                except Exception:
                    pass

            try:
                camera.set_string("TriggerMode", "Off")
            except Exception:
                pass
            self._set_auto_controls_off(camera)
            if self.packet_size > 0:
                try:
                    camera.set_integer("GevSCPSPacketSize", int(self.packet_size))
                except Exception:
                    pass
            if self.packet_delay > 0:
                try:
                    camera.set_integer("GevSCPD", int(self.packet_delay))
                except Exception:
                    pass

            payload = int(camera.get_payload())
            if payload <= 0:
                raise RuntimeError("Invalid payload size")

            for _ in range(max(2, self.buffers)):
                stream.push_buffer(Aravis.Buffer.new_allocate(payload))

            controls = self._read_controls(camera)
            runtime_meta = read_camera_runtime_metadata(camera) if read_camera_runtime_metadata is not None else {}
            serial = ""
            try:
                serial = str(camera.get_device_serial_number())
            except Exception:
                pass

            self._emit(
                "connected",
                {
                    "vendor": str(camera.get_vendor_name()),
                    "model": str(camera.get_model_name()),
                    "serial": serial,
                    "pixel_format": str(camera.get_pixel_format_as_string()),
                    "payload": payload,
                    "controls": controls,
                    "runtime": runtime_meta,
                },
            )

            try:
                camera.set_acquisition_mode(Aravis.AcquisitionMode.CONTINUOUS)
            except Exception:
                pass
            camera.start_acquisition()
            now_mono = time.monotonic()
            self._last_good_frame_ts = now_mono
            self._last_restart_ts = now_mono
            frame_counter = 0
            runtime_cache = runtime_meta if isinstance(runtime_meta, dict) else {}

            success_status = int(Aravis.BufferStatus.SUCCESS)
            while not self.stop_event.is_set():
                while True:
                    try:
                        cmd, value = self.cmd_q.get_nowait()
                    except queue.Empty:
                        break
                    try:
                        self._apply_command(camera, cmd, value)
                    except Exception as exc:
                        self._emit("status", f"Command error ({cmd}): {exc}")

                buffer = stream.timeout_pop_buffer(200)
                if buffer is None:
                    self._maybe_restart_acquisition(camera, "no buffers", time.monotonic())
                    continue
                status = int(buffer.get_status())
                if status == success_status:
                    try:
                        raw = bytes(buffer.get_image_data())
                        w = int(buffer.get_image_width())
                        h = int(buffer.get_image_height())
                        pf = int(buffer.get_image_pixel_format())
                        frame_counter += 1
                        if read_camera_runtime_metadata is not None and (frame_counter % 15) == 0:
                            try:
                                runtime_cache = read_camera_runtime_metadata(camera)
                            except Exception:
                                pass
                        meta = read_buffer_metadata(buffer) if read_buffer_metadata is not None else {}
                        if isinstance(meta, dict):
                            meta["width"] = w
                            meta["height"] = h
                            meta["pixel_format_int"] = pf
                            if "pixel_format_name" not in meta:
                                try:
                                    meta["pixel_format_name"] = str(camera.get_pixel_format_as_string())
                                except Exception:
                                    pass
                            if isinstance(runtime_cache, dict):
                                for key in ("exposure_us", "gain_db", "black_level"):
                                    if key in runtime_cache:
                                        meta[key] = runtime_cache.get(key)
                        self._last_good_frame_ts = time.monotonic()
                        self._emit("frame", FramePacket(w, h, pf, raw, time.time(), meta))
                    except Exception as exc:
                        self._emit("status", f"Frame decode error: {exc}")
                else:
                    now_bad = time.monotonic()
                    if (now_bad - self._last_bad_status_log_ts) > 1.5:
                        self._last_bad_status_log_ts = now_bad
                        self._emit("status", f"Bad frame status: {status}")
                    self._maybe_restart_acquisition(camera, f"buffer status {status}", now_bad)
                stream.push_buffer(buffer)
        except Exception as exc:
            self._emit("error", str(exc))
        finally:
            if camera is not None:
                try:
                    camera.stop_acquisition()
                except Exception:
                    pass
            self._emit("disconnected", None)


class BaumerLiveApp(tk.Tk):
    def __init__(
        self,
        interface: str,
        camera_ip: str,
        snapshot_dir: Path,
        packet_size: int = DEFAULT_PACKET_SIZE,
        packet_delay: int = DEFAULT_PACKET_DELAY,
        preview_fps: float = DEFAULT_PREVIEW_FPS,
        ui_poll_ms: int = DEFAULT_UI_POLL_MS,
    ) -> None:
        super().__init__()
        self.title("Baumer Live Viewer")
        self.geometry("1260x820")

        self.snapshot_dir = snapshot_dir
        self.snapshot_dir.mkdir(parents=True, exist_ok=True)

        self.interface_var = tk.StringVar(value=interface)
        self.camera_var = tk.StringVar(value=camera_ip)
        self.status_var = tk.StringVar(value="Idle")
        self.info_var = tk.StringVar(value="-")
        self.frame_var = tk.StringVar(value="-")
        self.profile_names = list_profile_names() if (SCIENTIFIC_CAPTURE_AVAILABLE and list_profile_names) else ["scene_capture"]
        self.capture_profile_var = tk.StringVar(value=self.profile_names[0])
        self.session_dir_var = tk.StringVar(value=str(self.snapshot_dir))
        self.session_frames_var = tk.StringVar(value="1")
        self.session_notes_var = tk.StringVar(value="")

        self.gain_var = tk.DoubleVar(value=0.0)
        self.exposure_var = tk.DoubleVar(value=10000.0)
        self.gain_entry_var = tk.StringVar(value="0.0")
        self.exposure_entry_var = tk.StringVar(value="10000.0")
        self.zoom_var = tk.DoubleVar(value=1.0)
        self.zoom_entry_var = tk.StringVar(value="1.00")
        self.rotation_var = tk.StringVar(value="0")
        self.gain_bounds = (0.0, 24.0)
        self.exposure_bounds = (100.0, 500000.0)

        self.worker: CameraWorker | None = None
        self.event_q: queue.Queue = queue.Queue(maxsize=64)
        self.cmd_q: queue.Queue = queue.Queue(maxsize=64)
        self.last_frame: FramePacket | None = None
        self.preview_photo: tk.PhotoImage | None = None
        self.last_render_ts = 0.0
        self.render_interval_s = 1.0 / max(1.0, float(preview_fps))
        self.ui_poll_ms = max(5, int(ui_poll_ms))
        self.packet_size = max(576, int(packet_size))
        self.packet_delay = max(0, int(packet_delay))
        self.max_events_per_poll = 24
        self.auto_fix_running = False
        self.auto_apply_delay_ms = 180
        self._gain_apply_after_id: str | None = None
        self._exposure_apply_after_id: str | None = None
        self._rx_fps = 0.0
        self._rx_frames = 0
        self._rx_fps_window_ts = time.monotonic()
        self._render_fps = 0.0
        self._render_frames = 0
        self._render_fps_window_ts = time.monotonic()
        self.camera_info: dict[str, object] = {}
        self.last_camera_config: dict[str, object] | None = None
        self.active_session_writer: SessionWriter | None = None
        self.active_session_profile: CaptureProfile | None = None
        self.session_remaining_frames = 0
        self.session_next_frame_index = 1

        self._build_ui()
        self._on_profile_change(None)  # initialize frame count from selected profile
        self.protocol("WM_DELETE_WINDOW", self._on_close)
        self.after(self.ui_poll_ms, self._poll_events)

    def _build_ui(self) -> None:
        root = ttk.Frame(self, padding=8)
        root.pack(fill=tk.BOTH, expand=True)
        root.columnconfigure(0, weight=1)
        root.columnconfigure(1, weight=0)
        root.rowconfigure(1, weight=1)

        top = ttk.Frame(root)
        top.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 8))
        for col in range(12):
            top.columnconfigure(col, weight=0)
        top.columnconfigure(3, weight=1)

        ttk.Label(top, text="Interface").grid(row=0, column=0, sticky="w")
        ttk.Entry(top, textvariable=self.interface_var, width=10).grid(row=0, column=1, padx=(6, 10))
        ttk.Label(top, text="Camera IP").grid(row=0, column=2, sticky="w")
        ttk.Entry(top, textvariable=self.camera_var, width=18).grid(row=0, column=3, sticky="w", padx=(6, 10))

        ttk.Button(top, text="Connect", command=self._connect).grid(row=0, column=4, padx=4)
        ttk.Button(top, text="Disconnect", command=self._disconnect).grid(row=0, column=5, padx=4)
        ttk.Button(top, text="Refresh Controls", command=self._refresh_controls).grid(row=0, column=6, padx=4)
        ttk.Button(top, text="Snapshot RAW", command=self._snapshot_scientific).grid(row=0, column=7, padx=4)
        ttk.Button(top, text="Auto Find/Fix", command=self._auto_find_fix).grid(row=0, column=8, padx=4)

        preview_frame = ttk.LabelFrame(root, text="Live")
        preview_frame.grid(row=1, column=0, sticky="nsew", padx=(0, 8))
        preview_frame.rowconfigure(0, weight=1)
        preview_frame.columnconfigure(0, weight=1)

        self.preview_canvas = tk.Canvas(preview_frame, background="#101010", highlightthickness=0)
        self.preview_canvas.grid(row=0, column=0, sticky="nsew")
        self.preview_canvas.bind("<Configure>", self._on_canvas_resize)
        self.preview_canvas.bind("<MouseWheel>", self._on_mouse_wheel)
        self.canvas_image_id = self.preview_canvas.create_image(0, 0, anchor="center")
        self.canvas_text_id = self.preview_canvas.create_text(
            0,
            0,
            text="No frame",
            fill="#d0d0d0",
            font=("Helvetica", 16),
        )

        right_panel = ttk.Frame(root)
        right_panel.grid(row=1, column=1, sticky="ns")
        right_panel.rowconfigure(0, weight=1)
        right_panel.columnconfigure(0, weight=1)

        self.right_canvas = tk.Canvas(right_panel, highlightthickness=0, width=360)
        self.right_canvas.grid(row=0, column=0, sticky="ns")
        self.right_scrollbar = ttk.Scrollbar(right_panel, orient=tk.VERTICAL, command=self.right_canvas.yview)
        self.right_scrollbar.grid(row=0, column=1, sticky="ns")
        self.right_canvas.configure(yscrollcommand=self.right_scrollbar.set)

        self.right_inner = ttk.Frame(self.right_canvas)
        self.right_canvas_window = self.right_canvas.create_window((0, 0), window=self.right_inner, anchor="nw")
        self.right_canvas.bind("<Configure>", self._on_right_canvas_configure)
        self.right_inner.bind("<Configure>", self._on_right_frame_configure)
        self.right_canvas.bind("<Enter>", self._bind_right_mousewheel)
        self.right_canvas.bind("<Leave>", self._unbind_right_mousewheel)

        right = ttk.LabelFrame(self.right_inner, text="Controls")
        right.grid(row=0, column=0, sticky="ew")
        right.columnconfigure(0, weight=1)

        ttk.Label(right, text="Gain").grid(row=0, column=0, sticky="w", padx=8, pady=(10, 2))
        self.gain_scale = ttk.Scale(
            right,
            from_=self.gain_bounds[0],
            to=self.gain_bounds[1],
            variable=self.gain_var,
            orient=tk.HORIZONTAL,
            length=280,
            command=self._on_gain_slide,
        )
        self.gain_scale.grid(row=1, column=0, padx=8, sticky="ew")
        gain_entry_row = ttk.Frame(right)
        gain_entry_row.grid(row=2, column=0, padx=8, pady=(4, 12), sticky="ew")
        gain_entry_row.columnconfigure(0, weight=1)
        self.gain_entry = ttk.Entry(gain_entry_row, textvariable=self.gain_entry_var)
        self.gain_entry.grid(row=0, column=0, sticky="ew")
        self.gain_entry.bind("<Return>", self._apply_gain_from_event)
        self.gain_entry.bind("<FocusOut>", self._apply_gain_from_event)
        self.gain_entry.bind("<KeyRelease>", self._on_gain_entry_edit)

        ttk.Label(right, text="Exposure Time (us)").grid(row=3, column=0, sticky="w", padx=8, pady=(0, 2))
        self.exposure_scale = ttk.Scale(
            right,
            from_=self.exposure_bounds[0],
            to=self.exposure_bounds[1],
            variable=self.exposure_var,
            orient=tk.HORIZONTAL,
            length=280,
            command=self._on_exposure_slide,
        )
        self.exposure_scale.grid(row=4, column=0, padx=8, sticky="ew")
        exposure_entry_row = ttk.Frame(right)
        exposure_entry_row.grid(row=5, column=0, padx=8, pady=(4, 12), sticky="ew")
        exposure_entry_row.columnconfigure(0, weight=1)
        self.exposure_entry = ttk.Entry(exposure_entry_row, textvariable=self.exposure_entry_var)
        self.exposure_entry.grid(row=0, column=0, sticky="ew")
        self.exposure_entry.bind("<Return>", self._apply_exposure_from_event)
        self.exposure_entry.bind("<FocusOut>", self._apply_exposure_from_event)
        self.exposure_entry.bind("<KeyRelease>", self._on_exposure_entry_edit)

        ttk.Separator(right).grid(row=6, column=0, sticky="ew", padx=8, pady=8)
        ttk.Label(right, text="Preview Zoom").grid(row=7, column=0, sticky="w", padx=8)
        zoom_row = ttk.Frame(right)
        zoom_row.grid(row=8, column=0, padx=8, pady=(2, 8), sticky="ew")
        zoom_row.columnconfigure(1, weight=1)
        ttk.Button(zoom_row, text="-", width=3, command=self._zoom_out).grid(row=0, column=0)
        self.zoom_scale = ttk.Scale(
            zoom_row,
            from_=0.25,
            to=4.0,
            variable=self.zoom_var,
            orient=tk.HORIZONTAL,
            length=170,
            command=self._on_zoom_slide,
        )
        self.zoom_scale.grid(row=0, column=1, padx=6, sticky="ew")
        ttk.Button(zoom_row, text="+", width=3, command=self._zoom_in).grid(row=0, column=2)
        self.zoom_entry = ttk.Entry(zoom_row, textvariable=self.zoom_entry_var, width=7)
        self.zoom_entry.grid(row=0, column=3, padx=(6, 0))
        self.zoom_entry.bind("<Return>", self._apply_zoom_from_event)
        ttk.Button(zoom_row, text="Set", command=self._apply_zoom_from_entry).grid(row=0, column=4, padx=(6, 0))

        ttk.Label(right, text="Preview Rotation").grid(row=9, column=0, sticky="w", padx=8)
        rot_row = ttk.Frame(right)
        rot_row.grid(row=10, column=0, padx=8, pady=(2, 8), sticky="ew")
        rot_row.columnconfigure(1, weight=1)
        ttk.Button(rot_row, text="Left", width=6, command=self._rotate_left).grid(row=0, column=0)
        self.rotation_combo = ttk.Combobox(
            rot_row,
            textvariable=self.rotation_var,
            state="readonly",
            values=("0", "90", "180", "270"),
            width=8,
        )
        self.rotation_combo.grid(row=0, column=1, padx=6, sticky="w")
        self.rotation_combo.bind("<<ComboboxSelected>>", self._on_rotation_change)
        ttk.Button(rot_row, text="Right", width=6, command=self._rotate_right).grid(row=0, column=2)

        ttk.Separator(right).grid(row=11, column=0, sticky="ew", padx=8, pady=8)
        ttk.Label(right, text="Camera").grid(row=12, column=0, sticky="w", padx=8)
        ttk.Label(right, textvariable=self.info_var, wraplength=280, justify="left").grid(row=13, column=0, sticky="w", padx=8, pady=(2, 8))

        ttk.Label(right, text="Frame").grid(row=14, column=0, sticky="w", padx=8)
        ttk.Label(right, textvariable=self.frame_var, wraplength=280, justify="left").grid(row=15, column=0, sticky="w", padx=8, pady=(2, 8))

        ttk.Separator(right).grid(row=16, column=0, sticky="ew", padx=8, pady=8)
        ttk.Label(right, text="Status").grid(row=17, column=0, sticky="w", padx=8)
        ttk.Label(right, textvariable=self.status_var, wraplength=280, justify="left").grid(row=18, column=0, sticky="w", padx=8, pady=(2, 8))

        ttk.Separator(right).grid(row=19, column=0, sticky="ew", padx=8, pady=8)
        ttk.Label(right, text="Scientific Capture").grid(row=20, column=0, sticky="w", padx=8)
        ttk.Label(right, text="Profile").grid(row=21, column=0, sticky="w", padx=8)
        self.profile_combo = ttk.Combobox(
            right,
            textvariable=self.capture_profile_var,
            values=self.profile_names,
            state="readonly",
            width=28,
        )
        self.profile_combo.grid(row=22, column=0, sticky="ew", padx=8, pady=(2, 4))
        self.profile_combo.bind("<<ComboboxSelected>>", self._on_profile_change)

        sf_row = ttk.Frame(right)
        sf_row.grid(row=23, column=0, sticky="ew", padx=8, pady=(2, 6))
        sf_row.columnconfigure(1, weight=1)
        ttk.Label(sf_row, text="Frames").grid(row=0, column=0, sticky="w")
        ttk.Entry(sf_row, textvariable=self.session_frames_var, width=10).grid(row=0, column=1, sticky="w", padx=(6, 0))

        ttk.Label(right, text="Session Output Dir").grid(row=24, column=0, sticky="w", padx=8)
        ttk.Entry(right, textvariable=self.session_dir_var, width=34).grid(row=25, column=0, sticky="ew", padx=8, pady=(2, 4))
        ttk.Label(right, text="Notes (optional)").grid(row=26, column=0, sticky="w", padx=8)
        ttk.Entry(right, textvariable=self.session_notes_var, width=34).grid(row=27, column=0, sticky="ew", padx=8, pady=(2, 6))

        cap_btn_row = ttk.Frame(right)
        cap_btn_row.grid(row=28, column=0, sticky="ew", padx=8, pady=(0, 8))
        ttk.Button(cap_btn_row, text="Capture Session", command=self._start_session_capture).grid(row=0, column=0, padx=(0, 6))
        ttk.Button(cap_btn_row, text="Stop", command=self._stop_session_capture).grid(row=0, column=1)

    def _on_right_canvas_configure(self, event: tk.Event) -> None:
        try:
            self.right_canvas.itemconfigure(self.right_canvas_window, width=event.width)
        except Exception:
            pass

    def _on_right_frame_configure(self, _event: tk.Event) -> None:
        try:
            self.right_canvas.configure(scrollregion=self.right_canvas.bbox("all"))
        except Exception:
            pass

    def _bind_right_mousewheel(self, _event: tk.Event) -> None:
        self.bind_all("<MouseWheel>", self._on_right_mousewheel)
        self.bind_all("<Button-4>", self._on_right_mousewheel)
        self.bind_all("<Button-5>", self._on_right_mousewheel)

    def _unbind_right_mousewheel(self, _event: tk.Event) -> None:
        self.unbind_all("<MouseWheel>")
        self.unbind_all("<Button-4>")
        self.unbind_all("<Button-5>")

    def _on_right_mousewheel(self, event: tk.Event) -> None:
        delta = getattr(event, "delta", 0)
        if delta:
            units = int(-delta / 120)
            if units == 0:
                units = -1 if delta > 0 else 1
            self.right_canvas.yview_scroll(units, "units")
            return
        num = getattr(event, "num", 0)
        if num == 4:
            self.right_canvas.yview_scroll(-1, "units")
        elif num == 5:
            self.right_canvas.yview_scroll(1, "units")

    def _on_profile_change(self, _event: tk.Event | None) -> None:
        if not SCIENTIFIC_CAPTURE_AVAILABLE or get_capture_profile is None:
            return
        name = self.capture_profile_var.get().strip()
        try:
            profile = get_capture_profile(name)
        except Exception:
            return
        self.session_frames_var.set(str(profile.frames_count))

    def _build_capture_profile(self, force_single: bool = False) -> CaptureProfile | None:
        if not SCIENTIFIC_CAPTURE_AVAILABLE or get_capture_profile is None or with_overrides is None:
            self.status_var.set("Scientific capture modules are unavailable")
            return None
        name = self.capture_profile_var.get().strip()
        try:
            base = get_capture_profile(name)
        except Exception as exc:
            self.status_var.set(f"Unknown profile: {name} ({exc})")
            return None
        exposure = self._parse_exposure_input()
        gain = self._parse_gain_input()
        if exposure is None or gain is None:
            return None
        try:
            frames_count = int(self.session_frames_var.get().strip())
        except ValueError:
            self.status_var.set(f"Invalid frames count: {self.session_frames_var.get().strip()}")
            return None
        if force_single:
            frames_count = 1
        profile = with_overrides(
            base,
            exposure_us=float(exposure),
            gain_db=float(gain),
            frames_count=frames_count,
        )
        return profile

    def _camera_metadata_for_session(self) -> dict[str, object]:
        meta: dict[str, object] = dict(self.camera_info)
        meta["interface"] = self.interface_var.get().strip()
        meta["camera_ip"] = self.camera_var.get().strip()
        if self.last_frame is not None:
            meta["width"] = self.last_frame.width
            meta["height"] = self.last_frame.height
            meta["pixel_format_int"] = self.last_frame.pixel_format
        if self.last_camera_config is not None:
            meta["configured_runtime"] = self.last_camera_config.get("runtime")
        return meta

    def _queue_profile_config(self, profile: CaptureProfile) -> None:
        if not (self.worker and self.worker.is_alive()):
            return
        if not SCIENTIFIC_CAPTURE_AVAILABLE or profile_to_dict is None:
            return
        try:
            self.cmd_q.put_nowait(("configure_profile", profile_to_dict(profile)))
        except queue.Full:
            self.status_var.set("Command queue is full")

    def _snapshot_scientific(self) -> None:
        self._start_session_capture(force_single=True)

    def _start_session_capture(self, force_single: bool = False) -> None:
        if not SCIENTIFIC_CAPTURE_AVAILABLE or SessionWriter is None or decode_buffer_to_ndarray is None:
            self.status_var.set("Scientific capture modules are unavailable")
            return
        if not (self.worker and self.worker.is_alive()):
            self.status_var.set("Connect camera first")
            return
        if self.active_session_writer is not None:
            self.status_var.set("Capture session is already running")
            return

        profile = self._build_capture_profile(force_single=force_single)
        if profile is None:
            return

        session_dir_raw = self.session_dir_var.get().strip()
        session_base = Path(session_dir_raw).expanduser() if session_dir_raw else self.snapshot_dir
        try:
            writer = SessionWriter(
                base_dir=session_base,
                profile=profile,
                camera_metadata=self._camera_metadata_for_session(),
                notes=self.session_notes_var.get().strip(),
            )
        except Exception as exc:
            self.status_var.set(f"Failed to create session directory: {exc}")
            return

        self.active_session_writer = writer
        self.active_session_profile = profile
        self.session_next_frame_index = 1
        self.session_remaining_frames = int(profile.frames_count)
        if self.session_remaining_frames == 0:
            self.session_remaining_frames = -1  # Capture until stopped.
        self._queue_profile_config(profile)

        if self.session_remaining_frames < 0:
            self.status_var.set(f"Session started: {writer.root_dir} (capture until stopped)")
        else:
            self.status_var.set(
                f"Session started: {writer.root_dir} ({self.session_remaining_frames} frame(s), profile={profile.name})"
            )

    def _stop_session_capture(self) -> None:
        self._finalize_active_session("Capture stopped")

    def _finalize_active_session(self, final_status: str) -> None:
        writer = self.active_session_writer
        if writer is not None:
            try:
                writer.finalize()
                self.status_var.set(f"{final_status}: {writer.root_dir}")
            except Exception as exc:
                self.status_var.set(f"{final_status}, finalize warning: {exc}")
        self.active_session_writer = None
        self.active_session_profile = None
        self.session_remaining_frames = 0
        self.session_next_frame_index = 1

    def _save_frame_to_active_session(self, frame: FramePacket) -> None:
        writer = self.active_session_writer
        profile = self.active_session_profile
        if writer is None or profile is None or decode_buffer_to_ndarray is None:
            return

        meta = dict(frame.meta or {})
        meta.setdefault("width", frame.width)
        meta.setdefault("height", frame.height)
        meta.setdefault("pixel_format_int", frame.pixel_format)
        if "pixel_format_name" not in meta:
            if self.camera_info.get("pixel_format"):
                meta["pixel_format_name"] = str(self.camera_info.get("pixel_format"))
            elif pixel_format_to_name is not None:
                meta["pixel_format_name"] = pixel_format_to_name(frame.pixel_format)
        meta.setdefault("pixel_format", meta.get("pixel_format_name", meta.get("pixel_format_int")))
        meta.setdefault("host_timestamp", frame.timestamp)
        meta.setdefault("exposure_us", float(self.exposure_var.get()))
        meta.setdefault("gain_db", float(self.gain_var.get()))

        frame_idx = self.session_next_frame_index
        try:
            raw_array = decode_buffer_to_ndarray(
                frame.raw,
                frame.width,
                frame.height,
                meta.get("pixel_format", frame.pixel_format),
            )
            writer.write_frame(
                frame_index=frame_idx,
                raw_array=raw_array,
                raw_bytes=frame.raw,
                frame_metadata=meta,
                save_preview=bool(profile.save_preview),
            )
        except Exception as exc:
            writer.add_warning(f"frame_{frame_idx:06d}: save failed: {exc}")
            self._finalize_active_session(f"Capture stopped due to frame save error ({exc})")
            return

        self.session_next_frame_index += 1
        if self.session_remaining_frames > 0:
            self.session_remaining_frames -= 1
            self.status_var.set(f"Captured frame {frame_idx} ({self.session_remaining_frames} remaining)")
            if self.session_remaining_frames == 0:
                self._finalize_active_session("Capture completed")
        else:
            self.status_var.set(f"Captured frame {frame_idx} (running)")

    def _connect(self) -> None:
        if self.worker and self.worker.is_alive():
            self.status_var.set("Already connected")
            return
        interface = self.interface_var.get().strip()
        camera_ip = self.camera_var.get().strip()
        if not interface:
            self.status_var.set("Set camera interface first (e.g. en10), then Connect")
            return
        if not camera_ip:
            self.status_var.set("Set camera IP first or run Auto Find/Fix, then Connect")
            return
        self.event_q = queue.Queue(maxsize=64)
        self.cmd_q = queue.Queue(maxsize=64)
        self.worker = CameraWorker(
            interface=interface,
            camera_ip=camera_ip,
            event_q=self.event_q,
            cmd_q=self.cmd_q,
            packet_size=self.packet_size,
            packet_delay=self.packet_delay,
            buffers=12,
        )
        self.status_var.set(
            f"Connecting... packet={self.packet_size}, delay={self.packet_delay}, target preview={1.0/self.render_interval_s:.1f} fps"
        )
        self.worker.start()

    def _disconnect(self) -> None:
        if self.active_session_writer is not None:
            self._finalize_active_session("Capture stopped")
        if self.worker and self.worker.is_alive():
            self.worker.stop()
            self.status_var.set("Disconnecting...")
        else:
            self.status_var.set("Not connected")

    def _refresh_controls(self) -> None:
        if self.worker and self.worker.is_alive():
            try:
                self.cmd_q.put_nowait(("refresh_controls", None))
            except queue.Full:
                pass

    def _push_ui_event(self, kind: str, payload: object | None = None) -> None:
        try:
            self.event_q.put_nowait((kind, payload))
        except queue.Full:
            pass

    def _auto_find_fix(self) -> None:
        if self.auto_fix_running:
            self.status_var.set("Auto Find/Fix is already running")
            return
        if not AUTO_FIX_AVAILABLE:
            self.status_var.set("Auto Find/Fix unavailable: helper modules import failed")
            return
        interface = self.interface_var.get().strip()
        if not interface:
            self.status_var.set("Auto Find/Fix failed: empty interface")
            return
        self.auto_fix_running = True
        self.status_var.set("Auto Find/Fix started...")
        threading.Thread(target=self._auto_find_fix_worker, args=(interface,), daemon=True).start()

    def _auto_find_fix_worker(self, interface: str) -> None:
        def emit_status(msg: str) -> None:
            self._push_ui_event("auto_fix_status", msg)

        def finish(msg: str) -> None:
            self._push_ui_event("auto_fix_done", msg)

        try:
            host_ip = get_interface_ipv4(interface)
            if not host_ip:
                finish(f"Auto Find/Fix failed: no IPv4 on {interface}")
                return
            netmask = get_interface_netmask(interface) or "255.255.255.0"
            target_ip = suggest_camera_ip(host_ip)

            emit_status(f"Discovering camera on {interface}...")
            cams = gvcp_discover(interface, duration=3.5, interval=0.25) if gvcp_discover else []
            if not cams:
                finish("No camera replies on discovery")
                return

            cam = cams[0]
            mac = (cam.mac or "").lower()
            discovered_ip = cam.current_ip or cam.source_ip
            emit_status(f"Found {cam.model_name} at {discovered_ip}")

            need_force = (not same_subnet(discovered_ip, host_ip, netmask)) or (discovered_ip != target_ip)
            if need_force and mac and gvcp_force_ip:
                emit_status(f"Applying ForceIP {target_ip}...")
                try:
                    gvcp_force_ip(
                        interface=interface,
                        target_mac=mac,
                        ip=target_ip,
                        subnet=netmask,
                        gateway="0.0.0.0",
                        timeout=1.2,
                    )
                    time.sleep(0.3)
                except Exception as exc:
                    emit_status(f"ForceIP error: {exc}")

                cams_after = gvcp_discover(interface, duration=3.5, interval=0.25) if gvcp_discover else []
                if cams_after:
                    selected = None
                    for c in cams_after:
                        if (c.mac or "").lower() == mac:
                            selected = c
                            break
                    if selected is None:
                        selected = cams_after[0]
                    discovered_ip = selected.current_ip or selected.source_ip
                    emit_status(f"Camera now at {discovered_ip}")
                else:
                    emit_status("Camera did not reply after ForceIP")

            self._push_ui_event("auto_fix_set_ip", discovered_ip)
            finish(f"Auto Find/Fix done, camera IP: {discovered_ip}")
        except Exception as exc:
            finish(f"Auto Find/Fix failed: {exc}")

    def _on_gain_slide(self, _value: str) -> None:
        self.gain_entry_var.set(f"{float(self.gain_var.get()):.3f}")
        self._schedule_gain_apply()

    def _on_exposure_slide(self, _value: str) -> None:
        self.exposure_entry_var.set(f"{float(self.exposure_var.get()):.1f}")
        self._schedule_exposure_apply()

    def _on_gain_entry_edit(self, _event: tk.Event) -> None:
        self._schedule_gain_apply()

    def _on_exposure_entry_edit(self, _event: tk.Event) -> None:
        self._schedule_exposure_apply()

    def _schedule_gain_apply(self) -> None:
        if self._gain_apply_after_id is not None:
            try:
                self.after_cancel(self._gain_apply_after_id)
            except Exception:
                pass
        self._gain_apply_after_id = self.after(self.auto_apply_delay_ms, self._auto_apply_gain)

    def _schedule_exposure_apply(self) -> None:
        if self._exposure_apply_after_id is not None:
            try:
                self.after_cancel(self._exposure_apply_after_id)
            except Exception:
                pass
        self._exposure_apply_after_id = self.after(self.auto_apply_delay_ms, self._auto_apply_exposure)

    def _auto_apply_gain(self) -> None:
        self._gain_apply_after_id = None
        self._apply_gain(silent_if_disconnected=True)

    def _auto_apply_exposure(self) -> None:
        self._exposure_apply_after_id = None
        self._apply_exposure(silent_if_disconnected=True)

    def _rerender_latest(self) -> None:
        if self.last_frame is not None:
            self._render_frame(self.last_frame)
            self.last_render_ts = time.monotonic()

    def _on_zoom_slide(self, _value: str) -> None:
        z = max(0.25, min(4.0, float(self.zoom_var.get())))
        self.zoom_var.set(z)
        self.zoom_entry_var.set(f"{z:.2f}")
        self._rerender_latest()

    def _parse_zoom_input(self) -> float | None:
        raw = self.zoom_entry_var.get().strip()
        if not raw:
            return float(self.zoom_var.get())
        try:
            z = float(raw)
        except ValueError:
            self.status_var.set(f"Invalid Zoom value: {raw}")
            return None
        z = max(0.25, min(4.0, z))
        self.zoom_var.set(z)
        self.zoom_entry_var.set(f"{z:.2f}")
        return z

    def _apply_zoom_from_entry(self) -> None:
        z = self._parse_zoom_input()
        if z is None:
            return
        self.zoom_var.set(z)
        self._rerender_latest()

    def _apply_zoom_from_event(self, _event: tk.Event) -> None:
        self._apply_zoom_from_entry()

    def _zoom_in(self) -> None:
        z = min(4.0, float(self.zoom_var.get()) + 0.1)
        self.zoom_var.set(z)
        self.zoom_entry_var.set(f"{z:.2f}")
        self._rerender_latest()

    def _zoom_out(self) -> None:
        z = max(0.25, float(self.zoom_var.get()) - 0.1)
        self.zoom_var.set(z)
        self.zoom_entry_var.set(f"{z:.2f}")
        self._rerender_latest()

    def _on_mouse_wheel(self, event: tk.Event) -> None:
        delta = getattr(event, "delta", 0)
        if delta == 0:
            return
        step = 0.1 if delta > 0 else -0.1
        z = max(0.25, min(4.0, float(self.zoom_var.get()) + step))
        self.zoom_var.set(z)
        self.zoom_entry_var.set(f"{z:.2f}")
        self._rerender_latest()

    def _get_rotation_deg(self) -> int:
        try:
            deg = int(self.rotation_var.get().strip())
        except ValueError:
            deg = 0
        if deg not in (0, 90, 180, 270):
            deg = 0
        return deg

    def _set_rotation_deg(self, deg: int) -> None:
        valid = (0, 90, 180, 270)
        d = deg % 360
        if d not in valid:
            d = 0
        self.rotation_var.set(str(d))
        self._rerender_latest()

    def _rotate_left(self) -> None:
        self._set_rotation_deg(self._get_rotation_deg() - 90)

    def _rotate_right(self) -> None:
        self._set_rotation_deg(self._get_rotation_deg() + 90)

    def _on_rotation_change(self, _event: tk.Event) -> None:
        self._set_rotation_deg(self._get_rotation_deg())

    def _parse_gain_input(self) -> float | None:
        raw = self.gain_entry_var.get().strip()
        if not raw:
            return float(self.gain_var.get())
        try:
            value = float(raw)
        except ValueError:
            self.status_var.set(f"Invalid Gain value: {raw}")
            return None
        low, high = self.gain_bounds
        value = max(low, min(high, value))
        self.gain_var.set(value)
        self.gain_entry_var.set(f"{value:.3f}")
        return value

    def _parse_exposure_input(self) -> float | None:
        raw = self.exposure_entry_var.get().strip()
        if not raw:
            return float(self.exposure_var.get())
        try:
            value = float(raw)
        except ValueError:
            self.status_var.set(f"Invalid Exposure value: {raw}")
            return None
        low, high = self.exposure_bounds
        value = max(low, min(high, value))
        self.exposure_var.set(value)
        self.exposure_entry_var.set(f"{value:.1f}")
        return value

    def _apply_gain_from_event(self, _event: tk.Event) -> None:
        self._apply_gain()

    def _apply_exposure_from_event(self, _event: tk.Event) -> None:
        self._apply_exposure()

    def _apply_gain(self, silent_if_disconnected: bool = False) -> None:
        if self._gain_apply_after_id is not None:
            try:
                self.after_cancel(self._gain_apply_after_id)
            except Exception:
                pass
            self._gain_apply_after_id = None
        if not (self.worker and self.worker.is_alive()):
            if not silent_if_disconnected:
                self.status_var.set("Connect camera first")
            return
        value = self._parse_gain_input()
        if value is None:
            return
        try:
            self.cmd_q.put_nowait(("set_gain", value))
        except queue.Full:
            self.status_var.set("Command queue is full")

    def _apply_exposure(self, silent_if_disconnected: bool = False) -> None:
        if self._exposure_apply_after_id is not None:
            try:
                self.after_cancel(self._exposure_apply_after_id)
            except Exception:
                pass
            self._exposure_apply_after_id = None
        if not (self.worker and self.worker.is_alive()):
            if not silent_if_disconnected:
                self.status_var.set("Connect camera first")
            return
        value = self._parse_exposure_input()
        if value is None:
            return
        try:
            self.cmd_q.put_nowait(("set_exposure", value))
        except queue.Full:
            self.status_var.set("Command queue is full")

    def _snapshot(self) -> None:
        # Kept for backwards compatibility of method name in old bindings.
        # Snapshot action now routes to scientific RAW session branch.
        self._snapshot_scientific()

    def _poll_events(self) -> None:
        try:
            for _ in range(self.max_events_per_poll):
                kind, payload = self.event_q.get_nowait()
                if kind == "connected":
                    info = payload if isinstance(payload, dict) else {}
                    self.status_var.set("Connected")
                    self.camera_info = dict(info)
                    vendor = info.get("vendor", "")
                    model = info.get("model", "")
                    serial = info.get("serial", "")
                    pix = info.get("pixel_format", "")
                    payload_size = info.get("payload", 0)
                    serial_line = f"\nSerial: {serial}" if serial else ""
                    self.info_var.set(f"{vendor} {model}{serial_line}\nPixelFormat: {pix}\nPayload: {payload_size} B")
                    controls = info.get("controls")
                    if isinstance(controls, dict):
                        self._apply_controls_dict(controls)
                    runtime = info.get("runtime")
                    if isinstance(runtime, dict):
                        self.camera_info["runtime"] = runtime
                elif kind == "controls":
                    if isinstance(payload, dict):
                        self._apply_controls_dict(payload)
                elif kind == "frame":
                    if isinstance(payload, FramePacket):
                        self.last_frame = payload
                        now = time.monotonic()
                        self._rx_frames += 1
                        rx_dt = now - self._rx_fps_window_ts
                        if rx_dt >= 1.0:
                            self._rx_fps = self._rx_frames / rx_dt
                            self._rx_frames = 0
                            self._rx_fps_window_ts = now
                        if self.active_session_writer is not None:
                            self._save_frame_to_active_session(payload)
                        now = time.monotonic()
                        if now - self.last_render_ts >= self.render_interval_s:
                            self._render_frame(payload)
                            self.last_render_ts = now
                elif kind == "camera_config":
                    if isinstance(payload, dict):
                        self.last_camera_config = payload
                        if self.active_session_writer is not None:
                            self.active_session_writer.set_configuration_result(payload)
                elif kind == "status":
                    self.status_var.set(str(payload))
                elif kind == "auto_fix_status":
                    self.status_var.set(str(payload))
                elif kind == "auto_fix_set_ip":
                    if isinstance(payload, str) and payload:
                        self.camera_var.set(payload)
                        self.status_var.set(f"Camera IP set to {payload}")
                elif kind == "auto_fix_done":
                    self.auto_fix_running = False
                    self.status_var.set(str(payload))
                elif kind == "error":
                    self.status_var.set(f"Error: {payload}")
                elif kind == "disconnected":
                    self.status_var.set("Disconnected")
                    if self.active_session_writer is not None:
                        self._finalize_active_session("Capture stopped: camera disconnected")
        except queue.Empty:
            pass
        self.after(self.ui_poll_ms, self._poll_events)

    def _apply_controls_dict(self, controls: dict[str, float]) -> None:
        if "gain_min" in controls and "gain_max" in controls:
            self.gain_bounds = (float(controls["gain_min"]), float(controls["gain_max"]))
            self.gain_scale.configure(from_=self.gain_bounds[0], to=self.gain_bounds[1])
        if "gain" in controls:
            gain = float(controls["gain"])
            self.gain_var.set(gain)
            self.gain_entry_var.set(f"{gain:.3f}")
        if "exposure_min" in controls and "exposure_max" in controls:
            self.exposure_bounds = (float(controls["exposure_min"]), float(controls["exposure_max"]))
            self.exposure_scale.configure(from_=self.exposure_bounds[0], to=self.exposure_bounds[1])
        if "exposure" in controls:
            exposure = float(controls["exposure"])
            self.exposure_var.set(exposure)
            self.exposure_entry_var.set(f"{exposure:.1f}")

    def _on_canvas_resize(self, _event: tk.Event) -> None:
        cx = self.preview_canvas.winfo_width() // 2
        cy = self.preview_canvas.winfo_height() // 2
        self.preview_canvas.coords(self.canvas_image_id, cx, cy)
        self.preview_canvas.coords(self.canvas_text_id, cx, cy)

    def _render_frame(self, frame: FramePacket) -> None:
        if frame.width <= 0 or frame.height <= 0:
            return
        expected = frame.width * frame.height
        if len(frame.raw) < expected:
            self.frame_var.set(
                f"{frame.width}x{frame.height}, fmt=0x{frame.pixel_format:08x}, bytes={len(frame.raw)} (short)"
            )
            return
        rot = self._get_rotation_deg()
        zoom = max(0.25, min(4.0, float(self.zoom_var.get())))
        frame_meta = frame.meta or {}
        pixel_format_name = str(
            frame_meta.get("pixel_format_name")
            or self.camera_info.get("pixel_format")
            or (pixel_format_to_name(frame.pixel_format) if pixel_format_to_name is not None else "")
        )
        pf_upper = pixel_format_name.upper()

        use_scientific_preview = bool(
            FAST_PREVIEW_AVAILABLE
            and SCIENTIFIC_CAPTURE_AVAILABLE
            and decode_buffer_to_ndarray is not None
            and make_preview_from_raw is not None
            and ("12" in pf_upper or "10" in pf_upper or len(frame.raw) >= expected * 2)
        )

        if use_scientific_preview:
            try:
                raw_array = decode_buffer_to_ndarray(frame.raw, frame.width, frame.height, pixel_format_name or frame.pixel_format)
                rgb_preview = make_preview_from_raw(raw_array, pixel_format_name or frame.pixel_format, 960, 640)
                img = Image.fromarray(rgb_preview, mode="RGB")
                if rot:
                    img = _rotate_pil_for_deg(img, rot)
                if abs(zoom - 1.0) > 1e-6:
                    zw = max(1, int(round(img.width * zoom)))
                    zh = max(1, int(round(img.height * zoom)))
                    img = img.resize((zw, zh), resample=Image.Resampling.NEAREST)
                out_w = int(img.width)
                out_h = int(img.height)
                mode_label = "RGB" if "BAYER" in pf_upper else "Mono"
                photo = ImageTk.PhotoImage(img)
            except Exception as exc:
                self.status_var.set(f"Scientific preview decode error: {exc}")
                return
        elif FAST_PREVIEW_AVAILABLE:
            try:
                img, out_w, out_h, mode_label = build_preview_image_fast(
                    frame.raw[:expected], frame.width, frame.height, frame.pixel_format, rot, zoom, 960, 640
                )
                if img is None:
                    return
                photo = ImageTk.PhotoImage(img)
            except Exception as exc:
                self.status_var.set(f"Preview fast-path error: {exc}")
                return
        else:
            if is_bayer_rg8(frame.pixel_format):
                out_w, out_h, rgb = preview_bayer_rg8_fast(frame.raw[:expected], frame.width, frame.height, 960, 640)
                if out_w <= 0 or out_h <= 0:
                    return
                mode_label = "RGB"
            else:
                out_w, out_h, mono = downsample_mono(frame.raw[:expected], frame.width, frame.height, 960, 640)
                if out_w <= 0 or out_h <= 0:
                    return
                rgb = mono_to_rgb_bytes(mono)
                mode_label = "Mono"
            if rot:
                out_w, out_h, rgb = rotate_rgb(rgb, out_w, out_h, rot)
            if abs(zoom - 1.0) > 1e-6:
                out_w, out_h, rgb = resize_rgb_nearest(rgb, out_w, out_h, zoom)
            png = rgb_to_png_bytes(rgb, out_w, out_h, level=1)
            if not png:
                self.status_var.set("Preview render error: empty PNG")
                return
            b64 = base64.b64encode(png).decode("ascii")
            try:
                photo = tk.PhotoImage(data=b64, format="PNG")
            except tk.TclError as exc:
                self.status_var.set(f"Preview render error: {exc}")
                return

        self.preview_photo = photo
        self.preview_canvas.itemconfigure(self.canvas_image_id, image=photo, state="normal")
        self.preview_canvas.itemconfigure(self.canvas_text_id, state="hidden")
        self._on_canvas_resize(None)
        now = time.monotonic()
        self._render_frames += 1
        render_dt = now - self._render_fps_window_ts
        if render_dt >= 1.0:
            self._render_fps = self._render_frames / render_dt
            self._render_frames = 0
            self._render_fps_window_ts = now
        self.frame_var.set(
            f"{frame.width}x{frame.height} -> preview {out_w}x{out_h} ({mode_label})\n"
            f"pixel_format=0x{frame.pixel_format:08x}, bytes={len(frame.raw)}, rot={rot}, zoom={zoom:.2f}x\n"
            f"rx={self._rx_fps:.1f} fps, preview={self._render_fps:.1f} fps, target={1.0/self.render_interval_s:.1f} fps"
        )

    def _on_close(self) -> None:
        if self.active_session_writer is not None:
            self._finalize_active_session("Capture stopped")
        self._disconnect()
        self.after(150, self.destroy)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Baumer live GUI")
    parser.add_argument("--interface", default="", help="GigE interface name (e.g. en10)")
    parser.add_argument("--camera", default="", help="Camera IPv4")
    parser.add_argument("--snapshot-dir", default="capture", help="Directory for snapshots")
    parser.add_argument(
        "--packet-size",
        type=int,
        default=DEFAULT_PACKET_SIZE,
        help="GevSCPSPacketSize value (bytes), e.g. 1440 for MTU 1500",
    )
    parser.add_argument(
        "--packet-delay",
        type=int,
        default=DEFAULT_PACKET_DELAY,
        help="GevSCPD inter-packet delay (camera ticks/ns, model dependent)",
    )
    parser.add_argument(
        "--preview-fps",
        type=float,
        default=DEFAULT_PREVIEW_FPS,
        help="Target preview render FPS in UI",
    )
    parser.add_argument(
        "--ui-poll-ms",
        type=int,
        default=DEFAULT_UI_POLL_MS,
        help="UI event polling period in milliseconds",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    app = BaumerLiveApp(
        interface=args.interface,
        camera_ip=args.camera,
        snapshot_dir=Path(args.snapshot_dir).expanduser(),
        packet_size=args.packet_size,
        packet_delay=args.packet_delay,
        preview_fps=args.preview_fps,
        ui_poll_ms=args.ui_poll_ms,
    )
    app.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
