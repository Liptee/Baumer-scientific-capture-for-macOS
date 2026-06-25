#!/usr/bin/env python3
"""
Hydra calibration/capture GUI for Baumer GigE cameras (macOS).

Workflow:
1) Connect page (Interface/IP/Auto Find-Fix)
2) Choice page (calibrate or load existing calibration)
3) Calibration wizard:
   - Crop calibration (16 white points, global offsets, manual point drag)
   - Flat field (flat map burst)
   - Geometry (5 valid chessboard captures, homography solve)
4) Main capture page (grid/single-lens preview and snapshot export).
"""

from __future__ import annotations

import argparse
import datetime as dt
import fcntl
import json
import math
import os
import queue
import re
import select
import shutil
import socket
import struct
import subprocess
import sys
import threading
import time
import tkinter as tk
from dataclasses import dataclass
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
import zaber_motion
from zaber_motion import Units
from zaber_motion.ascii import Connection

SIOCGIFADDR = 0xC0206921
DEFAULT_PACKET_SIZE = 1440
DEFAULT_PACKET_DELAY = 1000
DEFAULT_UI_POLL_MS = 10
DEFAULT_PREVIEW_FPS = 10.0

# Avoid OpenCV OpenCL/Metal runtime crashes on macOS during chessboard detection.
os.environ.setdefault("OPENCV_OPENCL_RUNTIME", "disabled")
os.environ.setdefault("OPENCV_OPENCL_CACHE_ENABLE", "0")

try:
    import numpy as np
except Exception:  # pragma: no cover - runtime dependency
    np = None  # type: ignore[assignment]

try:
    import cv2
except Exception:  # pragma: no cover - runtime dependency
    cv2 = None  # type: ignore[assignment]
else:
    try:
        if hasattr(cv2, "ocl") and hasattr(cv2.ocl, "setUseOpenCL"):
            cv2.ocl.setUseOpenCL(False)
    except Exception:
        pass
    try:
        if hasattr(cv2, "setNumThreads"):
            cv2.setNumThreads(1)
    except Exception:
        pass

try:
    from PIL import Image, ImageDraw, ImageTk
except Exception:  # pragma: no cover - runtime dependency
    Image = None  # type: ignore[assignment]
    ImageDraw = None  # type: ignore[assignment]
    ImageTk = None  # type: ignore[assignment]

try:
    from scipy.ndimage import convolve, convolve1d, gaussian_filter
except Exception:  # pragma: no cover - runtime dependency
    convolve = None  # type: ignore[assignment]
    convolve1d = None  # type: ignore[assignment]
    gaussian_filter = None  # type: ignore[assignment]

from baumer_capture_one import configure_aravis_gige_interface, open_camera_with_fallback
from camera_control import read_buffer_metadata, read_camera_runtime_metadata
from raw_decode import decode_buffer_to_ndarray, pixel_format_to_name

try:
    from baumer_force_ip import send_force_ip as gvcp_force_ip
    from baumer_gvcp_explorer import discover as gvcp_discover

    AUTO_FIX_AVAILABLE = True
except Exception:
    gvcp_force_ip = None  # type: ignore[assignment]
    gvcp_discover = None  # type: ignore[assignment]
    AUTO_FIX_AVAILABLE = False


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


def _decode_ifconfig_netmask(token: str) -> str:
    if token.startswith("0x"):
        try:
            return socket.inet_ntoa(struct.pack(">I", int(token, 16)))
        except Exception:
            return "255.255.255.0"
    return token


def get_interface_ipv4_entries(interface: str) -> list[tuple[str, str]]:
    try:
        txt = subprocess.check_output(["ifconfig", interface], text=True, stderr=subprocess.DEVNULL)
    except Exception:
        return []
    entries: list[tuple[str, str]] = []
    for line in txt.splitlines():
        parts = line.strip().split()
        if len(parts) < 4 or parts[0] != "inet" or parts[2] != "netmask":
            continue
        ip = parts[1]
        if ip.startswith("127."):
            continue
        entries.append((ip, _decode_ifconfig_netmask(parts[3])))
    return entries


def is_ipv4_literal(value: str) -> bool:
    try:
        socket.inet_aton(value)
        return value.count(".") == 3
    except OSError:
        return False


def get_interface_ipv4_for_peer(interface: str, peer_ip: str) -> str | None:
    entries = get_interface_ipv4_entries(interface)
    if not entries:
        return get_interface_ipv4(interface)
    if is_ipv4_literal(peer_ip):
        for ip, mask in entries:
            if same_subnet(ip, peer_ip, mask):
                return ip
    return entries[0][0]


def get_interface_netmask(interface: str) -> str | None:
    try:
        txt = subprocess.check_output(["ifconfig", interface], text=True, stderr=subprocess.DEVNULL)
    except Exception:
        return None
    for line in txt.splitlines():
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


def get_arp_mac_for_ip(interface: str, ip: str) -> str | None:
    try:
        txt = subprocess.check_output(["arp", "-an", "-i", interface], text=True, stderr=subprocess.DEVNULL)
    except Exception:
        return None
    pat = re.compile(rf"\({re.escape(ip)}\)\s+at\s+([0-9a-f:]+)\s+on\s+{re.escape(interface)}", re.IGNORECASE)
    for line in txt.splitlines():
        m = pat.search(line)
        if not m:
            continue
        mac = m.group(1).strip().lower()
        parts = mac.split(":")
        if len(parts) != 6:
            continue
        if any(not p for p in parts):
            continue
        try:
            norm = ":".join(f"{int(p, 16):02x}" for p in parts)
        except Exception:
            continue
        return norm
    return None


def get_arp_entries(interface: str) -> list[tuple[str, str]]:
    try:
        txt = subprocess.check_output(["arp", "-an", "-i", interface], text=True, stderr=subprocess.DEVNULL)
    except Exception:
        return []
    entries: list[tuple[str, str]] = []
    pat = re.compile(r"\(([^)]+)\)\s+at\s+([0-9a-f:]+)\s+on\s+" + re.escape(interface), re.IGNORECASE)
    for line in txt.splitlines():
        lo = line.lower()
        if "(incomplete)" in lo:
            continue
        m = pat.search(line)
        if not m:
            continue
        ip = m.group(1).strip()
        mac = m.group(2).strip().lower()
        parts = mac.split(":")
        if len(parts) != 6:
            continue
        try:
            norm = ":".join(f"{int(p, 16):02x}" for p in parts)
        except Exception:
            continue
        entries.append((ip, norm))
    return entries


def ipv4_to_u32(value: str) -> int:
    return int.from_bytes(socket.inet_aton(value), "big", signed=False)


def ip_to_int(ip: str) -> int:
    return int.from_bytes(socket.inet_aton(ip), "big", signed=False)


def same_subnet(ip_a: str, ip_b: str, mask: str) -> bool:
    try:
        ma = ip_to_int(mask)
        return (ip_to_int(ip_a) & ma) == (ip_to_int(ip_b) & ma)
    except Exception:
        return False


def suggest_camera_ip(host_ip: str, avoid: set[str] | None = None) -> str:
    avoid_set = set(avoid or ())
    parts = host_ip.split(".")
    if len(parts) != 4:
        return "192.168.88.1"
    try:
        a, b, c, _d = [int(p) for p in parts]
    except ValueError:
        return "192.168.88.1"
    prefix = f"{a}.{b}.{c}"
    preferred = [1, 2, 3, 4, 5, 10, 20, 50, 100, 200]
    for last in preferred:
        cand = f"{prefix}.{last}"
        if cand not in avoid_set:
            return cand
    for last in range(2, 255):
        cand = f"{prefix}.{last}"
        if cand not in avoid_set:
            return cand
    return f"{prefix}.1"


def masks_cfa_bayer(shape: tuple[int, int], pattern: str = "RGGB") -> tuple["np.ndarray", "np.ndarray", "np.ndarray"]:
    if np is None:
        raise RuntimeError("NumPy is unavailable")
    h, w = shape
    y, x = np.indices((h, w))
    y_even = (y % 2) == 0
    x_even = (x % 2) == 0
    p = pattern.upper()
    if p == "RGGB":
        r = y_even & x_even
        b = (~y_even) & (~x_even)
    elif p == "BGGR":
        r = (~y_even) & (~x_even)
        b = y_even & x_even
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
        raise RuntimeError("SciPy convolve1d is unavailable")
    return convolve1d(x, kernel, mode="mirror")


def _cnv_v(x: "np.ndarray", kernel: "np.ndarray") -> "np.ndarray":
    if convolve1d is None:
        raise RuntimeError("SciPy convolve1d is unavailable")
    return convolve1d(x, kernel, mode="mirror", axis=0)


def demosaic_bayer_menon2007(cfa: "np.ndarray", pattern: str = "RGGB") -> "np.ndarray":
    if np is None or convolve is None or convolve1d is None:
        raise RuntimeError("NumPy/SciPy demosaic dependencies are unavailable")

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
    m = d_v >= d_h
    g = np.where(m, g_h, g_v)
    mf = np.where(m, 1.0, 0.0)

    r_rows = np.transpose(np.any(r_m == 1, axis=1)[None]) * np.ones(r.shape, dtype=np.float32)
    b_rows = np.transpose(np.any(b_m == 1, axis=1)[None]) * np.ones(b.shape, dtype=np.float32)
    k_b = np.asarray([0.5, 0.0, 0.5], dtype=np.float32)

    r = np.where(np.logical_and(g_m == 1, r_rows == 1), g + _cnv_h(r, k_b) - _cnv_h(g, k_b), r)
    r = np.where(np.logical_and(g_m == 1, b_rows == 1), g + _cnv_v(r, k_b) - _cnv_v(g, k_b), r)
    b = np.where(np.logical_and(g_m == 1, b_rows == 1), g + _cnv_h(b, k_b) - _cnv_h(g, k_b), b)
    b = np.where(np.logical_and(g_m == 1, r_rows == 1), g + _cnv_v(b, k_b) - _cnv_v(g, k_b), b)

    r = np.where(
        np.logical_and(b_rows == 1, b_m == 1),
        np.where(mf == 1, b + _cnv_h(r, k_b) - _cnv_h(b, k_b), b + _cnv_v(r, k_b) - _cnv_v(b, k_b)),
        r,
    )
    b = np.where(
        np.logical_and(r_rows == 1, r_m == 1),
        np.where(mf == 1, r + _cnv_h(b, k_b) - _cnv_h(r, k_b), r + _cnv_v(b, k_b) - _cnv_v(r, k_b)),
        b,
    )
    return np.stack([r, g, b], axis=-1)


def demosaic_bayer_fast_preview(cfa: "np.ndarray", pattern: str = "RGGB") -> "np.ndarray":
    if np is None:
        raise RuntimeError("NumPy is unavailable")
    cfa = np.asarray(cfa, dtype=np.float32)
    r_m, g_m, b_m = masks_cfa_bayer(cfa.shape, pattern)
    r_mask = r_m > 0.5
    g_mask = g_m > 0.5
    b_mask = b_m > 0.5

    p = np.pad(cfa, ((1, 1), (1, 1)), mode="edge")
    n = p[:-2, 1:-1]
    s = p[2:, 1:-1]
    w = p[1:-1, :-2]
    e = p[1:-1, 2:]
    nw = p[:-2, :-2]
    ne = p[:-2, 2:]
    sw = p[2:, :-2]
    se = p[2:, 2:]

    g = cfa.copy()
    g_est = (n + s + w + e) * 0.25
    g[~g_mask] = g_est[~g_mask]

    r = np.zeros_like(cfa, dtype=np.float32)
    b = np.zeros_like(cfa, dtype=np.float32)
    r[r_mask] = cfa[r_mask]
    b[b_mask] = cfa[b_mask]

    r_rows = np.any(r_mask, axis=1)[:, None]
    b_rows = np.any(b_mask, axis=1)[:, None]
    g_on_r_rows = np.logical_and(g_mask, r_rows)
    g_on_b_rows = np.logical_and(g_mask, b_rows)
    diag = (nw + ne + sw + se) * 0.25

    r[g_on_r_rows] = ((w + e) * 0.5)[g_on_r_rows]
    r[g_on_b_rows] = ((n + s) * 0.5)[g_on_b_rows]
    r[b_mask] = diag[b_mask]

    b[g_on_b_rows] = ((w + e) * 0.5)[g_on_b_rows]
    b[g_on_r_rows] = ((n + s) * 0.5)[g_on_r_rows]
    b[r_mask] = diag[r_mask]

    return np.stack([r, g, b], axis=-1)


def raw_to_u8(raw_array: "np.ndarray", fmt_name: str, autostretch: bool = False) -> "np.ndarray":
    if np is None:
        raise RuntimeError("NumPy is unavailable")
    if autostretch:
        arr = raw_array.astype(np.float32, copy=False)
        lo = float(np.percentile(arr, 1.0))
        hi = float(np.percentile(arr, 99.5))
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            hi = lo + 1.0
        scaled = np.clip((arr - lo) * (255.0 / (hi - lo)), 0.0, 255.0)
        return scaled.astype(np.uint8)
    if raw_array.dtype == np.uint8:
        return raw_array
    name = fmt_name.upper()
    # Explicit 8-bit formats must preserve values even if array was promoted
    # to uint16/float earlier in the pipeline.
    if name in {"MONO8", "BAYERRG8", "BAYERGB8"}:
        return np.clip(raw_array, 0, 255).astype(np.uint8)
    if "12" in name:
        return np.clip(raw_array.astype(np.uint16) >> 4, 0, 255).astype(np.uint8)
    if "10" in name:
        return np.clip(raw_array.astype(np.uint16) >> 2, 0, 255).astype(np.uint8)
    max_v = int(np.max(raw_array)) if raw_array.size else 0
    if max_v <= 1023:
        return np.clip(raw_array.astype(np.uint16) >> 2, 0, 255).astype(np.uint8)
    if max_v <= 4095:
        return np.clip(raw_array.astype(np.uint16) >> 4, 0, 255).astype(np.uint8)
    return np.clip(raw_array.astype(np.uint16) >> 8, 0, 255).astype(np.uint8)


def bayer_pattern_from_fmt(fmt_name: str) -> str | None:
    name = str(fmt_name).upper()
    if "BAYER" not in name:
        return None
    if "RG" in name:
        return "RGGB"
    if "GB" in name:
        return "GBRG"
    if "GR" in name:
        return "GRBG"
    if "BG" in name:
        return "BGGR"
    return "RGGB"


def shift_bayer_pattern(pattern: str, x_off: int, y_off: int) -> str:
    p = pattern.upper()
    if len(p) != 4:
        return p
    grid = [[p[0], p[1]], [p[2], p[3]]]
    xo = int(x_off) & 1
    yo = int(y_off) & 1
    out = [
        [grid[(yo + 0) & 1][(xo + 0) & 1], grid[(yo + 0) & 1][(xo + 1) & 1]],
        [grid[(yo + 1) & 1][(xo + 0) & 1], grid[(yo + 1) & 1][(xo + 1) & 1]],
    ]
    return f"{out[0][0]}{out[0][1]}{out[1][0]}{out[1][1]}"


def bayer_pattern_for_crop(fmt_name: str, x_off: int, y_off: int) -> str | None:
    base = bayer_pattern_from_fmt(fmt_name)
    if base is None:
        return None
    return shift_bayer_pattern(base, x_off, y_off)


def debayer_menon_rgb(
    raw_array: "np.ndarray",
    fmt_name: str,
    pattern_override: str | None = None,
    autostretch: bool = False,
) -> "np.ndarray":
    if np is None:
        raise RuntimeError("NumPy is unavailable")
    u8 = raw_to_u8(raw_array, fmt_name, autostretch=autostretch)
    pattern = (pattern_override or bayer_pattern_from_fmt(fmt_name) or "").upper()
    if not pattern:
        return np.repeat(u8[:, :, None], 3, axis=2)
    try:
        rgbf = demosaic_bayer_menon2007(u8, pattern)
        return np.clip(np.rint(rgbf), 0, 255).astype(np.uint8)
    except Exception:
        try:
            rgbf = demosaic_bayer_fast_preview(u8, pattern)
            return np.clip(np.rint(rgbf), 0, 255).astype(np.uint8)
        except Exception:
            pass
        return np.repeat(u8[:, :, None], 3, axis=2)


def debayer_fast_preview_rgb(
    raw_array: "np.ndarray",
    fmt_name: str,
    pattern_override: str | None = None,
    autostretch: bool = False,
) -> "np.ndarray":
    if np is None:
        raise RuntimeError("NumPy is unavailable")
    u8 = raw_to_u8(raw_array, fmt_name, autostretch=autostretch)
    pattern = (pattern_override or bayer_pattern_from_fmt(fmt_name) or "").upper()
    if not pattern:
        return np.repeat(u8[:, :, None], 3, axis=2)
    try:
        rgbf = demosaic_bayer_fast_preview(u8, pattern)
        return np.clip(np.rint(rgbf), 0, 255).astype(np.uint8)
    except Exception:
        return np.repeat(u8[:, :, None], 3, axis=2)


def compose_grid16_rgb(lenses_rgb: "np.ndarray", gap: int = 2) -> "np.ndarray":
    if np is None:
        raise RuntimeError("NumPy is unavailable")
    h, w = int(lenses_rgb.shape[1]), int(lenses_rgb.shape[2])
    out = np.zeros((4 * h + 3 * gap, 4 * w + 3 * gap, 3), dtype=np.uint8)
    for i in range(16):
        r = i // 4
        c = i % 4
        y0 = r * (h + gap)
        x0 = c * (w + gap)
        out[y0 : y0 + h, x0 : x0 + w] = lenses_rgb[i]
    return out


@dataclass
class FramePacket:
    width: int
    height: int
    pixel_format: int
    raw: bytes
    timestamp: float
    meta: dict[str, object] | None = None


class CameraWorker(threading.Thread):
    def __init__(
        self,
        interface: str,
        camera_ip: str,
        event_q: queue.Queue,
        cmd_q: queue.Queue,
        packet_size: int,
        packet_delay: int,
        buffers: int = 12,
        debug: bool = False,
    ) -> None:
        super().__init__(daemon=True)
        self.interface = interface
        self.camera_ip = camera_ip
        self.event_q = event_q
        self.cmd_q = cmd_q
        self.packet_size = packet_size
        self.packet_delay = packet_delay
        self.buffers = buffers
        self.debug = bool(debug)
        self.stop_event = threading.Event()
        self.stream_stall_timeout_s = 2.5
        self.stream_restart_cooldown_s = 1.2
        self._last_good_frame_ts = 0.0
        self._last_restart_ts = 0.0
        self._last_bad_status_log_ts = 0.0
        self._debug_frame_count = 0
        self._if_ip_for_stream: str | None = None
        self._stream_port: int | None = None
        self._pending_gain: float | None = None
        self._pending_exposure: float | None = None
        self._pending_refresh_controls = False
        self._last_control_apply_ts = 0.0
        self.control_apply_interval_s = 0.08
        self._exposure_us_est = 50000.0
        self._consecutive_bad_status = 0
        self._last_transport_tune_ts = 0.0
        base_ps = int(self.packet_size) if int(self.packet_size) > 0 else 1440
        base_pd = int(self.packet_delay) if int(self.packet_delay) > 0 else 1000
        self._transport_profiles: list[tuple[int, int]] = [
            (base_ps, base_pd),
            (1200, max(1600, base_pd)),
            (1000, max(2600, base_pd)),
            (900, max(4200, base_pd)),
        ]
        # Keep unique while preserving order.
        uniq_profiles: list[tuple[int, int]] = []
        seen_profiles: set[tuple[int, int]] = set()
        for p in self._transport_profiles:
            if p in seen_profiles:
                continue
            seen_profiles.add(p)
            uniq_profiles.append(p)
        self._transport_profiles = uniq_profiles
        self._transport_profile_idx = 0

    def stop(self) -> None:
        self.stop_event.set()

    def _emit(self, kind: str, payload: object | None = None) -> None:
        # Keep UI responsive, but avoid starving calibration bursts by dropping too aggressively.
        if kind == "frame" and self.event_q.qsize() > 24:
            return
        try:
            self.event_q.put_nowait((kind, payload))
        except queue.Full:
            pass
        if self.debug:
            if kind in ("status", "error", "connected", "disconnected"):
                print(f"[hydra-debug][worker] {kind}: {payload}", flush=True)

    def _apply_stream_destination(self, camera) -> None:
        if not self._if_ip_for_stream:
            return
        try:
            camera.set_integer("GevSCDA", ipv4_to_u32(self._if_ip_for_stream))
        except Exception as exc:
            self._emit("status", f"GevSCDA set failed: {exc}")
        if self._stream_port is not None:
            try:
                camera.set_integer("GevSCPHostPort", int(self._stream_port))
            except Exception as exc:
                self._emit("status", f"GevSCPHostPort set failed: {exc}")

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
            self._apply_stream_destination(camera)
        except Exception:
            pass
        try:
            camera.start_acquisition()
            self._last_good_frame_ts = time.monotonic()
            self._emit("status", "Acquisition restarted")
        except Exception as exc:
            self._emit("status", f"Acquisition restart failed: {exc}")

    def _apply_transport_profile(self, camera, packet_size: int, packet_delay: int) -> None:
        try:
            camera.gv_set_packet_size(int(packet_size))
        except Exception:
            pass
        try:
            camera.set_integer("GevSCPSPacketSize", int(packet_size))
        except Exception:
            pass
        try:
            camera.set_integer("GevSCPD", int(packet_delay))
        except Exception:
            pass
        self.packet_size = int(packet_size)
        self.packet_delay = int(packet_delay)
        self._emit("status", f"Transport tuned: packet_size={packet_size}, packet_delay={packet_delay}")

    def _maybe_tune_transport_on_errors(self, camera, now_mono: float) -> None:
        if not is_ipv4_literal(self.camera_ip):
            return
        if self._transport_profile_idx >= (len(self._transport_profiles) - 1):
            return
        if (now_mono - self._last_transport_tune_ts) < 2.0:
            return
        self._transport_profile_idx += 1
        ps, pd = self._transport_profiles[self._transport_profile_idx]
        self._last_transport_tune_ts = now_mono
        self._apply_transport_profile(camera, ps, pd)

    def _read_controls(self, camera) -> dict[str, float]:
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
            self._exposure_us_est = max(20.0, float(out["exposure"]))
        except Exception:
            pass
        return out

    def _apply_command(self, camera, cmd: str, value: object | None) -> None:
        if cmd == "set_gain" and value is not None:
            camera.set_gain(float(value))
            self._emit("status", f"Gain: {float(value):.3f}")
        elif cmd == "set_exposure" and value is not None:
            camera.set_exposure_time(float(value))
            self._exposure_us_est = max(20.0, float(value))
            self._emit("status", f"Exposure: {float(value):.1f} us")
        elif cmd == "refresh_controls":
            pass
        else:
            return
        self._emit("controls", self._read_controls(camera))

    def _drain_command_queue(self) -> None:
        while True:
            try:
                cmd, value = self.cmd_q.get_nowait()
            except queue.Empty:
                break
            cmd_s = str(cmd)
            if cmd_s == "set_gain" and value is not None:
                try:
                    self._pending_gain = float(value)
                except Exception:
                    pass
            elif cmd_s == "set_exposure" and value is not None:
                try:
                    self._pending_exposure = float(value)
                except Exception:
                    pass
            elif cmd_s == "refresh_controls":
                self._pending_refresh_controls = True

    def _apply_pending_controls(self, camera, now_mono: float) -> None:
        if (now_mono - self._last_control_apply_ts) < self.control_apply_interval_s:
            return
        applied = False
        if self._pending_exposure is not None:
            value = float(self._pending_exposure)
            self._pending_exposure = None
            try:
                self._apply_command(camera, "set_exposure", value)
            except Exception as exc:
                self._emit("status", f"Command error (set_exposure): {exc}")
            applied = True
        if self._pending_gain is not None:
            value = float(self._pending_gain)
            self._pending_gain = None
            try:
                self._apply_command(camera, "set_gain", value)
            except Exception as exc:
                self._emit("status", f"Command error (set_gain): {exc}")
            applied = True
        if self._pending_refresh_controls:
            self._pending_refresh_controls = False
            self._emit("controls", self._read_controls(camera))
            applied = True
        if applied:
            self._last_control_apply_ts = now_mono

    def run(self) -> None:
        camera = None
        try:
            import gi  # noqa: PLC0415

            gi.require_version("Aravis", "0.8")
            from gi.repository import Aravis  # type: ignore  # noqa: PLC0415

            configure_aravis_gige_interface(Aravis, self.interface)
            try:
                Aravis.update_device_list()
            except Exception:
                pass

            camera, open_note = open_camera_with_fallback(Aravis, self.camera_ip, self.interface)
            if camera is None:
                raise RuntimeError(f"Cannot open camera {self.camera_ip}: {open_note}")
            if open_note:
                self._emit("status", f"Connect note: {open_note}")

            camera.gv_set_stream_options(Aravis.GvStreamOption.PACKET_SOCKET_DISABLED)
            camera.gv_set_packet_size_adjustment(Aravis.GvPacketSizeAdjustment.NEVER)
            if self.packet_size > 0:
                try:
                    camera.gv_set_packet_size(int(self.packet_size))
                except Exception:
                    pass

            stream = camera.create_stream(None, None)
            if stream is None:
                raise RuntimeError("create_stream returned None")

            if hasattr(stream, "get_port"):
                try:
                    self._stream_port = int(stream.get_port())
                except Exception:
                    self._stream_port = None

            if_ip = get_interface_ipv4_for_peer(self.interface, self.camera_ip)
            self._if_ip_for_stream = if_ip
            if self.debug:
                print(
                    f"[hydra-debug][worker] stream bind candidate: if_ip={if_ip} "
                    f"camera_ip={self.camera_ip} port={self._stream_port}",
                    flush=True,
                )
            if if_ip:
                self._apply_stream_destination(camera)

            try:
                camera.set_string("TriggerMode", "Off")
            except Exception:
                pass
            try:
                camera.set_string("ExposureAuto", "Off")
            except Exception:
                pass
            try:
                camera.set_string("GainAuto", "Off")
            except Exception:
                pass
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

            runtime = read_camera_runtime_metadata(camera)
            controls = self._read_controls(camera)
            self._emit(
                "connected",
                {
                    "vendor": str(camera.get_vendor_name()),
                    "model": str(camera.get_model_name()),
                    "serial": str(camera.get_device_serial_number()),
                    "pixel_format": str(camera.get_pixel_format_as_string()),
                    "payload": payload,
                    "controls": controls,
                    "runtime": runtime,
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

            success_status = int(Aravis.BufferStatus.SUCCESS)
            while not self.stop_event.is_set():
                now_mono = time.monotonic()
                self._drain_command_queue()
                self._apply_pending_controls(camera, now_mono)

                pop_timeout_ms = int(max(200.0, min(1800.0, (self._exposure_us_est / 1000.0) * 2.5 + 120.0)))
                buffer = stream.timeout_pop_buffer(pop_timeout_ms)
                if buffer is None:
                    self._maybe_restart_acquisition(camera, "no buffers", time.monotonic())
                    continue
                try:
                    status = int(buffer.get_status())
                    if status == success_status:
                        raw = bytes(buffer.get_image_data())
                        w = int(buffer.get_image_width())
                        h = int(buffer.get_image_height())
                        pf = int(buffer.get_image_pixel_format())
                        meta = read_buffer_metadata(buffer)
                        meta["width"] = w
                        meta["height"] = h
                        meta["pixel_format_int"] = pf
                        if "pixel_format_name" not in meta:
                            try:
                                meta["pixel_format_name"] = str(camera.get_pixel_format_as_string())
                            except Exception:
                                pass
                        self._last_good_frame_ts = time.monotonic()
                        self._consecutive_bad_status = 0
                        self._debug_frame_count += 1
                        if self.debug and (self._debug_frame_count % 40) == 1:
                            print(
                                f"[hydra-debug][worker] frame_ok #{self._debug_frame_count}: "
                                f"{w}x{h} pf=0x{pf:08x} bytes={len(raw)} frame_id={meta.get('frame_id')}",
                                flush=True,
                            )
                        self._emit("frame", FramePacket(w, h, pf, raw, time.time(), meta))
                    else:
                        now_bad = time.monotonic()
                        self._consecutive_bad_status += 1
                        if (now_bad - self._last_bad_status_log_ts) > 1.2:
                            self._last_bad_status_log_ts = now_bad
                            self._emit("status", f"Bad frame status: {status}")
                            if self.debug:
                                print(
                                    f"[hydra-debug][worker] bad_status={status} "
                                    f"frame_id={getattr(buffer, 'get_frame_id', lambda: None)()}",
                                    flush=True,
                                )
                        # Recover from persistent bad buffers: tune transport first, then restart acquisition.
                        if self._consecutive_bad_status >= 8:
                            self._maybe_tune_transport_on_errors(camera, now_bad)
                            self._maybe_restart_acquisition(camera, f"bad status {status}", now_bad)
                except Exception as exc:
                    self._emit("status", f"Frame decode error: {exc}")
                finally:
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


@dataclass
class CropCalibrationData:
    image_width: int
    image_height: int
    centers_xy: list[list[float]]
    offsets: dict[str, int]
    boxes: list[dict[str, int]]
    order: str = "top_to_bottom_left_to_right"


class HydraWizardApp(tk.Tk):
    def __init__(
        self,
        interface: str,
        camera_ip: str,
        output_dir: Path,
        packet_size: int = DEFAULT_PACKET_SIZE,
        packet_delay: int = DEFAULT_PACKET_DELAY,
        preview_fps: float = DEFAULT_PREVIEW_FPS,
        ui_poll_ms: int = DEFAULT_UI_POLL_MS,
        debug: bool = False,
        force_u8_mode: bool = True,
    ) -> None:
        super().__init__()
        self.title("Hydra Baumer Capture")
        self.geometry("560x250")
        self.minsize(540, 220)

        if np is None or Image is None or ImageTk is None:
            raise RuntimeError("Required dependencies are missing: numpy and pillow")

        self.packet_size = int(packet_size)
        self.packet_delay = int(packet_delay)
        self.ui_poll_ms = max(5, int(ui_poll_ms))
        self.render_interval_s = 1.0 / max(1.0, float(preview_fps))
        self.debug = bool(debug)
        self.force_u8_mode = bool(force_u8_mode)

        self.output_dir = output_dir.expanduser()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.session_root: Path | None = None

        self.interface_var = tk.StringVar(value=(interface or "en10"))
        self.camera_var = tk.StringVar(value=(camera_ip or ""))
        self.camera_pick_var = tk.StringVar(value="")
        self.status_var = tk.StringVar(value="Idle")
        self.connection_info_var = tk.StringVar(value="-")
        self.wavelength_map_path_var = tk.StringVar(value="")
        self.camera_scan_running = False
        self.camera_scan_display_to_id: dict[str, str] = {}

        self.gain_var = tk.DoubleVar(value=0.0)
        self.exposure_var = tk.DoubleVar(value=12000.0)
        self.gain_entry_var = tk.StringVar(value="0.0")
        self.exposure_entry_var = tk.StringVar(value="12000.0")
        self.gain_bounds = (0.0, 24.0)
        self.exposure_bounds = (100.0, 500000.0)

        self.crop_left_var = tk.IntVar(value=120)
        self.crop_right_var = tk.IntVar(value=120)
        self.crop_top_var = tk.IntVar(value=120)
        self.crop_bottom_var = tk.IntVar(value=120)
        self.manual_points_var = tk.BooleanVar(value=False)
        self.crop_limits: dict[str, int] = {"left": 120, "right": 120, "top": 120, "bottom": 120}
        self.crop_centers: "np.ndarray | None" = None
        self.crop_boxes: list[dict[str, int]] = []
        self._drag_center_idx: int | None = None

        self.dark_burst_var = tk.IntVar(value=8)
        self.flat_burst_var = tk.IntVar(value=8)
        self.flat_sigma_var = tk.DoubleVar(value=1.5)
        self.geometry_target_var = tk.IntVar(value=5)
        self.geometry_cols_var = tk.IntVar(value=3)
        self.geometry_rows_var = tk.IntVar(value=3)
        self.geometry_capture_idx = 0
        self.geometry_progress_var = tk.DoubleVar(value=0.0)
        self.geometry_progress_text_var = tk.StringVar(value="")
        self.main_view_mode = tk.StringVar(value="grid")
        self.main_lens_var = tk.IntVar(value=1)
        self.face_seg_lens_var = tk.IntVar(value=1)
        self.face_seg_lens_combo_var = tk.StringVar(value="1")
        self.face_seg_status_var = tk.StringVar(value="Face segmentation: model is not loaded")
        self.analyze_button_var = tk.StringVar(value="Анализ")
        self.classify_button_var = tk.StringVar(value="Classification")
        self.white_point_button_var = tk.StringVar(value="Задать точку белого")
        self.white_point_info_var = tk.StringVar(value="White point: not set")
        self.max_fps_var = tk.StringVar(value="Max FPS: -")

        # Zaber motion (optional, independent from camera connection).
        self.zaber_port_var = tk.StringVar(value="")
        self.zaber_axis_var = tk.IntVar(value=1)
        self.zaber_step_mm_var = tk.DoubleVar(value=1.0)
        self.zaber_target_mm_var = tk.DoubleVar(value=0.0)
        self.zaber_pos_mm_var = tk.StringVar(value="-")
        self.zaber_status_var = tk.StringVar(value="Zaber: disconnected")
        self.zaber_panel_expanded = tk.BooleanVar(value=False)
        self._zaber_conn: Connection | None = None
        self._zaber_axis = None
        self._zaber_cmd_q: queue.Queue = queue.Queue(maxsize=64)
        self._zaber_stop_event = threading.Event()
        self._zaber_thread = threading.Thread(target=self._zaber_worker_loop, name="hydra-zaber", daemon=True)
        self._zaber_thread.start()
        self._zaber_last_pos_mm: float | None = None
        self._serial_running = False
        self._serial_positions_mm: list[float] = []
        self._serial_idx = 0
        self._serial_wait_deadline = 0.0
        self._serial_last_capture_count = 0

        self.worker: CameraWorker | None = None
        self.event_q: queue.Queue = queue.Queue(maxsize=96)
        self.cmd_q: queue.Queue = queue.Queue(maxsize=64)
        self.last_frame: FramePacket | None = None
        self.last_render_ts = 0.0
        self.connected = False
        self.auto_fix_running = False
        self.camera_info: dict[str, object] = {}

        self.current_page = "connect"
        self.calib_stage = ""
        self._calib_photo: tk.PhotoImage | None = None
        self._main_photo: tk.PhotoImage | None = None
        self._calib_display_map: tuple[float, float, float] | None = None
        self._main_display_map: tuple[float, float, float] | None = None
        self._main_rgb_shape: tuple[int, int] | None = None
        self._white_pick_active = False
        self._white_pick_start_xy: tuple[float, float] | None = None
        self._white_pick_end_xy: tuple[float, float] | None = None
        self.white_point_roi: tuple[int, int, int, int] | None = None
        self.white_point_lens: int | None = None
        self.white_point_ref: "np.ndarray | None" = None
        self.white_point_created_utc: str | None = None
        self._last_rendered_frame_id: int | None = None
        self._debug_frame_count = 0
        self._debug_last_render_log_ts = 0.0
        self._dark_preview_hint_ts = 0.0
        self._render_submit_seq = 0
        self._render_applied_seq = 0
        self._render_in_q: queue.Queue = queue.Queue(maxsize=1)
        self._render_out_q: queue.Queue = queue.Queue(maxsize=2)
        self._render_stop_event = threading.Event()
        self._render_thread = threading.Thread(target=self._render_worker_loop, name="hydra-render", daemon=True)
        self._render_thread.start()
        self._face_seg_model_lock = threading.Lock()
        self._face_seg_model = None
        self._face_seg_processor = None
        self._face_seg_torch = None
        self._face_seg_device = "cpu"
        self._face_seg_model_ready = False
        self._face_seg_model_loading = False
        self._face_seg_model_error: str | None = None
        self._face_seg_preload_started = False
        self._analysis_recon_checked = False
        self._analysis_recon_ready = False
        self._analysis_recon_error: str | None = None
        self._recon_worker_lock = threading.Lock()
        self._recon_worker_proc: "subprocess.Popen[str] | None" = None
        self._recon_worker_req_id = 0
        self._recon_worker_ready_info: dict[str, object] | None = None
        self._analysis_busy = False
        self._analysis_mode = False
        self._analysis_base_rgb: "np.ndarray | None" = None
        self._analysis_view_rgb: "np.ndarray | None" = None
        self._analysis_face_mask: "np.ndarray | None" = None
        self._analysis_hsi_hwc: "np.ndarray | None" = None
        self._analysis_wavelengths_nm: "np.ndarray | None" = None
        self._analysis_spectrum: "np.ndarray | None" = None
        self._analysis_pick_xy: tuple[int, int] | None = None
        self._analysis_fas_label: str | None = None
        self._analysis_fas_is_live: bool | None = None
        self._analysis_image_width = 0
        self._analysis_rotation_quarters = 0
        self.analysis_smooth_enabled_var = tk.BooleanVar(value=False)
        self.analysis_smooth_sigma_var = tk.DoubleVar(value=1.2)
        self.analysis_smooth_window_var = tk.IntVar(value=5)
        self.analysis_auto_y_var = tk.BooleanVar(value=False)
        self._fas_mock_next_live = False
        self._face_cls_model: dict[str, object] | None = None
        self._face_cls_model_ready = False
        self._face_cls_model_loading = False
        self._face_cls_model_error: str | None = None
        self._face_cls_preload_started = False

        self.frame_dedupe_id: int | None = None
        self.dark_capture_active = False
        self.flat_capture_active = False
        self.dark_frames: list["np.ndarray"] = []
        self.flat_frames: list["np.ndarray"] = []
        self.dark_map: "np.ndarray | None" = None
        self.noise_map: "np.ndarray | None" = None
        self.flat_raw_mean: "np.ndarray | None" = None
        self.flat_norm: "np.ndarray | None" = None
        self.locked_gain: float | None = None
        self.locked_exposure: float | None = None
        self.geometry_corners: list[dict[int, "np.ndarray"]] = []
        self.geometry_focus_scores: list["np.ndarray"] = []
        self.geometry_h: "np.ndarray | None" = None
        self.geometry_correction_enabled = True
        self.reference_lens = 0
        self.wavelength_mapping_entries: list[dict[str, object]] = []
        self.wavelength_mapping_source: str | None = None

        self._build_ui()
        self._update_max_fps_label()
        self._update_analyze_button_state()
        self._dbg(
            f"init: interface={self.interface_var.get()} camera={self.camera_var.get()} "
            f"preview_fps={1.0/self.render_interval_s:.2f} force_u8={self.force_u8_mode}"
        )
        self.protocol("WM_DELETE_WINDOW", self._on_close)
        self.after(self.ui_poll_ms, self._poll_events)

    # ----------------------------- UI -----------------------------
    def _build_ui(self) -> None:
        self.root = ttk.Frame(self, padding=8)
        self.root.pack(fill=tk.BOTH, expand=True)
        self.root.rowconfigure(0, weight=1)
        self.root.columnconfigure(0, weight=1)

        self.page_connect = ttk.Frame(self.root)
        self.page_choice = ttk.Frame(self.root)
        self.page_calib = ttk.Frame(self.root)
        self.page_main = ttk.Frame(self.root)
        for p in (self.page_connect, self.page_choice, self.page_calib, self.page_main):
            p.grid(row=0, column=0, sticky="nsew")

        self._build_connect_page()
        self._build_choice_page()
        self._build_calibration_page()
        self._build_main_page()
        self._show_page("connect")

    def _dbg(self, message: str) -> None:
        if not self.debug:
            return
        print(f"[hydra-debug] {message}", flush=True)

    def _build_connect_page(self) -> None:
        f = self.page_connect
        for c in range(4):
            f.columnconfigure(c, weight=1 if c == 1 else 0)
        ttk.Label(f, text="Interface").grid(row=0, column=0, sticky="w", pady=(4, 4))
        ttk.Entry(f, textvariable=self.interface_var, width=12).grid(row=0, column=1, sticky="ew", padx=(8, 8))
        ttk.Label(f, text="Camera ID / IP").grid(row=1, column=0, sticky="w", pady=(4, 4))
        ttk.Entry(f, textvariable=self.camera_var, width=42).grid(row=1, column=1, sticky="ew", padx=(8, 8))
        ttk.Label(f, text="Detected Cameras").grid(row=2, column=0, sticky="w", pady=(4, 4))
        self.camera_pick_combo = ttk.Combobox(
            f,
            textvariable=self.camera_pick_var,
            values=[],
            state="readonly",
            width=42,
        )
        self.camera_pick_combo.grid(row=2, column=1, sticky="ew", padx=(8, 8))
        self.camera_pick_combo.bind("<<ComboboxSelected>>", self._on_camera_pick_selected)

        btn_row = ttk.Frame(f)
        btn_row.grid(row=0, column=2, rowspan=3, sticky="ns", padx=4)
        ttk.Button(btn_row, text="Auto Find/Fix", command=self._auto_find_fix).grid(row=0, column=0, sticky="ew", pady=(0, 4))
        ttk.Button(btn_row, text="Scan Cameras", command=self._scan_cameras).grid(row=1, column=0, sticky="ew", pady=(0, 4))
        ttk.Button(btn_row, text="Connect", command=self._connect).grid(row=2, column=0, sticky="ew")

        ttk.Label(f, text="Status").grid(row=3, column=0, sticky="nw", pady=(8, 0))
        ttk.Label(f, textvariable=self.status_var, wraplength=460, justify="left").grid(
            row=3, column=1, columnspan=2, sticky="w", pady=(8, 0)
        )

    def _build_choice_page(self) -> None:
        f = self.page_choice
        f.columnconfigure(0, weight=1)
        ttk.Label(f, text="Camera connected", font=("Helvetica", 14, "bold")).grid(row=0, column=0, sticky="w", pady=(4, 8))
        ttk.Label(f, textvariable=self.connection_info_var, justify="left").grid(row=1, column=0, sticky="w", pady=(0, 10))
        wmap = ttk.LabelFrame(f, text="Wavelength Mapping (Optional)")
        wmap.grid(row=2, column=0, sticky="ew", pady=(0, 8))
        wmap.columnconfigure(0, weight=1)
        ttk.Entry(wmap, textvariable=self.wavelength_map_path_var).grid(row=0, column=0, sticky="ew", padx=(8, 6), pady=8)
        ttk.Button(wmap, text="Browse...", command=self._browse_wavelength_mapping).grid(row=0, column=1, padx=(0, 8), pady=8)
        ttk.Label(
            wmap,
            text="Format: crop_1_R.png: 730 (48 lines for L1..L16 RGB).",
            justify="left",
        ).grid(row=1, column=0, columnspan=2, sticky="w", padx=8, pady=(0, 8))
        ttk.Button(f, text="Calibrate Camera", command=self._start_calibration_session).grid(row=3, column=0, sticky="w", pady=4)
        ttk.Button(
            f,
            text="Use Calibration From Existing Session",
            command=self._load_existing_calibration,
        ).grid(row=4, column=0, sticky="w", pady=4)
        ttk.Button(f, text="Disconnect", command=self._disconnect).grid(row=5, column=0, sticky="w", pady=8)
        ttk.Label(f, textvariable=self.status_var, wraplength=900, justify="left").grid(row=6, column=0, sticky="w", pady=(8, 0))

    def _build_calibration_page(self) -> None:
        f = self.page_calib
        f.columnconfigure(0, weight=0, minsize=330)
        f.columnconfigure(1, weight=1)
        f.rowconfigure(1, weight=1)

        self.calib_title_var = tk.StringVar(value="Calibration")
        ttk.Label(f, textvariable=self.calib_title_var, font=("Helvetica", 13, "bold")).grid(
            row=0, column=0, columnspan=2, sticky="w", pady=(0, 6)
        )

        self.calib_controls = ttk.Frame(f)
        self.calib_controls.grid(row=1, column=0, sticky="ns", padx=(0, 8))
        self.calib_controls.columnconfigure(0, weight=1)
        self.calib_controls.grid_propagate(True)

        preview_wrap = ttk.LabelFrame(f, text="Live Preview")
        preview_wrap.grid(row=1, column=1, sticky="nsew")
        preview_wrap.columnconfigure(0, weight=1)
        preview_wrap.rowconfigure(0, weight=1)
        self.calib_canvas = tk.Canvas(preview_wrap, bg="#101010", highlightthickness=0)
        self.calib_canvas.grid(row=0, column=0, sticky="nsew")
        self.calib_canvas.bind("<ButtonPress-1>", self._on_calib_canvas_press)
        self.calib_canvas.bind("<B1-Motion>", self._on_calib_canvas_drag)
        self.calib_canvas.bind("<ButtonRelease-1>", self._on_calib_canvas_release)

        self.stage_crop_frame = ttk.LabelFrame(self.calib_controls, text="Crop Calibration")
        self.stage_dark_frame = ttk.LabelFrame(self.calib_controls, text="Black Level Calibration")
        self.stage_flat_frame = ttk.LabelFrame(self.calib_controls, text="Flat Field Calibration")
        self.stage_geom_frame = ttk.LabelFrame(self.calib_controls, text="Geometry Calibration")

        self._build_stage_crop_controls()
        self._build_stage_dark_controls()
        self._build_stage_flat_controls()
        self._build_stage_geometry_controls()

        ttk.Label(f, textvariable=self.status_var, wraplength=980, justify="left").grid(
            row=2, column=0, columnspan=2, sticky="w", pady=(8, 0)
        )

    def _build_main_page(self) -> None:
        f = self.page_main
        f.columnconfigure(0, weight=1)
        f.columnconfigure(1, weight=0)
        f.rowconfigure(0, weight=1)

        preview_wrap = ttk.LabelFrame(f, text="Hydra Preview")
        preview_wrap.grid(row=0, column=0, sticky="nsew", padx=(0, 8))
        preview_wrap.columnconfigure(0, weight=1)
        preview_wrap.rowconfigure(0, weight=1)
        self.main_canvas = tk.Canvas(preview_wrap, bg="#101010", highlightthickness=0)
        self.main_canvas.grid(row=0, column=0, sticky="nsew")
        self.main_canvas.bind("<ButtonPress-1>", self._on_main_canvas_press)
        self.main_canvas.bind("<B1-Motion>", self._on_main_canvas_drag)
        self.main_canvas.bind("<ButtonRelease-1>", self._on_main_canvas_release)

        side = ttk.LabelFrame(f, text="Main Controls")
        side.grid(row=0, column=1, sticky="ns")
        side.columnconfigure(0, weight=1)
        ttk.Radiobutton(side, text="Grid 4x4", variable=self.main_view_mode, value="grid").grid(row=0, column=0, sticky="w", padx=8, pady=(8, 2))
        ttk.Radiobutton(side, text="Single Lens", variable=self.main_view_mode, value="single").grid(row=1, column=0, sticky="w", padx=8, pady=2)
        lens_values = [str(i) for i in range(1, 17)]
        self.main_lens_combo = ttk.Combobox(side, values=lens_values, textvariable=tk.StringVar(value="1"), width=6, state="readonly")
        self.main_lens_combo.grid(row=2, column=0, sticky="w", padx=8, pady=(2, 8))
        self.main_lens_combo.bind("<<ComboboxSelected>>", self._on_main_lens_change)
        ttk.Label(side, text="Face Segmentation / Analysis", font=("Helvetica", 10, "bold")).grid(
            row=3, column=0, sticky="w", padx=8, pady=(2, 2)
        )
        seg_row = ttk.Frame(side)
        seg_row.grid(row=4, column=0, sticky="w", padx=8, pady=(0, 2))
        ttk.Label(seg_row, text="Segmentation lens").pack(side=tk.LEFT)
        self.face_seg_lens_combo = ttk.Combobox(
            seg_row,
            values=lens_values,
            textvariable=self.face_seg_lens_combo_var,
            width=6,
            state="readonly",
        )
        self.face_seg_lens_combo.pack(side=tk.LEFT, padx=(8, 0))
        self.face_seg_lens_combo.bind("<<ComboboxSelected>>", self._on_face_seg_lens_change)
        analyze_row = ttk.Frame(side)
        analyze_row.grid(row=5, column=0, sticky="ew", padx=8, pady=(2, 2))
        analyze_row.columnconfigure(0, weight=1)
        analyze_row.columnconfigure(1, weight=1)
        self.analyze_btn_fake = ttk.Button(analyze_row, textvariable=self.analyze_button_var, command=lambda: self._start_analysis(False))
        self.analyze_btn_fake.grid(row=0, column=0, sticky="ew", padx=(0, 1))
        self.analyze_btn_live = ttk.Button(analyze_row, textvariable=self.analyze_button_var, command=lambda: self._start_analysis(True))
        self.analyze_btn_live.grid(row=0, column=1, sticky="ew", padx=(1, 0))
        self.classify_btn = ttk.Button(side, textvariable=self.classify_button_var, command=self._start_classification)
        self.classify_btn.grid(row=6, column=0, sticky="ew", padx=8, pady=(0, 2))
        self.close_analysis_btn = ttk.Button(side, text="Закрыть", command=self._close_analysis_mode)
        self.close_analysis_btn.grid(row=7, column=0, sticky="ew", padx=8, pady=(0, 2))
        self.close_analysis_btn.state(["disabled"])
        self.analysis_rotate_btn = ttk.Button(side, text="Повернуть 90°", command=self._rotate_analysis_90)
        self.analysis_rotate_btn.grid(row=8, column=0, sticky="ew", padx=8, pady=(0, 2))
        self.analysis_rotate_btn.state(["disabled"])
        smooth = ttk.LabelFrame(side, text="Spectrum Smoothing")
        smooth.grid(row=9, column=0, sticky="ew", padx=8, pady=(2, 4))
        smooth.columnconfigure(1, weight=1)
        ttk.Checkbutton(
            smooth,
            text="Enable",
            variable=self.analysis_smooth_enabled_var,
            command=self._on_analysis_plot_controls_changed,
        ).grid(row=0, column=0, columnspan=2, sticky="w", padx=8, pady=(6, 4))
        ttk.Checkbutton(
            smooth,
            text="Auto Y scale",
            variable=self.analysis_auto_y_var,
            command=self._on_analysis_plot_controls_changed,
        ).grid(row=1, column=0, columnspan=2, sticky="w", padx=8, pady=(0, 4))
        ttk.Label(smooth, text="Sigma").grid(row=2, column=0, sticky="w", padx=8, pady=(2, 2))
        self.analysis_sigma_entry = ttk.Entry(smooth, textvariable=self.analysis_smooth_sigma_var, width=8)
        self.analysis_sigma_entry.grid(row=2, column=1, sticky="w", padx=(0, 8), pady=(2, 2))
        self.analysis_sigma_entry.bind("<Return>", self._on_analysis_plot_controls_changed)
        self.analysis_sigma_entry.bind("<FocusOut>", self._on_analysis_plot_controls_changed)
        ttk.Label(smooth, text="Window").grid(row=3, column=0, sticky="w", padx=8, pady=(2, 6))
        self.analysis_window_entry = ttk.Entry(smooth, textvariable=self.analysis_smooth_window_var, width=8)
        self.analysis_window_entry.grid(row=3, column=1, sticky="w", padx=(0, 8), pady=(2, 6))
        self.analysis_window_entry.bind("<Return>", self._on_analysis_plot_controls_changed)
        self.analysis_window_entry.bind("<FocusOut>", self._on_analysis_plot_controls_changed)
        ttk.Label(side, textvariable=self.face_seg_status_var, wraplength=260, justify="left").grid(
            row=10, column=0, sticky="w", padx=8, pady=(0, 6)
        )
        ttk.Separator(side).grid(row=11, column=0, sticky="ew", padx=8, pady=(2, 6))
        ttk.Label(side, text="Live Camera Controls", font=("Helvetica", 10, "bold")).grid(row=12, column=0, sticky="w", padx=8, pady=(0, 2))
        self._build_main_gain_exposure_controls(side, row_start=13)

        self._build_zaber_controls(side, row_start=17)

        ttk.Button(side, textvariable=self.white_point_button_var, command=self._toggle_white_point_pick).grid(
            row=23, column=0, sticky="ew", padx=8, pady=(6, 4)
        )
        ttk.Label(side, textvariable=self.white_point_info_var, wraplength=260, justify="left").grid(
            row=24, column=0, sticky="w", padx=8, pady=(0, 6)
        )
        ttk.Label(
            side,
            text="Snapshot uses current live Gain/Exposure.",
            wraplength=260,
            justify="left",
        ).grid(row=25, column=0, sticky="w", padx=8, pady=(2, 6))
        ttk.Button(side, text="Snapshot", command=self._snapshot).grid(row=26, column=0, sticky="ew", padx=8, pady=(6, 4))
        ttk.Button(side, text="Back To Choice", command=lambda: self._show_page("choice")).grid(row=27, column=0, sticky="ew", padx=8, pady=4)
        ttk.Button(side, text="Disconnect", command=self._disconnect).grid(row=28, column=0, sticky="ew", padx=8, pady=4)
        ttk.Label(side, textvariable=self.status_var, wraplength=260, justify="left").grid(row=29, column=0, sticky="w", padx=8, pady=(10, 8))

    def _build_zaber_controls(self, parent: ttk.Frame, row_start: int) -> None:
        self.zaber_toggle_btn = ttk.Button(parent, text="Zaber Motion ▸", command=self._toggle_zaber_panel)
        self.zaber_toggle_btn.grid(row=row_start, column=0, sticky="ew", padx=8, pady=(4, 4))

        wrap = ttk.LabelFrame(parent, text="Zaber Motion")
        wrap.grid(row=row_start + 1, column=0, sticky="ew", padx=8, pady=(0, 6))
        wrap.columnconfigure(1, weight=1)
        self.zaber_wrap = wrap

        ttk.Label(wrap, text="Port").grid(row=0, column=0, sticky="w", padx=(8, 6), pady=(8, 2))
        ttk.Entry(wrap, textvariable=self.zaber_port_var, width=22).grid(row=0, column=1, sticky="ew", padx=(0, 6), pady=(8, 2))
        ttk.Button(wrap, text="Connect", command=self._zaber_connect).grid(row=0, column=2, sticky="ew", padx=(0, 8), pady=(8, 2))

        ttk.Label(wrap, text="Axis").grid(row=1, column=0, sticky="w", padx=(8, 6), pady=2)
        ttk.Entry(wrap, textvariable=self.zaber_axis_var, width=6).grid(row=1, column=1, sticky="w", padx=(0, 6), pady=2)
        ttk.Button(wrap, text="Disconnect", command=self._zaber_disconnect).grid(row=1, column=2, sticky="ew", padx=(0, 8), pady=2)

        pos_row = ttk.Frame(wrap)
        pos_row.grid(row=2, column=0, columnspan=3, sticky="ew", padx=8, pady=(2, 2))
        ttk.Label(pos_row, text="Pos (mm)").pack(side=tk.LEFT)
        ttk.Label(pos_row, textvariable=self.zaber_pos_mm_var, width=10).pack(side=tk.LEFT, padx=(8, 0))
        ttk.Label(pos_row, textvariable=self.zaber_status_var, wraplength=200, justify="left").pack(side=tk.LEFT, padx=(8, 0))

        jog_row = ttk.Frame(wrap)
        jog_row.grid(row=3, column=0, columnspan=3, sticky="ew", padx=8, pady=(4, 2))
        ttk.Label(jog_row, text="Step (mm)").pack(side=tk.LEFT)
        ttk.Entry(jog_row, textvariable=self.zaber_step_mm_var, width=8).pack(side=tk.LEFT, padx=(8, 10))
        ttk.Button(jog_row, text="◀ Jog", command=lambda: self._zaber_jog(-1)).pack(side=tk.LEFT)
        ttk.Button(jog_row, text="Jog ▶", command=lambda: self._zaber_jog(+1)).pack(side=tk.LEFT, padx=(6, 0))
        ttk.Button(jog_row, text="Stop", command=self._zaber_stop).pack(side=tk.LEFT, padx=(10, 0))

        goto_row = ttk.Frame(wrap)
        goto_row.grid(row=4, column=0, columnspan=3, sticky="ew", padx=8, pady=(2, 8))
        ttk.Label(goto_row, text="GoTo (mm)").pack(side=tk.LEFT)
        ttk.Entry(goto_row, textvariable=self.zaber_target_mm_var, width=10).pack(side=tk.LEFT, padx=(8, 10))
        ttk.Button(goto_row, text="Go", command=self._zaber_goto).pack(side=tk.LEFT)
        ttk.Button(goto_row, text="Home", command=self._zaber_home).pack(side=tk.LEFT, padx=(6, 0))

        serial_row = ttk.Frame(wrap)
        serial_row.grid(row=5, column=0, columnspan=3, sticky="ew", padx=8, pady=(2, 8))
        ttk.Button(serial_row, text="Get Serials", command=self._serial_start).pack(side=tk.LEFT)
        ttk.Button(serial_row, text="Stop Serials", command=self._serial_stop).pack(side=tk.LEFT, padx=(8, 0))

        ttk.Separator(parent).grid(row=row_start + 2, column=0, sticky="ew", padx=8, pady=(2, 6))

        # Collapsed by default; user can expand from dropdown-like toggle.
        self.zaber_wrap.grid_remove()

    def _toggle_zaber_panel(self) -> None:
        expanded = bool(self.zaber_panel_expanded.get())
        expanded = not expanded
        self.zaber_panel_expanded.set(expanded)
        if hasattr(self, "zaber_wrap"):
            if expanded:
                self.zaber_wrap.grid()
            else:
                self.zaber_wrap.grid_remove()
        if hasattr(self, "zaber_toggle_btn"):
            self.zaber_toggle_btn.configure(text=("Zaber Motion ▾" if expanded else "Zaber Motion ▸"))

    def _build_stage_crop_controls(self) -> None:
        f = self.stage_crop_frame
        f.columnconfigure(0, weight=1)
        ttk.Label(
            f,
            text="Place chessboard 4x4 cells in view.\nDetect center seam (3x3 inner center) on all 16 lenses,\nthen adjust global offsets and save crop.",
            wraplength=250,
            justify="left",
        ).grid(row=0, column=0, sticky="w", padx=8, pady=(8, 8))
        self._build_gain_exposure_controls(f, row_start=1, allow_lock=False)
        ttk.Button(f, text="Detect 16 Chess Centers", command=self._detect_crop_points).grid(row=6, column=0, sticky="ew", padx=8, pady=(4, 4))
        ttk.Checkbutton(f, text="Manual point drag", variable=self.manual_points_var).grid(row=7, column=0, sticky="w", padx=8, pady=(0, 6))
        self._add_offset_row(f, "Left", self.crop_left_var, 8, self._on_crop_offsets_changed)
        self._add_offset_row(f, "Right", self.crop_right_var, 9, self._on_crop_offsets_changed)
        self._add_offset_row(f, "Top", self.crop_top_var, 10, self._on_crop_offsets_changed)
        self._add_offset_row(f, "Bottom", self.crop_bottom_var, 11, self._on_crop_offsets_changed)
        ttk.Button(f, text="Save Crop And Continue", command=self._save_crop_and_continue).grid(row=12, column=0, sticky="ew", padx=8, pady=(8, 8))
        ttk.Button(f, text="Load files", command=lambda: self._load_stage_files("crop")).grid(
            row=13, column=0, sticky="ew", padx=8, pady=(0, 8)
        )

    def _build_stage_dark_controls(self) -> None:
        f = self.stage_dark_frame
        f.columnconfigure(0, weight=1)
        ttk.Label(
            f,
            text="Set Exposure/Gain, then cover sensor and disable light.\nCreate dark map from RAW burst.",
            wraplength=250,
            justify="left",
        ).grid(row=0, column=0, sticky="w", padx=8, pady=(8, 8))
        self._build_gain_exposure_controls(f, row_start=1, allow_lock=True)
        ttk.Label(f, textvariable=self.max_fps_var).grid(row=6, column=0, sticky="w", padx=8, pady=(4, 2))
        burst_row = ttk.Frame(f)
        burst_row.grid(row=7, column=0, sticky="ew", padx=8, pady=(2, 6))
        ttk.Label(burst_row, text="Burst frames").pack(side=tk.LEFT)
        ttk.Entry(burst_row, textvariable=self.dark_burst_var, width=8).pack(side=tk.LEFT, padx=(8, 0))
        ttk.Button(f, text="Create Dark Map", command=self._start_dark_capture).grid(row=8, column=0, sticky="ew", padx=8, pady=(8, 8))
        ttk.Button(f, text="Load files", command=lambda: self._load_stage_files("dark")).grid(
            row=9, column=0, sticky="ew", padx=8, pady=(0, 8)
        )

    def _build_stage_flat_controls(self) -> None:
        f = self.stage_flat_frame
        f.columnconfigure(0, weight=1)
        ttk.Label(
            f,
            text="Use uniform bright target (Spectralon).",
            wraplength=250,
            justify="left",
        ).grid(row=0, column=0, sticky="w", padx=8, pady=(8, 8))
        burst_row = ttk.Frame(f)
        burst_row.grid(row=1, column=0, sticky="ew", padx=8, pady=(2, 4))
        ttk.Label(burst_row, text="Burst frames").pack(side=tk.LEFT)
        ttk.Entry(burst_row, textvariable=self.flat_burst_var, width=8).pack(side=tk.LEFT, padx=(8, 0))
        sigma_row = ttk.Frame(f)
        sigma_row.grid(row=2, column=0, sticky="ew", padx=8, pady=(2, 8))
        ttk.Label(sigma_row, text="Low-pass sigma").pack(side=tk.LEFT)
        ttk.Entry(sigma_row, textvariable=self.flat_sigma_var, width=8).pack(side=tk.LEFT, padx=(8, 0))
        ttk.Button(f, text="Create Flat Map", command=self._start_flat_capture).grid(row=3, column=0, sticky="ew", padx=8, pady=(8, 8))
        ttk.Button(f, text="Load files", command=lambda: self._load_stage_files("flat")).grid(
            row=4, column=0, sticky="ew", padx=8, pady=(0, 8)
        )
        ttk.Button(f, text="Skip Flat Field Calibration", command=self._skip_flat_calibration).grid(
            row=5, column=0, sticky="ew", padx=8, pady=(0, 8)
        )

    def _build_stage_geometry_controls(self) -> None:
        f = self.stage_geom_frame
        f.columnconfigure(0, weight=1)
        ttk.Label(
            f,
            text="Chessboard 4x4 cells => 3x3 inner corners.\nCapture 5 valid frames with board on all 16 lenses.",
            wraplength=250,
            justify="left",
        ).grid(row=0, column=0, sticky="w", padx=8, pady=(8, 8))
        row_cfg = ttk.Frame(f)
        row_cfg.grid(row=1, column=0, sticky="ew", padx=8, pady=(0, 4))
        ttk.Label(row_cfg, text="Inner corners").pack(side=tk.LEFT)
        ttk.Entry(row_cfg, textvariable=self.geometry_cols_var, width=4).pack(side=tk.LEFT, padx=(8, 2))
        ttk.Label(row_cfg, text="x").pack(side=tk.LEFT)
        ttk.Entry(row_cfg, textvariable=self.geometry_rows_var, width=4).pack(side=tk.LEFT, padx=(2, 0))
        row_n = ttk.Frame(f)
        row_n.grid(row=2, column=0, sticky="ew", padx=8, pady=(0, 4))
        ttk.Label(row_n, text="Valid frames target").pack(side=tk.LEFT)
        ttk.Entry(row_n, textvariable=self.geometry_target_var, width=6).pack(side=tk.LEFT, padx=(8, 0))
        self.geometry_btn_var = tk.StringVar(value="Capture chess frame 1/5")
        ttk.Button(f, textvariable=self.geometry_btn_var, command=self._capture_geometry_frame).grid(
            row=3, column=0, sticky="ew", padx=8, pady=(8, 4)
        )
        ttk.Progressbar(f, maximum=100, variable=self.geometry_progress_var).grid(row=4, column=0, sticky="ew", padx=8, pady=(6, 2))
        ttk.Label(f, textvariable=self.geometry_progress_text_var, wraplength=250, justify="left").grid(
            row=5, column=0, sticky="w", padx=8, pady=(0, 8)
        )
        ttk.Button(f, text="Load files", command=lambda: self._load_stage_files("geometry")).grid(
            row=6, column=0, sticky="ew", padx=8, pady=(0, 8)
        )
        ttk.Button(f, text="Skip Geometry Calibration", command=self._skip_geometry_calibration).grid(
            row=7, column=0, sticky="ew", padx=8, pady=(0, 8)
        )

    def _build_gain_exposure_controls(self, parent: ttk.Frame, row_start: int, allow_lock: bool) -> None:
        ttk.Label(parent, text="Gain").grid(row=row_start, column=0, sticky="w", padx=8, pady=(2, 0))
        gain_row = ttk.Frame(parent)
        gain_row.grid(row=row_start + 1, column=0, sticky="ew", padx=8, pady=(2, 6))
        gain_row.columnconfigure(0, weight=1)
        scale = ttk.Scale(gain_row, from_=self.gain_bounds[0], to=self.gain_bounds[1], variable=self.gain_var, orient=tk.HORIZONTAL, command=self._on_gain_slide)
        scale.grid(row=0, column=0, sticky="ew")
        ent = ttk.Entry(gain_row, textvariable=self.gain_entry_var, width=8)
        ent.grid(row=0, column=1, padx=(6, 0))
        ent.bind("<Return>", self._on_gain_entry_commit)

        ttk.Label(parent, text="Exposure (us)").grid(row=row_start + 2, column=0, sticky="w", padx=8, pady=(2, 0))
        exp_row = ttk.Frame(parent)
        exp_row.grid(row=row_start + 3, column=0, sticky="ew", padx=8, pady=(2, 6))
        exp_row.columnconfigure(0, weight=1)
        scale_e = ttk.Scale(
            exp_row,
            from_=self.exposure_bounds[0],
            to=self.exposure_bounds[1],
            variable=self.exposure_var,
            orient=tk.HORIZONTAL,
            command=self._on_exposure_slide,
        )
        scale_e.grid(row=0, column=0, sticky="ew")
        ent_e = ttk.Entry(exp_row, textvariable=self.exposure_entry_var, width=10)
        ent_e.grid(row=0, column=1, padx=(6, 0))
        ent_e.bind("<Return>", self._on_exposure_entry_commit)
        if not allow_lock:
            return

    def _build_main_gain_exposure_controls(self, parent: ttk.Frame, row_start: int) -> None:
        ttk.Label(parent, text="Gain").grid(row=row_start, column=0, sticky="w", padx=8, pady=(2, 0))
        gain_row = ttk.Frame(parent)
        gain_row.grid(row=row_start + 1, column=0, sticky="ew", padx=8, pady=(2, 6))
        gain_row.columnconfigure(0, weight=1)
        scale = ttk.Scale(
            gain_row,
            from_=self.gain_bounds[0],
            to=self.gain_bounds[1],
            variable=self.gain_var,
            orient=tk.HORIZONTAL,
            command=self._on_gain_slide,
        )
        scale.grid(row=0, column=0, sticky="ew")
        ent = ttk.Entry(gain_row, textvariable=self.gain_entry_var, width=8)
        ent.grid(row=0, column=1, padx=(6, 0))
        ent.bind("<Return>", self._on_gain_entry_commit)

        ttk.Label(parent, text="Exposure (us)").grid(row=row_start + 2, column=0, sticky="w", padx=8, pady=(2, 0))
        exp_row = ttk.Frame(parent)
        exp_row.grid(row=row_start + 3, column=0, sticky="ew", padx=8, pady=(2, 6))
        exp_row.columnconfigure(0, weight=1)
        scale_e = ttk.Scale(
            exp_row,
            from_=self.exposure_bounds[0],
            to=self.exposure_bounds[1],
            variable=self.exposure_var,
            orient=tk.HORIZONTAL,
            command=self._on_exposure_slide,
        )
        scale_e.grid(row=0, column=0, sticky="ew")
        ent_e = ttk.Entry(exp_row, textvariable=self.exposure_entry_var, width=10)
        ent_e.grid(row=0, column=1, padx=(6, 0))
        ent_e.bind("<Return>", self._on_exposure_entry_commit)

    def _add_offset_row(self, parent: ttk.Frame, label: str, var: tk.IntVar, row: int, callback) -> None:
        r = ttk.Frame(parent)
        r.grid(row=row, column=0, sticky="ew", padx=8, pady=(0, 3))
        r.columnconfigure(1, weight=1)
        ttk.Label(r, text=label).grid(row=0, column=0, sticky="w")
        sc = ttk.Scale(
            r,
            from_=1,
            to=300,
            orient=tk.HORIZONTAL,
            command=lambda value, v=var: self._on_offset_scale(v, value, callback),
        )
        sc.set(float(var.get()))
        sc.grid(row=0, column=1, sticky="ew", padx=(8, 6))
        ent = ttk.Entry(r, textvariable=var, width=7)
        ent.grid(row=0, column=2, sticky="e")
        ent.bind("<Return>", lambda _e: callback())

    def _on_offset_scale(self, var: tk.IntVar, value: str, callback) -> None:
        try:
            var.set(int(round(float(value))))
        except Exception:
            return
        callback()

    # ----------------------------- Navigation -----------------------------
    def _show_page(self, page: str) -> None:
        self.current_page = page
        if page == "connect":
            self.geometry("900x320")
            self.page_connect.tkraise()
        elif page == "choice":
            self.geometry("760x360")
            self.page_choice.tkraise()
        elif page == "calib":
            self.geometry("1320x860")
            self.page_calib.tkraise()
        elif page == "main":
            self.geometry("1320x860")
            self.page_main.tkraise()
            self._start_face_seg_model_preload()
            self._start_face_cls_model_preload()
            self._start_analysis_recon_precheck()
            self._update_analyze_button_state()

    def _show_calibration_stage(self, stage: str) -> None:
        if stage == "dark":
            # Dark stage is deprecated; proceed directly to flat calibration.
            stage = "flat"
        self.calib_stage = stage
        for fr in (self.stage_crop_frame, self.stage_dark_frame, self.stage_flat_frame, self.stage_geom_frame):
            fr.grid_remove()
        if stage == "crop":
            self.calib_title_var.set("Crop Calibration")
            self.stage_crop_frame.grid(row=0, column=0, sticky="nsew")
        elif stage == "dark":
            self.calib_title_var.set("Black Level Calibration")
            self.stage_dark_frame.grid(row=0, column=0, sticky="nsew")
        elif stage == "flat":
            self.calib_title_var.set("Flat Field Calibration")
            self.stage_flat_frame.grid(row=0, column=0, sticky="nsew")
        elif stage == "geometry":
            self.calib_title_var.set("Geometry Calibration")
            self.stage_geom_frame.grid(row=0, column=0, sticky="nsew")

    # ----------------------------- Camera connect -----------------------------
    def _on_camera_pick_selected(self, _event: tk.Event | None = None) -> None:
        label = self.camera_pick_var.get().strip()
        camera_id = self.camera_scan_display_to_id.get(label)
        if camera_id:
            self.camera_var.set(camera_id)

    @staticmethod
    def _parse_arv_list_output(text: str) -> list[tuple[str, str]]:
        out: list[tuple[str, str]] = []
        seen: set[str] = set()
        for raw in text.splitlines():
            line = raw.strip()
            if not line:
                continue
            if line.startswith("["):
                continue
            lo = line.lower()
            if lo.startswith("no device found"):
                continue
            if lo.startswith("option parsing failed"):
                continue
            if lo.startswith("error"):
                continue
            if lo.startswith("warning"):
                continue
            dev_id = line
            transport = "unknown"
            if " (" in line and line.endswith(")"):
                head, tail = line.rsplit(" (", 1)
                tail = tail[:-1].strip()
                if head.strip():
                    dev_id = head.strip()
                    transport = tail or "unknown"
            if dev_id in seen:
                continue
            seen.add(dev_id)
            out.append((dev_id, transport))
        return out

    def _scan_cameras(self) -> None:
        if self.camera_scan_running:
            self.status_var.set("Camera scan already running")
            return
        self.camera_scan_running = True
        interface = self.interface_var.get().strip()
        self.status_var.set("Scanning cameras...")
        threading.Thread(target=self._scan_cameras_worker, args=(interface,), daemon=True).start()

    def _scan_cameras_via_arv(self, interface: str, timeout_s: float = 6.0) -> list[tuple[str, str]]:
        arv_tool = shutil.which("arv-tool-0.8") or shutil.which("arv-tool")
        if not arv_tool:
            return []
        cmds: list[list[str]] = [[arv_tool]]
        if interface:
            cmds.insert(0, [arv_tool, f"--gv-discovery-interface={interface}"])
        merged: list[tuple[str, str]] = []
        seen: set[str] = set()
        for cmd in cmds:
            try:
                out = subprocess.check_output(cmd, text=True, stderr=subprocess.STDOUT, timeout=timeout_s)
            except Exception as exc:
                out = str(exc)
            for dev_id, transport in self._parse_arv_list_output(out):
                if dev_id in seen:
                    continue
                seen.add(dev_id)
                merged.append((dev_id, transport))
        return merged

    def _scan_cameras_worker(self, interface: str) -> None:
        def emit(kind: str, payload: object) -> None:
            try:
                self.event_q.put_nowait((kind, payload))
            except Exception:
                pass

        try:
            if not (shutil.which("arv-tool-0.8") or shutil.which("arv-tool")):
                emit("camera_scan_done", {"ok": False, "error": "arv-tool not found"})
                return
            merged = self._scan_cameras_via_arv(interface=interface, timeout_s=6.0)
            emit("camera_scan_done", {"ok": True, "devices": merged})
        finally:
            emit("camera_scan_finish", None)

    def _connect(self) -> None:
        if self.worker and self.worker.is_alive():
            self.status_var.set("Already connected")
            return
        interface = self.interface_var.get().strip()
        camera_id = self.camera_var.get().strip()
        self._dbg(f"connect requested: interface={interface} camera={camera_id}")
        if not camera_id:
            self.status_var.set("Empty camera ID / IP")
            return
        if (not interface) and is_ipv4_literal(camera_id):
            self.status_var.set("Empty interface for GigE camera IP")
            return
        if interface and is_ipv4_literal(camera_id):
            entries = get_interface_ipv4_entries(interface)
            local_ips = {ip for ip, _mask in entries}
            if camera_id in local_ips:
                self.status_var.set(
                    f"Camera IP {camera_id} is assigned to local interface {interface}. Remove that alias first."
                )
                return
            if entries and not any(same_subnet(ip, camera_id, mask) for ip, mask in entries):
                host_list = ", ".join(ip for ip, _ in entries)
                self.status_var.set(
                    f"Camera IP {camera_id} is outside interface subnets ({host_list}). "
                    "Add matching alias (e.g. 169.254.x.x/16)."
                )
                return
            host_ip = get_interface_ipv4_for_peer(interface, camera_id) or get_interface_ipv4(interface)
            if host_ip and camera_id == host_ip:
                self.status_var.set(f"Camera IP equals host IP ({host_ip}). Use camera IP (typically x.x.x.1).")
                return
        self.event_q = queue.Queue(maxsize=96)
        self.cmd_q = queue.Queue(maxsize=64)
        self.worker = CameraWorker(
            interface=(interface or "en10"),
            camera_ip=camera_id,
            event_q=self.event_q,
            cmd_q=self.cmd_q,
            packet_size=self.packet_size,
            packet_delay=self.packet_delay,
            buffers=24,
            debug=self.debug,
        )
        self.status_var.set("Connecting...")
        self.worker.start()

    def _disconnect(self) -> None:
        self._dbg("disconnect requested")
        if self.worker and self.worker.is_alive():
            self.worker.stop()
            self.status_var.set("Disconnecting...")
        self.connected = False
        self.last_frame = None
        self._close_analysis_mode()
        self._analysis_busy = False
        self._update_analyze_button_state()
        self._show_page("connect")

    def _auto_find_fix(self) -> None:
        if self.auto_fix_running:
            self.status_var.set("Auto Find/Fix already running")
            return
        if not AUTO_FIX_AVAILABLE:
            self.status_var.set("Auto Find/Fix unavailable")
            return
        interface = self.interface_var.get().strip()
        if not interface:
            self.status_var.set("Empty interface")
            return
        self.auto_fix_running = True
        self.status_var.set("Auto Find/Fix started...")
        threading.Thread(target=self._auto_find_fix_worker, args=(interface,), daemon=True).start()

    def _auto_find_fix_worker(self, interface: str) -> None:
        def emit(kind: str, payload: object) -> None:
            try:
                self.event_q.put_nowait((kind, payload))
            except Exception:
                pass

        try:
            host_ip = get_interface_ipv4(interface)
            self._dbg(f"auto_find_fix: host_ip on {interface} = {host_ip}")
            if not host_ip:
                emit("status", f"Auto Find/Fix failed: no IPv4 on {interface}")
                emit("auto_fix_done", None)
                return
            netmask = get_interface_netmask(interface) or "255.255.255.0"
            local_ips = {ip for ip, _mask in get_interface_ipv4_entries(interface)}
            target_ip = suggest_camera_ip(host_ip, avoid=local_ips)

            cams = gvcp_discover(interface, duration=3.5, interval=0.25) if gvcp_discover else []
            self._dbg(f"auto_find_fix: discovered cameras count={len(cams)}")
            if not cams:
                # Fallback: Aravis scan can still succeed in some network setups where
                # raw GVCP broadcast probing is filtered or unstable.
                fallback = self._scan_cameras_via_arv(interface=interface, timeout_s=4.0)
                if fallback:
                    pick_id = fallback[0][0]
                    pick_transport = fallback[0][1]
                    if is_ipv4_literal(pick_transport):
                        discovered_ip = pick_transport
                        need_force = (not same_subnet(discovered_ip, host_ip, netmask)) or (discovered_ip != target_ip)
                        if need_force and gvcp_force_ip:
                            mac = get_arp_mac_for_ip(interface, discovered_ip)
                            if not mac:
                                mac = get_arp_mac_for_ip(interface, target_ip)
                            if not mac:
                                arp_entries = get_arp_entries(interface)
                                for arp_ip, arp_mac in arp_entries:
                                    if arp_ip == host_ip:
                                        continue
                                    if arp_ip.startswith("224.") or arp_ip.startswith("169.254.255.255"):
                                        continue
                                    mac = arp_mac
                                    break
                            if mac:
                                emit("status", f"GVCP timeout; applying ForceIP {target_ip} via ARP MAC {mac}")
                                try:
                                    gvcp_force_ip(
                                        interface=interface,
                                        target_mac=mac,
                                        ip=target_ip,
                                        subnet=netmask,
                                        gateway="0.0.0.0",
                                        timeout=1.2,
                                    )
                                    time.sleep(0.4)
                                    fallback_after = self._scan_cameras_via_arv(interface=interface, timeout_s=2.5)
                                    promoted = None
                                    for dev_id2, transport2 in fallback_after:
                                        if is_ipv4_literal(transport2):
                                            promoted = transport2
                                            if transport2 == target_ip:
                                                break
                                    if promoted:
                                        discovered_ip = promoted
                                except Exception as exc:
                                    emit("status", f"ForceIP fallback error: {exc}")
                            else:
                                emit("status", f"GVCP timeout; cannot resolve ARP MAC for {discovered_ip}")
                        emit("auto_fix_set_ip", discovered_ip)
                        emit("status", f"GVCP timeout; Aravis sees {pick_id}, using IP {discovered_ip}")
                        emit("auto_fix_done", None)
                        return
                    # For GigE, Aravis camera IDs are often not routable endpoints.
                    # Prefer guessed camera IPv4 on the host subnet to keep connect flow stable.
                    if ("gige" in pick_transport.lower()) and (not is_ipv4_literal(pick_id)):
                        guess_ip = suggest_camera_ip(host_ip)
                        emit("auto_fix_set_ip", guess_ip)
                        emit(
                            "status",
                            f"GVCP timeout; Aravis sees {pick_id} ({pick_transport}), trying guessed IP {guess_ip}",
                        )
                    else:
                        emit("auto_fix_set_ip", pick_id)
                        emit("status", f"GVCP timeout; using Aravis camera id: {pick_id} ({pick_transport})")
                    emit("auto_fix_done", None)
                    return
                emit("status", "No GVCP replies")
                emit("auto_fix_done", None)
                return
            cam = cams[0]
            mac = (cam.mac or "").lower()
            discovered_ip = cam.current_ip or cam.source_ip
            emit("status", f"Found {cam.model_name} at {discovered_ip}")

            need_force = (not same_subnet(discovered_ip, host_ip, netmask)) or (discovered_ip != target_ip)
            if need_force and mac and gvcp_force_ip:
                emit("status", f"Applying ForceIP {target_ip}")
                try:
                    gvcp_force_ip(interface=interface, target_mac=mac, ip=target_ip, subnet=netmask, gateway="0.0.0.0", timeout=1.2)
                    time.sleep(0.4)
                except Exception as exc:
                    emit("status", f"ForceIP error: {exc}")
                cams_after = gvcp_discover(interface, duration=2.8, interval=0.2) if gvcp_discover else []
                if cams_after:
                    picked = None
                    for c in cams_after:
                        if (c.mac or "").lower() == mac:
                            picked = c
                            break
                    if picked is None:
                        picked = cams_after[0]
                    discovered_ip = picked.current_ip or picked.source_ip
            emit("auto_fix_set_ip", discovered_ip)
            emit("status", f"Auto Find/Fix done: {discovered_ip}")
        except Exception as exc:
            emit("status", f"Auto Find/Fix failed: {exc}")
        finally:
            emit("auto_fix_done", None)

    # ----------------------------- Session / calibration filesystem -----------------------------
    def _browse_wavelength_mapping(self) -> None:
        path = filedialog.askopenfilename(
            title="Select wavelength mapping file (optional)",
            filetypes=[("Text files", "*.txt *.csv"), ("All files", "*.*")],
        )
        if not path:
            return
        self.wavelength_map_path_var.set(path)
        ok, msg = self._load_wavelength_mapping_from_path(path, silent=False)
        if ok:
            self.status_var.set(msg)
        else:
            self.status_var.set(f"Wavelength mapping error: {msg}")

    def _parse_wavelength_mapping_file(self, path: Path) -> tuple[list[dict[str, object]], list[str]]:
        if not path.exists():
            raise RuntimeError(f"File not found: {path}")
        entries: list[dict[str, object]] = []
        warnings: list[str] = []
        seen: set[int] = set()
        line_re = re.compile(r"^\s*([^:\s]+)\s*:\s*([-+]?\d+(?:\.\d+)?)\s*$")
        name_re = re.compile(r"(?i)(?:^|.*?)(?:crop_)?(\d+)_([rgb])(?:\.[a-z0-9_]+)?$")
        for lineno, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            m = line_re.match(line)
            if not m:
                warnings.append(f"line {lineno}: skipped (invalid format)")
                continue
            name = m.group(1).strip()
            try:
                wl = float(m.group(2))
            except Exception:
                warnings.append(f"line {lineno}: skipped (invalid wavelength)")
                continue
            nm = name_re.match(name)
            if not nm:
                warnings.append(f"line {lineno}: skipped (cannot parse lens/channel from '{name}')")
                continue
            lens = int(nm.group(1))
            ch = str(nm.group(2)).upper()
            if lens < 1 or lens > 16:
                warnings.append(f"line {lineno}: skipped (lens out of range 1..16)")
                continue
            ch_off = {"R": 0, "G": 1, "B": 2}[ch]
            channel_index = (lens - 1) * 3 + ch_off
            if channel_index in seen:
                warnings.append(f"line {lineno}: duplicate mapping for L{lens}_{ch}, skipped")
                continue
            seen.add(channel_index)
            entries.append(
                {
                    "line": lineno,
                    "source_name": name,
                    "lens": lens,
                    "channel": ch,
                    "label": f"L{lens}_{ch}",
                    "channel_index": channel_index,
                    "wavelength_nm": wl,
                }
            )
        if not entries:
            raise RuntimeError("No valid mapping rows were parsed")
        return entries, warnings

    def _load_wavelength_mapping_from_path(self, path_str: str, silent: bool = False) -> tuple[bool, str]:
        p = Path(path_str).expanduser()
        try:
            entries, warnings = self._parse_wavelength_mapping_file(p)
        except Exception as exc:
            if not silent:
                self._dbg(f"wavelength mapping parse failed: {exc}")
            self.wavelength_mapping_entries = []
            self.wavelength_mapping_source = None
            return False, str(exc)
        self.wavelength_mapping_entries = entries
        self.wavelength_mapping_source = str(p)
        msg = f"Wavelength mapping loaded: {len(entries)} channels from {p.name}"
        if warnings and not silent and self.debug:
            self._dbg("wavelength mapping warnings: " + "; ".join(warnings[:8]))
        return True, msg

    def _save_wavelength_mapping_to_session(self) -> None:
        if self.session_root is None:
            return
        payload: dict[str, object] = {
            "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            "source_path": self.wavelength_mapping_source,
            "count": int(len(self.wavelength_mapping_entries)),
            "entries": self.wavelength_mapping_entries,
            "expected_labels": [f"L{i}_{ch}" for i in range(1, 17) for ch in ("R", "G", "B")],
            "source_order_note": "entries order follows file lines and is used for snapshot channel ordering",
        }
        self._save_json(self.session_root / "wavelength_mapping.json", payload)

    def _load_wavelength_mapping_from_session(self, session: Path) -> None:
        mapping_path = session / "wavelength_mapping.json"
        if not mapping_path.exists():
            return
        try:
            payload = json.loads(mapping_path.read_text(encoding="utf-8"))
            raw_entries = payload.get("entries", [])
            parsed: list[dict[str, object]] = []
            for e in raw_entries:
                if not isinstance(e, dict):
                    continue
                if "channel_index" not in e or "wavelength_nm" not in e or "label" not in e:
                    continue
                parsed.append(
                    {
                        "line": int(e.get("line", 0)),
                        "source_name": str(e.get("source_name", "")),
                        "lens": int(e.get("lens", 0)),
                        "channel": str(e.get("channel", "")),
                        "label": str(e.get("label")),
                        "channel_index": int(e.get("channel_index")),
                        "wavelength_nm": float(e.get("wavelength_nm")),
                    }
                )
            if parsed:
                self.wavelength_mapping_entries = parsed
                src = payload.get("source_path", "")
                self.wavelength_mapping_source = str(src) if src else str(mapping_path)
                self.wavelength_map_path_var.set(self.wavelength_mapping_source)
        except Exception as exc:
            if self.debug:
                self._dbg(f"failed to load session wavelength mapping: {exc}")

    def _clear_white_point_state(self) -> None:
        self._white_pick_active = False
        self._white_pick_start_xy = None
        self._white_pick_end_xy = None
        self.white_point_roi = None
        self.white_point_lens = None
        self.white_point_ref = None
        self.white_point_created_utc = None
        self.white_point_button_var.set("Задать точку белого")
        self.white_point_info_var.set("White point: not set")
        self._update_analyze_button_state()

    def _update_white_point_info(self) -> None:
        if self.white_point_ref is None or int(np.asarray(self.white_point_ref).size) <= 0:
            self.white_point_info_var.set("White point: not set")
            return
        if self.white_point_roi is None or self.white_point_lens is None:
            self.white_point_info_var.set(f"White point: configured ({int(np.asarray(self.white_point_ref).size)} channels)")
            return
        x0, y0, x1, y1 = self.white_point_roi
        w = max(0, int(x1) - int(x0))
        h = max(0, int(y1) - int(y0))
        created = str(self.white_point_created_utc or "")
        created_short = created.replace("T", " ").replace("+00:00", "Z") if created else "-"
        self.white_point_info_var.set(
            f"White point: L{int(self.white_point_lens) + 1}, ROI {w}x{h} @ ({int(x0)}, {int(y0)}), {created_short}"
        )

    def _save_white_point_to_session(self) -> None:
        if self.session_root is None:
            return
        if self.white_point_ref is None or self.white_point_roi is None or self.white_point_lens is None:
            return
        try:
            ref = np.asarray(self.white_point_ref, dtype=np.float32).reshape(-1)
            if ref.size == 3:
                ref = np.tile(ref, 16)
            if ref.size != 48:
                raise RuntimeError(f"invalid white point size: {ref.size}")
            self.session_root.mkdir(parents=True, exist_ok=True)
            np.save(self.session_root / "white_point_reference.npy", ref.astype(np.float32))
            self._save_json(
                self.session_root / "white_point.json",
                {
                    "created_utc": str(self.white_point_created_utc or dt.datetime.now(dt.timezone.utc).isoformat()),
                    "lens_index": int(self.white_point_lens),
                    "lens_number": int(self.white_point_lens) + 1,
                    "roi_xyxy": [int(v) for v in self.white_point_roi],
                    "channels": int(ref.size),
                    "source": "single_lens_roi_mean",
                },
            )
        except Exception as exc:
            if self.debug:
                self._dbg(f"failed to save white point: {exc}")

    def _load_white_point_from_session(self, session: Path) -> None:
        ref_path = session / "white_point_reference.npy"
        if not ref_path.exists():
            return
        try:
            ref = np.asarray(np.load(ref_path), dtype=np.float32).reshape(-1)
            if ref.size == 3:
                ref = np.tile(ref, 16)
            if ref.size != 48:
                raise RuntimeError(f"invalid white point size: {ref.size}")
            self.white_point_ref = np.maximum(ref.astype(np.float32), 1e-6)
            meta_path = session / "white_point.json"
            if meta_path.exists():
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
                roi_raw = meta.get("roi_xyxy")
                if isinstance(roi_raw, list) and len(roi_raw) == 4:
                    self.white_point_roi = tuple(int(v) for v in roi_raw)  # type: ignore[assignment]
                lens_idx = meta.get("lens_index")
                if lens_idx is not None:
                    self.white_point_lens = int(lens_idx)
                self.white_point_created_utc = str(meta.get("created_utc", ""))
            self._white_pick_active = False
            self._white_pick_start_xy = None
            self._white_pick_end_xy = None
            self.white_point_button_var.set("Задать точку белого")
            self._update_white_point_info()
            self._update_analyze_button_state()
        except Exception as exc:
            self._clear_white_point_state()
            if self.debug:
                self._dbg(f"failed to load white point from session: {exc}")

    def _start_calibration_session(self) -> None:
        ts = dt.datetime.now().strftime("%Y-%m-%d_%H-%M-%S_%f")
        self.session_root = self.output_dir / f"session_{ts}_calibration"
        self.session_root.mkdir(parents=True, exist_ok=True)
        self.geometry_h = None
        self.geometry_correction_enabled = True
        self.reference_lens = 0
        self.geometry_capture_idx = 0
        self.geometry_corners = []
        self.geometry_focus_scores = []
        self.geometry_progress_var.set(0.0)
        self.geometry_progress_text_var.set("")
        self.dark_map = None
        self.noise_map = None
        self.locked_gain = None
        self.locked_exposure = None
        self.flat_raw_mean = None
        self.flat_norm = None
        self.wavelength_mapping_entries = []
        self.wavelength_mapping_source = None
        self._clear_white_point_state()
        map_path = self.wavelength_map_path_var.get().strip()
        if map_path:
            ok, msg = self._load_wavelength_mapping_from_path(map_path, silent=False)
            if not ok:
                self.status_var.set(f"Wavelength mapping warning: {msg}")
            else:
                self._save_wavelength_mapping_to_session()
        self._show_page("calib")
        self._show_calibration_stage("crop")
        if self.wavelength_mapping_entries:
            self.status_var.set(
                "Crop calibration: detect 16 chessboard centers (4x4 board, center of 3x3 inner corners). "
                f"Wavelength map loaded ({len(self.wavelength_mapping_entries)} channels)."
            )
        else:
            self.status_var.set("Crop calibration: detect 16 chessboard centers (4x4 board, center of 3x3 inner corners)")

    def _copy_files_to_current_session(self, source_session: Path, file_names: list[str]) -> None:
        if self.session_root is None:
            return
        self.session_root.mkdir(parents=True, exist_ok=True)
        for name in file_names:
            src = source_session / name
            if not src.exists() or not src.is_file():
                continue
            dst = self.session_root / name
            try:
                if src.resolve() == dst.resolve():
                    continue
            except Exception:
                pass
            shutil.copy2(src, dst)

    def _load_crop_from_session(self, session: Path) -> None:
        crop = json.loads((session / "crop.json").read_text(encoding="utf-8"))
        boxes_raw = crop.get("boxes", [])
        if len(boxes_raw) != 16:
            raise RuntimeError("crop.json has invalid boxes")
        boxes: list[dict[str, int]] = []
        for b in boxes_raw:
            boxes.append(
                {
                    "index": int(b["index"]),
                    "row": int(b["row"]),
                    "col": int(b["col"]),
                    "x": int(b["x"]),
                    "y": int(b["y"]),
                    "width": int(b["width"]),
                    "height": int(b["height"]),
                }
            )
        self.crop_boxes = boxes
        centers_raw = crop.get("centers_xy", [])
        if isinstance(centers_raw, list) and len(centers_raw) == 16:
            try:
                arr = np.asarray(centers_raw, dtype=np.float32)
                if arr.shape == (16, 2):
                    self.crop_centers = arr
            except Exception:
                pass
        offsets = crop.get("offsets", {})
        if isinstance(offsets, dict):
            try:
                self.crop_left_var.set(int(offsets.get("left", self.crop_left_var.get())))
                self.crop_right_var.set(int(offsets.get("right", self.crop_right_var.get())))
                self.crop_top_var.set(int(offsets.get("top", self.crop_top_var.get())))
                self.crop_bottom_var.set(int(offsets.get("bottom", self.crop_bottom_var.get())))
            except Exception:
                pass

    def _load_dark_from_session(self, session: Path) -> None:
        # Dark calibration is deprecated and intentionally ignored.
        _ = session
        self.dark_map = None
        self.noise_map = None
        self.locked_gain = None
        self.locked_exposure = None
        self.dark_capture_active = False

    def _load_flat_from_session(self, session: Path) -> None:
        if self._read_flat_meta_from_session(session):
            self.flat_norm = None
            self.flat_raw_mean = None
            self.flat_capture_active = False
            return
        flat_norm_path = session / "flat_norm.npy"
        if not flat_norm_path.exists():
            raise RuntimeError("Missing file: flat_norm.npy")
        self.flat_norm = np.load(flat_norm_path)
        flat_raw_mean_path = session / "flat_raw_mean.npy"
        if flat_raw_mean_path.exists():
            self.flat_raw_mean = np.load(flat_raw_mean_path)
        self.flat_capture_active = False

    def _read_flat_meta_from_session(self, session: Path) -> bool:
        skipped = False
        flat_meta_path = session / "flat_calibration.json"
        if flat_meta_path.exists():
            try:
                flat_meta = json.loads(flat_meta_path.read_text(encoding="utf-8"))
                skipped = bool(flat_meta.get("skipped", False))
            except Exception:
                skipped = False
        return skipped

    def _read_geometry_meta_from_session(self, session: Path) -> tuple[int, bool]:
        ref = 0
        skipped = False
        geo_path = session / "geometry_calibration.json"
        if geo_path.exists():
            geo = json.loads(geo_path.read_text(encoding="utf-8"))
            try:
                ref = int(geo.get("reference_lens", 0))
            except Exception:
                ref = 0
            skipped = bool(geo.get("skipped", False))
        ref = max(0, min(15, int(ref)))
        return ref, skipped

    def _load_geometry_from_session(self, session: Path) -> None:
        ref, skipped = self._read_geometry_meta_from_session(session)
        self.reference_lens = ref
        if skipped:
            self.geometry_h = None
            self.geometry_correction_enabled = False
            self.geometry_progress_var.set(100.0)
            self.geometry_progress_text_var.set("Skipped")
            return
        h_path = session / "H_lens.npy"
        if not h_path.exists():
            raise RuntimeError("Missing file: H_lens.npy")
        self.geometry_h = np.load(h_path)
        self.geometry_correction_enabled = True
        self.geometry_progress_var.set(100.0)
        self.geometry_progress_text_var.set("Done")

    def _load_stage_files(self, stage: str) -> None:
        if self.session_root is None:
            self.status_var.set("No active calibration session")
            return
        source_root = filedialog.askdirectory(title=f"Select session folder for {stage} stage files")
        if not source_root:
            return
        source = Path(source_root)
        try:
            copied: list[str] = []
            if stage == "crop":
                if not (source / "crop.json").exists():
                    raise RuntimeError("Missing file: crop.json")
                self._load_crop_from_session(source)
                copied.extend(["crop.json"])
                self._show_calibration_stage("flat")
                self.status_var.set("Crop files loaded. Proceed to flat map.")
            elif stage == "dark":
                # Deprecated stage: keep for backward compatibility in UI callbacks.
                self.dark_map = None
                self.noise_map = None
                self.locked_gain = None
                self.locked_exposure = None
                self._show_calibration_stage("flat")
                self.status_var.set("Dark calibration stage is disabled. Proceed to flat map.")
            elif stage == "flat":
                flat_skipped = self._read_flat_meta_from_session(source)
                if (not flat_skipped) and (not (source / "flat_norm.npy").exists()):
                    raise RuntimeError("Missing file: flat_norm.npy")
                if not self.crop_boxes and (source / "crop.json").exists():
                    self._load_crop_from_session(source)
                    copied.append("crop.json")
                self._load_flat_from_session(source)
                copied.extend(["flat_norm.npy", "flat_raw_mean.npy", "flat_frames.npy", "flat_frames_minus_dark.npy", "flat_calibration.json"])
                self._show_calibration_stage("geometry")
                self.geometry_capture_idx = 0
                self.geometry_corners = []
                self.geometry_focus_scores = []
                self.geometry_btn_var.set(f"Capture chess frame 1/{max(1, int(self.geometry_target_var.get()))}")
                if self.flat_norm is None:
                    self.status_var.set("Flat stage loaded as skipped. Proceed to geometry calibration.")
                else:
                    self.status_var.set("Flat files loaded. Proceed to geometry calibration.")
            elif stage == "geometry":
                if not self.crop_boxes and (source / "crop.json").exists():
                    self._load_crop_from_session(source)
                    copied.append("crop.json")
                if self.flat_norm is None and ((source / "flat_norm.npy").exists() or (source / "flat_calibration.json").exists()):
                    self._load_flat_from_session(source)
                    copied.extend(["flat_norm.npy", "flat_raw_mean.npy", "flat_frames.npy", "flat_frames_minus_dark.npy", "flat_calibration.json"])
                if not self.crop_boxes:
                    raise RuntimeError("Crop files are required before geometry stage")
                _ref, geo_skipped = self._read_geometry_meta_from_session(source)
                if (not geo_skipped) and (not (source / "H_lens.npy").exists()):
                    raise RuntimeError("Missing file: H_lens.npy")
                self._load_geometry_from_session(source)
                copied.extend(["focus_scores.npy", "focus_scores.json", "geometry_calibration.json", "H_lens.npy"])
                self._show_page("main")
                if self.geometry_correction_enabled:
                    self.status_var.set("Geometry files loaded. Calibration complete.")
                else:
                    self.status_var.set("Geometry stage loaded as skipped. Calibration complete (no geometric correction).")
            else:
                raise RuntimeError(f"Unsupported stage: {stage}")

            if (source / "wavelength_mapping.json").exists():
                self._copy_files_to_current_session(source, ["wavelength_mapping.json"])
                self._load_wavelength_mapping_from_session(source)
            if (source / "white_point_reference.npy").exists():
                self._copy_files_to_current_session(source, ["white_point_reference.npy", "white_point.json"])
                self._load_white_point_from_session(source)
            self._copy_files_to_current_session(source, copied)
            self._render_preview(force=True)
        except Exception as exc:
            self.status_var.set(f"Load files failed: {exc}")

    def _load_existing_calibration(self) -> None:
        root = filedialog.askdirectory(title="Select calibration session folder")
        if not root:
            return
        session = Path(root)
        _, geo_skipped = self._read_geometry_meta_from_session(session)
        flat_skipped = self._read_flat_meta_from_session(session)
        required = [
            session / "crop.json",
        ]
        if not flat_skipped:
            required.append(session / "flat_norm.npy")
        if (session / "geometry_calibration.json").exists():
            required.append(session / "geometry_calibration.json")
        if not geo_skipped:
            required.append(session / "H_lens.npy")
        missing = [p.name for p in required if not p.exists()]
        if missing:
            self.status_var.set(f"Missing calibration files: {', '.join(missing)}")
            return
        try:
            self._clear_white_point_state()
            self._load_crop_from_session(session)
            self._load_flat_from_session(session)
            self._load_geometry_from_session(session)
            self._load_wavelength_mapping_from_session(session)
            self._load_white_point_from_session(session)
            self.session_root = session
            self._show_page("main")
            self.status_var.set(f"Loaded calibration from {session}")
        except Exception as exc:
            self.status_var.set(f"Failed to load calibration: {exc}")

    def _save_json(self, path: Path, payload: dict[str, object]) -> None:
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    # ----------------------------- Controls -----------------------------
    def _queue_command(self, cmd: str, value: object | None) -> None:
        if not (self.worker and self.worker.is_alive()):
            return
        try:
            self.cmd_q.put_nowait((cmd, value))
        except queue.Full:
            self.status_var.set("Command queue is full")

    def _on_gain_slide(self, _value: str) -> None:
        v = max(self.gain_bounds[0], min(self.gain_bounds[1], float(self.gain_var.get())))
        self.gain_var.set(v)
        self.gain_entry_var.set(f"{v:.3f}")
        self._update_max_fps_label()
        self._queue_command("set_gain", v)

    def _on_exposure_slide(self, _value: str) -> None:
        v = max(self.exposure_bounds[0], min(self.exposure_bounds[1], float(self.exposure_var.get())))
        self.exposure_var.set(v)
        self.exposure_entry_var.set(f"{v:.1f}")
        self._update_max_fps_label()
        self._queue_command("set_exposure", v)

    def _on_gain_entry_commit(self, _event: tk.Event) -> None:
        try:
            v = float(self.gain_entry_var.get().strip())
        except ValueError:
            return
        self.gain_var.set(max(self.gain_bounds[0], min(self.gain_bounds[1], v)))
        self._on_gain_slide("")

    def _on_exposure_entry_commit(self, _event: tk.Event) -> None:
        try:
            v = float(self.exposure_entry_var.get().strip())
        except ValueError:
            return
        self.exposure_var.set(max(self.exposure_bounds[0], min(self.exposure_bounds[1], v)))
        self._on_exposure_slide("")

    def _gain_control_allowed(self) -> bool:
        _ = self.current_page, self.calib_stage
        return True

    def _update_max_fps_label(self) -> None:
        exp = max(1.0, float(self.exposure_var.get()))
        fps = 1_000_000.0 / exp
        self.max_fps_var.set(f"Max FPS (exposure-only estimate): {fps:.2f}")

    # ----------------------------- RAW helpers -----------------------------
    def _decode_frame(self, frame: FramePacket) -> tuple["np.ndarray", str]:
        meta = frame.meta or {}
        fmt = meta.get("pixel_format_name") or meta.get("pixel_format") or frame.pixel_format
        fmt_name = pixel_format_to_name(fmt)
        arr = decode_buffer_to_ndarray(frame.raw, frame.width, frame.height, fmt)
        if self.force_u8_mode and arr.dtype != np.uint8:
            # Force 8-bit domain while preserving linear intensity values.
            arr = raw_to_u8(arr, fmt_name, autostretch=False)
        if self.debug:
            raw_len = len(frame.raw)
            self._dbg(
                f"decode: w={frame.width} h={frame.height} raw_bytes={raw_len} "
                f"fmt={fmt_name} out_dtype={arr.dtype} out_shape={tuple(arr.shape)}"
            )
        return arr, fmt_name

    def _get_latest_raw(self) -> tuple["np.ndarray", str] | None:
        if self.last_frame is None:
            return None
        try:
            return self._decode_frame(self.last_frame)
        except Exception as exc:
            self.status_var.set(f"Decode failed: {exc}")
            return None

    @staticmethod
    def _apply_crop_to_raw_boxes(raw_array: "np.ndarray", crop_boxes: list[dict[str, int]]) -> "np.ndarray | None":
        if not crop_boxes or len(crop_boxes) != 16:
            return None
        h_img = int(raw_array.shape[0])
        w_img = int(raw_array.shape[1])
        out: list[np.ndarray] = []
        for b in sorted(crop_boxes, key=lambda z: int(z["index"])):
            x = int(b["x"])
            y = int(b["y"])
            w = int(b["width"])
            h = int(b["height"])
            x0 = max(0, min(w_img - 1, x))
            y0 = max(0, min(h_img - 1, y))
            x1 = max(0, min(w_img, x + w))
            y1 = max(0, min(h_img, y + h))
            if x1 <= x0 or y1 <= y0:
                return None
            out.append(raw_array[y0:y1, x0:x1])
        if not out:
            return None
        min_h = min(int(a.shape[0]) for a in out)
        min_w = min(int(a.shape[1]) for a in out)
        if min_h < 2 or min_w < 2:
            return None
        if any(int(a.shape[0]) != min_h or int(a.shape[1]) != min_w for a in out):
            cropped: list[np.ndarray] = []
            for a in out:
                ah = int(a.shape[0])
                aw = int(a.shape[1])
                oy = max(0, (ah - min_h) // 2)
                ox = max(0, (aw - min_w) // 2)
                cropped.append(a[oy : oy + min_h, ox : ox + min_w])
            out = cropped
        try:
            return np.stack(out, axis=0)
        except Exception:
            return None

    def _apply_crop_to_raw(self, raw_array: "np.ndarray") -> "np.ndarray | None":
        return self._apply_crop_to_raw_boxes(raw_array, self.crop_boxes)

    def _apply_dark_flat(self, raw_lenses: "np.ndarray") -> "np.ndarray":
        work = raw_lenses.astype(np.float32)
        if self.flat_norm is not None and self.flat_norm.shape == raw_lenses.shape:
            safe = np.where(np.abs(self.flat_norm) < 1e-6, 1e-6, self.flat_norm)
            work = work / safe
        return work

    def _preview_rgb_menon(
        self,
        raw_array: "np.ndarray",
        fmt: str,
        autostretch: bool = False,
        pattern_override: str | None = None,
    ) -> "np.ndarray":
        return debayer_menon_rgb(
            raw_array,
            fmt,
            pattern_override=pattern_override,
            autostretch=autostretch,
        )

    def _preview_rgb_main_fast(
        self,
        raw_array: "np.ndarray",
        fmt: str,
        autostretch: bool = False,
        pattern_override: str | None = None,
    ) -> "np.ndarray":
        return debayer_fast_preview_rgb(
            raw_array,
            fmt,
            pattern_override=pattern_override,
            autostretch=autostretch,
        )

    def _sorted_crop_boxes(self) -> list[dict[str, int]]:
        if not self.crop_boxes:
            return []
        return sorted(self.crop_boxes, key=lambda z: int(z["index"]))

    # ----------------------------- Crop stage -----------------------------
    def _equal_bands(self, length: int, expected: int) -> list[tuple[int, int]]:
        bands: list[tuple[int, int]] = []
        step = max(1, length // expected)
        for i in range(expected):
            a = i * step
            b = length if i == expected - 1 else (i + 1) * step
            if b <= a:
                b = min(length, a + 1)
            bands.append((a, b))
        return bands

    def _find_bands_from_profile(self, profile: "np.ndarray", expected: int, min_len: int) -> list[tuple[int, int]]:
        if profile.size < expected:
            return self._equal_bands(int(profile.size), expected)
        smooth = np.asarray(profile, dtype=np.float32)
        k = max(7, int(round(profile.size * 0.015)))
        if (k % 2) == 0:
            k += 1
        kernel = np.ones((k,), dtype=np.float32) / float(k)
        smooth = np.convolve(smooth, kernel, mode="same")
        lo = float(np.min(smooth))
        hi = float(np.max(smooth))
        if hi <= lo + 1e-6:
            return self._equal_bands(int(profile.size), expected)
        thr = max(float(np.percentile(smooth, 55.0)), lo + 0.35 * (hi - lo))
        mask = (smooth >= thr).astype(np.uint8)
        close_k = max(5, int(round(profile.size * 0.01)))
        close_kernel = np.ones((close_k,), dtype=np.uint8)
        mask = (np.convolve(mask, close_kernel, mode="same") >= max(1, close_k // 2)).astype(np.uint8)

        runs: list[tuple[int, int, float]] = []
        i = 0
        n = int(mask.size)
        while i < n:
            if mask[i] == 0:
                i += 1
                continue
            s = i
            while i < n and mask[i] == 1:
                i += 1
            e = i
            if (e - s) >= min_len:
                runs.append((s, e, float(np.mean(smooth[s:e]))))

        if len(runs) < expected:
            return self._equal_bands(int(profile.size), expected)
        if len(runs) > expected:
            runs.sort(key=lambda x: (x[2], x[1] - x[0]), reverse=True)
            runs = runs[:expected]
        runs.sort(key=lambda x: x[0])
        bands = [(int(s), int(e)) for s, e, _m in runs]
        # Validate spacing/widths: reject pathological split (e.g. one thin top strip + duplicated first row).
        widths = np.asarray([max(1, b[1] - b[0]) for b in bands], dtype=np.float32)
        med_w = float(np.median(widths)) if widths.size else 1.0
        if med_w <= 1.0:
            return self._equal_bands(int(profile.size), expected)
        if float(np.min(widths)) < max(float(min_len), 0.60 * med_w) or float(np.max(widths)) > 1.95 * med_w:
            return self._equal_bands(int(profile.size), expected)
        centers = np.asarray([0.5 * (b[0] + b[1]) for b in bands], dtype=np.float32)
        diffs = np.diff(centers)
        if diffs.size:
            med_d = float(np.median(diffs))
            if med_d <= 1.0 or float(np.min(diffs)) < 0.50 * med_d or float(np.max(diffs)) > 1.85 * med_d:
                return self._equal_bands(int(profile.size), expected)
        return bands

    def _detect_peak_in_roi(self, roi_u8: "np.ndarray") -> tuple[float, float, float]:
        h, w = int(roi_u8.shape[0]), int(roi_u8.shape[1])
        if h <= 0 or w <= 0:
            return 0.0, 0.0, 0.0
        patch = roi_u8
        blur = cv2.GaussianBlur(patch, (0, 0), 3.0)
        hp = np.clip(patch.astype(np.float32) - blur.astype(np.float32), 0.0, None)
        hp_max = float(np.max(hp))
        if hp_max <= 1e-6:
            _mn0, max_v0, _mn_loc0, max_loc0 = cv2.minMaxLoc(patch)
            return float(max_loc0[0]), float(max_loc0[1]), float(max_v0)

        hp_u8 = np.clip((hp / hp_max) * 255.0, 0.0, 255.0).astype(np.uint8)
        thr = float(np.percentile(hp_u8, 99.6))
        thr = max(8.0, thr)
        _ret, bw = cv2.threshold(hp_u8, thr, 255, cv2.THRESH_BINARY)
        k = np.ones((3, 3), dtype=np.uint8)
        bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, k, iterations=1)

        cc = cv2.connectedComponentsWithStats(bw, connectivity=8)
        n_labels, labels, stats, centroids = cc
        roi_area = max(1, h * w)
        best_idx = -1
        best_score = -1e9
        best_conf = 0.0
        for i in range(1, int(n_labels)):
            area = int(stats[i, cv2.CC_STAT_AREA])
            if area < 1 or area > int(roi_area * 0.06):
                continue
            mask_i = labels == i
            local_max = float(np.max(hp[mask_i])) if np.any(mask_i) else 0.0
            score = local_max - 0.03 * float(area)
            if score > best_score:
                best_score = score
                best_idx = i
                best_conf = local_max

        if best_idx >= 0:
            cx, cy = centroids[best_idx]
            return float(cx), float(cy), float(best_conf)

        _mn, max_v, _mn_loc, max_loc = cv2.minMaxLoc(hp)
        if float(max_v) < 0.5:
            _mn2, max_v2, _mn_loc2, max_loc2 = cv2.minMaxLoc(patch)
            return float(max_loc2[0]), float(max_loc2[1]), float(max_v2)
        return float(max_loc[0]), float(max_loc[1]), float(max_v)

    def _find_chessboard_corners_crop(self, roi_u8: "np.ndarray", cols: int, rows: int) -> "np.ndarray | None":
        # Crop-stage detector: tuned for 16 boards in one full frame (one board per lens ROI).
        # This path intentionally does NOT reuse geometry detector to avoid cross-stage coupling.
        if cv2 is None:
            return None
        h, w = int(roi_u8.shape[0]), int(roi_u8.shape[1])
        if h < 14 or w < 14:
            return None

        candidates: list[np.ndarray] = [roi_u8]
        try:
            clahe_obj = cv2.createCLAHE(clipLimit=2.4, tileGridSize=(8, 8))
            clahe = clahe_obj.apply(roi_u8)
            candidates.append(clahe)
            blur = cv2.GaussianBlur(clahe, (0, 0), 1.0)
            sharp = cv2.addWeighted(clahe, 1.55, blur, -0.55, 0)
            candidates.append(sharp)
        except Exception:
            pass
        candidates.extend([cv2.bitwise_not(img) for img in list(candidates)])

        scales: list[float] = [1.0]
        if min(h, w) < 260:
            scales.append(2.0)

        best: np.ndarray | None = None
        best_score = -1e9
        for sc in scales:
            for img in candidates:
                if abs(sc - 1.0) > 1e-6:
                    ww = max(8, int(round(w * sc)))
                    hh = max(8, int(round(h * sc)))
                    probe = cv2.resize(img, (ww, hh), interpolation=cv2.INTER_LINEAR)
                    scale_back = 1.0 / sc
                else:
                    probe = img
                    scale_back = 1.0

                flags = int(getattr(cv2, "CALIB_CB_ADAPTIVE_THRESH", 0))
                flags |= int(getattr(cv2, "CALIB_CB_NORMALIZE_IMAGE", 0))
                flags |= int(getattr(cv2, "CALIB_CB_FILTER_QUADS", 0))
                try:
                    found, corners = cv2.findChessboardCorners(probe, (cols, rows), flags)
                except Exception:
                    found, corners = (False, None)
                if not found or corners is None or int(corners.shape[0]) != (cols * rows):
                    continue

                crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 35, 0.001)
                try:
                    corners = cv2.cornerSubPix(probe, corners, (5, 5), (-1, -1), crit)
                except Exception:
                    pass
                pts = corners.reshape(-1, 2).astype(np.float32)
                if abs(scale_back - 1.0) > 1e-6:
                    pts *= float(scale_back)

                cidx = (rows // 2) * cols + (cols // 2)
                cx = float(pts[cidx, 0])
                cy = float(pts[cidx, 1])
                # Crop-stage prior: board center should stay reasonably central in each lens ROI.
                if cx < 0.10 * w or cx > 0.90 * w or cy < 0.10 * h or cy > 0.90 * h:
                    continue
                spread = float(np.mean(np.linalg.norm(pts - pts[cidx], axis=1)))
                dx = cx - 0.5 * w
                dy = cy - 0.5 * h
                score = spread - 0.010 * float(dx * dx + dy * dy)
                if score > best_score:
                    best_score = score
                    best = pts
        return best

    def _detect_chess_center_in_roi(self, roi_u8: "np.ndarray") -> tuple[float, float, float] | None:
        # 4x4 chessboard cells => 3x3 inner corners, center is index 4.
        cols = 3
        rows = 3
        corners = self._find_chessboard_corners_crop(roi_u8, cols, rows)
        if corners is None or corners.shape[0] != (cols * rows):
            return None
        cidx = (rows // 2) * cols + (cols // 2)
        cx = float(corners[cidx, 0])
        cy = float(corners[cidx, 1])
        spread = float(np.mean(np.linalg.norm(corners - corners[cidx], axis=1)))
        return (cx, cy, spread)

    def _fit_center_grid_from_found(self, found: dict[int, tuple[float, float, float]]) -> dict[int, tuple[float, float, float]]:
        # Model center positions as bilinear function of (row, col) on the 4x4 lens lattice.
        # Bilinear term (c*r) better handles mild lattice warping than plain affine.
        if len(found) < 4:
            return {}
        rows = []
        xvals = []
        yvals = []
        for idx, (x, y, _c) in found.items():
            r = float(idx // 4)
            c = float(idx % 4)
            rows.append([1.0, c, r, c * r])
            xvals.append(float(x))
            yvals.append(float(y))
        A = np.asarray(rows, dtype=np.float32)
        bx = np.asarray(xvals, dtype=np.float32)
        by = np.asarray(yvals, dtype=np.float32)
        try:
            px, _resx, _rankx, _sx = np.linalg.lstsq(A, bx, rcond=None)
            py, _resy, _ranky, _sy = np.linalg.lstsq(A, by, rcond=None)
        except Exception:
            return {}
        out: dict[int, tuple[float, float, float]] = {}
        for idx in range(16):
            if idx in found:
                continue
            r = float(idx // 4)
            c = float(idx % 4)
            vx = float(px[0] + px[1] * c + px[2] * r + px[3] * c * r)
            vy = float(py[0] + py[1] * c + py[2] * r + py[3] * c * r)
            out[idx] = (vx, vy, 0.0)
        return out

    def _refine_center_near_hint(self, roi_u8: "np.ndarray", x_hint: float, y_hint: float, radius: int = 26) -> tuple[float, float, float]:
        if cv2 is None:
            return x_hint, y_hint, 0.0
        h, w = int(roi_u8.shape[0]), int(roi_u8.shape[1])
        if h <= 4 or w <= 4:
            return x_hint, y_hint, 0.0
        xh = int(max(0, min(w - 1, round(float(x_hint)))))
        yh = int(max(0, min(h - 1, round(float(y_hint)))))
        x0 = max(0, xh - radius)
        y0 = max(0, yh - radius)
        x1 = min(w, xh + radius + 1)
        y1 = min(h, yh + radius + 1)
        if x1 <= x0 + 2 or y1 <= y0 + 2:
            return float(xh), float(yh), 0.0
        patch = roi_u8[y0:y1, x0:x1]
        # Prefer true chess center via local robust call around hint.
        got = self._detect_chess_center_in_roi(patch)
        if got is not None:
            gx, gy, conf = got
            return float(x0 + gx), float(y0 + gy), float(conf)
        # If local chessboard lock failed, keep geometric hint (avoid drifting to unrelated bright structures).
        return float(xh), float(yh), 0.0

    def _detect_crop_points(self) -> None:
        if cv2 is None:
            self.status_var.set("OpenCV is required for chess-center detection")
            return
        got = self._get_latest_raw()
        if got is None:
            self.status_var.set("No frame for crop detection")
            return
        raw_arr, fmt = got
        gray_det = raw_to_u8(raw_arr, fmt, autostretch=True)
        h, w = int(gray_det.shape[0]), int(gray_det.shape[1])
        min_len_x = max(30, w // 20)
        min_len_y = max(30, h // 20)
        col_bands = self._find_bands_from_profile(gray_det.mean(axis=0), expected=4, min_len=min_len_x)
        row_bands = self._find_bands_from_profile(gray_det.mean(axis=1), expected=4, min_len=min_len_y)
        pts: list[tuple[float, float, float]] = []
        roi_boxes: list[tuple[int, int, int, int]] = []
        found_map: dict[int, tuple[float, float, float]] = {}
        missing: list[int] = []
        lens_idx = 0
        for r, (y0, y1) in enumerate(row_bands):
            for c, (x0, x1) in enumerate(col_bands):
                x0c = max(0, min(w - 1, int(x0)))
                x1c = max(x0c + 1, min(w, int(x1)))
                y0c = max(0, min(h - 1, int(y0)))
                y1c = max(y0c + 1, min(h, int(y1)))
                roi_boxes.append((x0c, y0c, x1c, y1c))
                roi = gray_det[y0c:y1c, x0c:x1c]
                got_center = self._detect_chess_center_in_roi(roi)
                if got_center is None:
                    missing.append(lens_idx + 1)
                    # Fallback keeps array shape consistent, but user is warned and save is blocked.
                    px, py, conf = self._detect_peak_in_roi(roi)
                    pts.append((x0c + px, y0c + py, conf))
                else:
                    px, py, conf = got_center
                    gx = x0c + px
                    gy = y0c + py
                    pts.append((gx, gy, conf))
                    found_map[lens_idx] = (gx, gy, conf)
                lens_idx += 1
        if self.debug:
            confs = np.asarray([p[2] for p in pts], dtype=np.float32)
            self._dbg(
                f"crop detect grid: row_bands={row_bands} col_bands={col_bands} "
                f"gray_min={int(gray_det.min())} gray_max={int(gray_det.max())} "
                f"conf(min/mean/max)={float(np.min(confs)):.2f}/{float(np.mean(confs)):.2f}/{float(np.max(confs)):.2f}"
            )
        if len(pts) != 16:
            self.status_var.set(f"Internal detection error: got {len(pts)} points instead of 16")
            return
        # Recovery pass: estimate missing centers from 4x4 lattice + local ROI refinement.
        recovered_count = 0
        if missing:
            preds = self._fit_center_grid_from_found(found_map)
            if self.crop_centers is not None and len(self.crop_centers) == 16:
                for midx in [m - 1 for m in missing]:
                    if midx not in preds:
                        preds[midx] = (float(self.crop_centers[midx, 0]), float(self.crop_centers[midx, 1]), 0.0)
            recovered: list[int] = []
            for midx1 in list(missing):
                midx = midx1 - 1
                if midx < 0 or midx >= 16 or midx not in preds:
                    continue
                x0c, y0c, x1c, y1c = roi_boxes[midx]
                roi = gray_det[y0c:y1c, x0c:x1c]
                px_hint = float(preds[midx][0] - x0c)
                py_hint = float(preds[midx][1] - y0c)
                px_hint = max(0.0, min(float(x1c - x0c - 1), px_hint))
                py_hint = max(0.0, min(float(y1c - y0c - 1), py_hint))
                px, py, conf = self._refine_center_near_hint(roi, px_hint, py_hint, radius=28)
                gx = float(x0c + px)
                gy = float(y0c + py)
                pts[midx] = (gx, gy, conf)
                found_map[midx] = (gx, gy, conf)
                recovered.append(midx1)
            if recovered:
                recovered_count = len(recovered)
                missing = [m for m in missing if m not in recovered]
                if self.debug:
                    self._dbg(f"crop detect recovery: recovered={recovered} remaining_missing={missing}")
        if missing:
            self.status_var.set(
                f"Chessboard center not found on lenses: {', '.join(str(i) for i in missing)}. "
                "Use 4x4 chessboard and keep it visible on all 16 lenses."
            )
            return
        ordered = np.asarray([[p[0], p[1]] for p in pts], dtype=np.float32)
        conf_mean = float(np.mean(np.asarray([p[2] for p in pts], dtype=np.float32)))
        self.crop_centers = ordered
        self._recompute_crop_limits(raw_arr.shape[1], raw_arr.shape[0])
        self._on_crop_offsets_changed()
        if recovered_count > 0:
            self.status_var.set(
                f"Detected 16 chess centers (recovered {recovered_count} by grid model). "
                f"Mean confidence={conf_mean:.2f}. Adjust offsets and save crop."
            )
        else:
            self.status_var.set(f"Detected 16 chess centers. Mean confidence={conf_mean:.2f}. Adjust offsets and save crop.")

    def _recompute_crop_limits(self, w: int, h: int) -> None:
        if self.crop_centers is None:
            return
        centers = self.crop_centers.reshape(4, 4, 2)
        left_vals: list[int] = []
        right_vals: list[int] = []
        top_vals: list[int] = []
        bottom_vals: list[int] = []
        for r in range(4):
            for c in range(4):
                cx = float(centers[r, c, 0])
                cy = float(centers[r, c, 1])
                left_bound = 0.0 if c == 0 else 0.5 * (cx + float(centers[r, c - 1, 0]))
                right_bound = (w - 1) if c == 3 else 0.5 * (cx + float(centers[r, c + 1, 0]))
                top_bound = 0.0 if r == 0 else 0.5 * (cy + float(centers[r - 1, c, 1]))
                bottom_bound = (h - 1) if r == 3 else 0.5 * (cy + float(centers[r + 1, c, 1]))
                left_vals.append(max(1, int(math.floor(cx - left_bound))))
                right_vals.append(max(1, int(math.floor(right_bound - cx))))
                top_vals.append(max(1, int(math.floor(cy - top_bound))))
                bottom_vals.append(max(1, int(math.floor(bottom_bound - cy))))
        self.crop_limits = {
            "left": max(1, min(left_vals)),
            "right": max(1, min(right_vals)),
            "top": max(1, min(top_vals)),
            "bottom": max(1, min(bottom_vals)),
        }

    def _on_crop_offsets_changed(self) -> None:
        if self.crop_centers is None or self.last_frame is None:
            return
        w = int(self.last_frame.width)
        h = int(self.last_frame.height)
        self._recompute_crop_limits(w, h)
        left = max(1, min(int(self.crop_left_var.get()), self.crop_limits["left"]))
        right = max(1, min(int(self.crop_right_var.get()), self.crop_limits["right"]))
        top = max(1, min(int(self.crop_top_var.get()), self.crop_limits["top"]))
        bottom = max(1, min(int(self.crop_bottom_var.get()), self.crop_limits["bottom"]))
        self.crop_left_var.set(left)
        self.crop_right_var.set(right)
        self.crop_top_var.set(top)
        self.crop_bottom_var.set(bottom)

        centers = self.crop_centers.reshape(4, 4, 2)
        boxes: list[dict[str, int]] = []
        idx = 0
        for r in range(4):
            for c in range(4):
                cx = float(centers[r, c, 0])
                cy = float(centers[r, c, 1])
                left_bound = 0.0 if c == 0 else 0.5 * (cx + float(centers[r, c - 1, 0]))
                right_bound = (w - 1) if c == 3 else 0.5 * (cx + float(centers[r, c + 1, 0]))
                top_bound = 0.0 if r == 0 else 0.5 * (cy + float(centers[r - 1, c, 1]))
                bottom_bound = (h - 1) if r == 3 else 0.5 * (cy + float(centers[r + 1, c, 1]))

                x0 = int(round(max(left_bound, cx - left)))
                x1 = int(round(min(right_bound, cx + right)))
                y0 = int(round(max(top_bound, cy - top)))
                y1 = int(round(min(bottom_bound, cy + bottom)))
                if x1 <= x0:
                    x1 = x0 + 1
                if y1 <= y0:
                    y1 = y0 + 1
                boxes.append(
                    {
                        "index": idx,
                        "row": r,
                        "col": c,
                        "x": x0,
                        "y": y0,
                        "width": x1 - x0 + 1,
                        "height": y1 - y0 + 1,
                    }
                )
                idx += 1
        self.crop_boxes = boxes
        self._render_preview(force=True)

    def _save_crop_and_continue(self) -> None:
        if self.session_root is None:
            self.status_var.set("No active calibration session")
            return
        if self.crop_centers is None or len(self.crop_boxes) != 16:
            self.status_var.set("Detect points and adjust crop first")
            return
        payload: dict[str, object] = {
            "stage": "crop_calibration",
            "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            "image_width": int(self.last_frame.width if self.last_frame else 0),
            "image_height": int(self.last_frame.height if self.last_frame else 0),
            "order": "top_to_bottom_left_to_right",
            "centers_xy": self.crop_centers.tolist(),
            "offsets": {
                "left": int(self.crop_left_var.get()),
                "right": int(self.crop_right_var.get()),
                "top": int(self.crop_top_var.get()),
                "bottom": int(self.crop_bottom_var.get()),
            },
            "boxes": self.crop_boxes,
        }
        self._save_json(self.session_root / "crop.json", payload)
        self._show_calibration_stage("flat")
        self.status_var.set("Crop calibration saved. Proceed to flat map.")

    # ----------------------------- Dark stage -----------------------------
    def _start_dark_capture(self) -> None:
        if self.crop_boxes is None or len(self.crop_boxes) != 16:
            self.status_var.set("Crop calibration is required first")
            return
        if self.dark_capture_active:
            self.status_var.set("Dark capture already running")
            return
        try:
            target = max(1, int(self.dark_burst_var.get()))
        except Exception:
            self.status_var.set("Invalid burst count")
            return
        self._queue_command("set_gain", float(self.gain_var.get()))
        self._queue_command("set_exposure", float(self.exposure_var.get()))
        self.dark_frames = []
        self.frame_dedupe_id = None
        self.dark_capture_active = True
        self._dark_target = target
        self.status_var.set(f"Collecting dark burst: 0/{target}")

    def _consume_dark_frame(self, frame: FramePacket) -> None:
        if not self.dark_capture_active:
            return
        frame_id = int((frame.meta or {}).get("frame_id", -1))
        if frame_id >= 0 and frame_id == self.frame_dedupe_id:
            return
        self.frame_dedupe_id = frame_id
        try:
            raw_arr, _fmt = self._decode_frame(frame)
            lenses = self._apply_crop_to_raw(raw_arr)
            if lenses is None:
                self.dark_capture_active = False
                self.status_var.set("Dark map failed: invalid crop ROIs. Re-detect crop centers or load crop files.")
                return
            self.dark_frames.append(lenses.copy())
            count = len(self.dark_frames)
            self.status_var.set(f"Collecting dark burst: {count}/{self._dark_target}")
            if count < self._dark_target:
                return
            stack = np.stack(self.dark_frames, axis=0).astype(np.float32)
            mean_dark = stack.mean(axis=0)
            std_dark = stack.std(axis=0)
            if lenses.dtype == np.uint8:
                self.dark_map = np.clip(np.rint(mean_dark), 0, 255).astype(np.uint8)
            elif lenses.dtype == np.uint16:
                self.dark_map = np.clip(np.rint(mean_dark), 0, 65535).astype(np.uint16)
            else:
                self.dark_map = mean_dark.astype(np.float32)
            self.noise_map = std_dark.astype(np.float32)
            self.dark_capture_active = False
            self.locked_gain = float(self.gain_var.get())
            self.locked_exposure = float(self.exposure_var.get())
            if self.session_root is not None:
                np.save(self.session_root / "darkMap.npy", self.dark_map)
                np.save(self.session_root / "noiseMap.npy", self.noise_map)
                np.save(self.session_root / "dark_frames.npy", stack)
                self._save_json(
                    self.session_root / "camera_settings.json",
                    {
                        "exposure_us": self.locked_exposure,
                        "gain_db": self.locked_gain,
                        "locked_after_stage": "black_level",
                    },
                )
            self._show_calibration_stage("flat")
            self.status_var.set("Dark map created. Exposure/Gain locked for this session.")
        except Exception as exc:
            self.dark_capture_active = False
            self.status_var.set(f"Dark map failed: {exc}")

    # ----------------------------- Flat stage -----------------------------
    def _start_flat_capture(self) -> None:
        if self.flat_capture_active:
            self.status_var.set("Flat capture already running")
            return
        try:
            target = max(1, int(self.flat_burst_var.get()))
            sigma = max(0.1, float(self.flat_sigma_var.get()))
        except Exception:
            self.status_var.set("Invalid flat settings")
            return
        self.flat_frames = []
        self.frame_dedupe_id = None
        self.flat_capture_active = True
        self._flat_target = target
        self._flat_sigma = sigma
        self.status_var.set(f"Collecting flat burst: 0/{target}")

    def _consume_flat_frame(self, frame: FramePacket) -> None:
        if not self.flat_capture_active:
            return
        frame_id = int((frame.meta or {}).get("frame_id", -1))
        if frame_id >= 0 and frame_id == self.frame_dedupe_id:
            return
        self.frame_dedupe_id = frame_id
        try:
            raw_arr, _fmt = self._decode_frame(frame)
            lenses = self._apply_crop_to_raw(raw_arr)
            if lenses is None:
                self.flat_capture_active = False
                self.status_var.set("Flat map failed: invalid crop ROIs. Re-detect crop centers or load crop files.")
                return
            work = lenses.astype(np.float32)
            self.flat_frames.append(work.copy())
            count = len(self.flat_frames)
            self.status_var.set(f"Collecting flat burst: {count}/{self._flat_target}")
            if count < self._flat_target:
                return
            stack = np.stack(self.flat_frames, axis=0).astype(np.float32)
            flat_mean = stack.mean(axis=0)
            sigma = float(self._flat_sigma)
            if gaussian_filter is not None:
                flat_smooth = np.empty_like(flat_mean, dtype=np.float32)
                for i in range(flat_mean.shape[0]):
                    flat_smooth[i] = gaussian_filter(flat_mean[i], sigma=sigma)
            else:
                flat_smooth = flat_mean
            flat_norm = np.empty_like(flat_smooth, dtype=np.float32)
            for i in range(flat_smooth.shape[0]):
                m = float(np.mean(flat_smooth[i]))
                if abs(m) < 1e-6:
                    m = 1.0
                flat_norm[i] = flat_smooth[i] / m
            self.flat_raw_mean = flat_mean
            self.flat_norm = flat_norm
            self.flat_capture_active = False
            if self.session_root is not None:
                np.save(self.session_root / "flat_raw_mean.npy", self.flat_raw_mean.astype(np.float32))
                np.save(self.session_root / "flat_norm.npy", self.flat_norm.astype(np.float32))
                np.save(self.session_root / "flat_frames.npy", stack)
                # Compatibility with older sessions/loaders that expect this filename.
                np.save(self.session_root / "flat_frames_minus_dark.npy", stack)
                self._save_json(
                    self.session_root / "flat_calibration.json",
                    {
                        "stage": "flat_field_calibration",
                        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
                        "skipped": False,
                        "sigma": sigma,
                        "burst_count": int(stack.shape[0]),
                        "shape": list(flat_norm.shape),
                        "normalization": "per_lens_mean",
                        "dark_subtraction_applied": False,
                    },
                )
            self._show_calibration_stage("geometry")
            self.geometry_capture_idx = 0
            self.geometry_corners = []
            self.geometry_focus_scores = []
            self.geometry_btn_var.set(f"Capture chess frame 1/{max(1, int(self.geometry_target_var.get()))}")
            self.status_var.set("Flat map created. Proceed to geometry calibration.")
        except Exception as exc:
            self.flat_capture_active = False
            self.status_var.set(f"Flat map failed: {exc}")

    def _skip_flat_calibration(self) -> None:
        self.flat_capture_active = False
        self.flat_frames = []
        self.flat_norm = None
        self.flat_raw_mean = None
        if self.session_root is not None:
            self._save_json(
                self.session_root / "flat_calibration.json",
                {
                    "stage": "flat_field_calibration",
                    "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
                    "skipped": True,
                    "note": "flat_correction_disabled",
                },
            )
        self._show_calibration_stage("geometry")
        self.geometry_capture_idx = 0
        self.geometry_corners = []
        self.geometry_focus_scores = []
        self.geometry_btn_var.set(f"Capture chess frame 1/{max(1, int(self.geometry_target_var.get()))}")
        self.status_var.set("Flat field calibration skipped. Proceed to geometry calibration.")

    # ----------------------------- Geometry stage -----------------------------
    def _find_chessboard_corners_robust(
        self,
        u8: "np.ndarray",
        cols: int,
        rows: int,
        progress_cb=None,
    ) -> tuple["np.ndarray | None", str]:
        if cv2 is None:
            return None, "opencv_unavailable"
        h, w = int(u8.shape[0]), int(u8.shape[1])
        if h < 12 or w < 12:
            return None, "too_small"

        def _roi_candidates(img_u8: "np.ndarray") -> list[tuple["np.ndarray", int, int]]:
            rois: list[tuple["np.ndarray", int, int]] = [(img_u8, 0, 0)]
            hh, ww = int(img_u8.shape[0]), int(img_u8.shape[1])
            if hh < 20 or ww < 20:
                return rois
            # Geometry-stage prior: board is usually near the lens center.
            try:
                cx0 = int(round(ww * 0.12))
                cy0 = int(round(hh * 0.10))
                cx1 = int(round(ww * 0.88))
                cy1 = int(round(hh * 0.90))
                if (cx1 - cx0) >= 16 and (cy1 - cy0) >= 16:
                    rois.append((img_u8[cy0:cy1, cx0:cx1], cx0, cy0))
            except Exception:
                pass
            try:
                blur = cv2.GaussianBlur(img_u8, (5, 5), 0.0)
                p = float(np.percentile(blur, 62.0))
                thr = max(8.0, min(245.0, p))
                _ret, bw = cv2.threshold(blur, thr, 255, cv2.THRESH_BINARY)
                k = np.ones((5, 5), dtype=np.uint8)
                bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, k, iterations=1)
                bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, k, iterations=1)
                n, _labels, stats, _cent = cv2.connectedComponentsWithStats(bw, connectivity=8)
                items: list[tuple[float, int, int, int, int]] = []
                area_all = max(1, hh * ww)
                for i in range(1, int(n)):
                    x = int(stats[i, cv2.CC_STAT_LEFT])
                    y = int(stats[i, cv2.CC_STAT_TOP])
                    cw = int(stats[i, cv2.CC_STAT_WIDTH])
                    ch = int(stats[i, cv2.CC_STAT_HEIGHT])
                    area = int(stats[i, cv2.CC_STAT_AREA])
                    if area < int(0.005 * area_all) or area > int(0.92 * area_all):
                        continue
                    if cw < 10 or ch < 10:
                        continue
                    aspect = float(cw) / float(max(1, ch))
                    if aspect < 0.30 or aspect > 3.3:
                        continue
                    squareness = 1.0 - min(1.0, abs(aspect - 1.0))
                    score = float(area) * (0.6 + 0.4 * squareness)
                    items.append((score, x, y, cw, ch))
                items.sort(key=lambda t: t[0], reverse=True)
                for _score, x, y, cw, ch in items[:5]:
                    padx = max(8, int(round(cw * 0.28)))
                    pady = max(8, int(round(ch * 0.28)))
                    x0 = max(0, x - padx)
                    y0 = max(0, y - pady)
                    x1 = min(ww, x + cw + padx)
                    y1 = min(hh, y + ch + pady)
                    if (x1 - x0) < 14 or (y1 - y0) < 14:
                        continue
                    rois.append((img_u8[y0:y1, x0:x1], x0, y0))
            except Exception:
                pass
            return rois

        def _try_one(img_u8: "np.ndarray", scale_back: float, ox: int = 0, oy: int = 0, method_prefix: str = "") -> tuple["np.ndarray | None", str]:
            corners = None
            found = False
            detector_used = "none"
            # Classic CPU detector first for stability on macOS; try several flag sets.
            flag_sets: list[tuple[str, int]] = []
            f0 = int(getattr(cv2, "CALIB_CB_ADAPTIVE_THRESH", 0)) | int(getattr(cv2, "CALIB_CB_NORMALIZE_IMAGE", 0))
            f1 = f0 | int(getattr(cv2, "CALIB_CB_FILTER_QUADS", 0))
            f2 = int(getattr(cv2, "CALIB_CB_NORMALIZE_IMAGE", 0))
            for name, fs in (("classic_adapt_norm", f0), ("classic_adapt_norm_quads", f1), ("classic_norm", f2)):
                if all(fs != x[1] for x in flag_sets):
                    flag_sets.append((name, fs))
            for fs_name, fs in flag_sets:
                if progress_cb is not None:
                    try:
                        progress_cb(f"{method_prefix}:classic[{fs_name}]")
                    except Exception:
                        pass
                try:
                    found, corners = cv2.findChessboardCorners(img_u8, (cols, rows), fs)
                except Exception:
                    found, corners = (False, None)
                if found and corners is not None:
                    detector_used = fs_name
                    break
            if found and corners is not None:
                crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 35, 0.001)
                corners = cv2.cornerSubPix(img_u8, corners, (5, 5), (-1, -1), crit)

            # Optional fallback to SB only if explicitly enabled by env.
            if (not found) and hasattr(cv2, "findChessboardCornersSB") and os.environ.get("HYDRA_ENABLE_SB", "0") == "1":
                sb_flags = 0
                for name in ("CALIB_CB_NORMALIZE_IMAGE", "CALIB_CB_EXHAUSTIVE", "CALIB_CB_ACCURACY"):
                    sb_flags |= int(getattr(cv2, name, 0))
                if progress_cb is not None:
                    try:
                        progress_cb(f"{method_prefix}:sb")
                    except Exception:
                        pass
                try:
                    found, corners = cv2.findChessboardCornersSB(img_u8, (cols, rows), sb_flags)
                    if found and corners is not None:
                        detector_used = "sb"
                except Exception:
                    found, corners = (False, None)
            if not found or corners is None or int(corners.shape[0]) != (cols * rows):
                return None, f"{method_prefix}:none"
            out = corners.reshape(-1, 2).astype(np.float32)
            if abs(scale_back - 1.0) > 1e-6:
                out *= float(scale_back)
            if ox or oy:
                out[:, 0] += float(ox)
                out[:, 1] += float(oy)
            return out, f"{method_prefix}:{detector_used}"

        # Candidate inputs: raw, equalized, sharpened, inverted variants.
        base = u8
        norm = None
        eq = None
        clahe = None
        sharp = None
        sharp2 = None
        adapt = None
        adapt_inv = None
        otsu = None
        otsu_inv = None
        try:
            norm = cv2.normalize(base, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            eq = cv2.equalizeHist(base)
            clahe_obj = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            clahe = clahe_obj.apply(base)
            blur = cv2.GaussianBlur(clahe, (0, 0), 1.2)
            sharp = cv2.addWeighted(clahe, 1.6, blur, -0.6, 0)
            blur2 = cv2.GaussianBlur(clahe, (0, 0), 2.1)
            sharp2 = cv2.addWeighted(clahe, 1.9, blur2, -0.9, 0)
            adapt = cv2.adaptiveThreshold(
                clahe,
                255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY,
                31,
                3,
            )
            adapt_inv = cv2.adaptiveThreshold(
                clahe,
                255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY_INV,
                31,
                3,
            )
            _ret, otsu = cv2.threshold(clahe, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            _ret2, otsu_inv = cv2.threshold(clahe, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        except Exception:
            pass
        candidates: list[tuple[str, np.ndarray]] = [("base", base)]
        if norm is not None:
            candidates.append(("norm", norm))
        if eq is not None:
            candidates.append(("eq", eq))
        if clahe is not None:
            candidates.append(("clahe", clahe))
        if sharp is not None:
            candidates.append(("sharp", sharp))
        if sharp2 is not None:
            candidates.append(("sharp2", sharp2))
        if adapt is not None:
            candidates.append(("adapt", adapt))
        if adapt_inv is not None:
            candidates.append(("adapt_inv", adapt_inv))
        if otsu is not None:
            candidates.append(("otsu", otsu))
        if otsu_inv is not None:
            candidates.append(("otsu_inv", otsu_inv))
        inv_base = [(name, img) for name, img in (("base", base), ("norm", norm), ("eq", eq), ("clahe", clahe), ("sharp", sharp), ("sharp2", sharp2)) if img is not None]
        for name, img in inv_base:
            candidates.append((f"{name}_inv", cv2.bitwise_not(img)))

        # If board is small in ROI, upscaling often helps SB/classic detector.
        scales: list[float] = [1.0]
        if min(h, w) < 700:
            scales.append(1.5)
        if min(h, w) < 520:
            scales.append(2.0)
        if min(h, w) < 360:
            scales.append(2.5)
        if min(h, w) < 320:
            scales.append(3.0)

        for sc in scales:
            for cname, img in candidates:
                for roi_img, ox, oy in _roi_candidates(img):
                    rh, rw = int(roi_img.shape[0]), int(roi_img.shape[1])
                    method_prefix = f"{cname}@s{sc:.1f},roi({ox},{oy},{rw}x{rh})"
                    if progress_cb is not None:
                        try:
                            progress_cb(method_prefix)
                        except Exception:
                            pass
                    if abs(sc - 1.0) > 1e-6:
                        up = cv2.resize(roi_img, (int(round(rw * sc)), int(round(rh * sc))), interpolation=cv2.INTER_LINEAR)
                        c, m = _try_one(up, scale_back=(1.0 / sc), ox=ox, oy=oy, method_prefix=method_prefix)
                    else:
                        c, m = _try_one(roi_img, scale_back=1.0, ox=ox, oy=oy, method_prefix=method_prefix)
                    if c is not None:
                        return c, m
        return None, "no_match"

    def _find_chessboard_corners_geometry(
        self,
        u8: "np.ndarray",
        cols: int,
        rows: int,
        progress_cb=None,
    ) -> tuple["np.ndarray | None", str]:
        # Geometry-stage detector: one board per already-cropped lens image.
        # Kept isolated from crop-stage detector by dedicated method.
        return self._find_chessboard_corners_robust(u8, cols, rows, progress_cb=progress_cb)

    def _skip_geometry_calibration(self) -> None:
        self.geometry_h = None
        self.geometry_correction_enabled = False
        self.reference_lens = 0
        self.geometry_capture_idx = 0
        self.geometry_corners = []
        self.geometry_focus_scores = []
        self.geometry_progress_var.set(100.0)
        self.geometry_progress_text_var.set("Skipped")
        if self.session_root is not None:
            self._save_json(
                self.session_root / "geometry_calibration.json",
                {
                    "stage": "geometry_calibration",
                    "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
                    "reference_lens": int(self.reference_lens),
                    "skipped": True,
                    "note": "geometry_correction_disabled",
                },
            )
            try:
                # Keep compatibility with loaders that expect H_lens.npy.
                np.save(self.session_root / "H_lens.npy", np.full((16, 3, 3), np.nan, dtype=np.float64))
            except Exception:
                pass
        self._show_page("main")
        self.status_var.set("Geometry calibration skipped. Geometric correction disabled.")

    def _capture_geometry_frame(self) -> None:
        if cv2 is None:
            self.status_var.set("OpenCV is required for geometry calibration")
            return
        got = self._get_latest_raw()
        if got is None:
            self.status_var.set("No frame for geometry capture")
            return
        try:
            cols = max(2, int(self.geometry_cols_var.get()))
            rows = max(2, int(self.geometry_rows_var.get()))
            target = max(1, int(self.geometry_target_var.get()))
        except Exception:
            self.status_var.set("Invalid geometry settings")
            return
        raw_arr, fmt = got
        lenses = self._apply_crop_to_raw(raw_arr)
        if lenses is None:
            self.status_var.set("Crop configuration is invalid")
            return
        work = self._apply_dark_flat(lenses)
        corners_dict: dict[int, np.ndarray] = {}
        focus_scores = np.zeros((16,), dtype=np.float32)
        for i in range(16):
            lens_no = i + 1
            gray_f = work[i]
            u8 = raw_to_u8(np.clip(gray_f, 0, 65535).astype(np.uint16), fmt)
            u8_stretch = raw_to_u8(np.clip(gray_f, 0, 65535).astype(np.uint16), fmt, autostretch=True)
            # Fallback source: original RAW crop as uint8, useful if flat normalization weakens pattern contrast.
            raw_u8 = raw_to_u8(np.asarray(lenses[i]), fmt, autostretch=False)
            raw_u8_stretch = raw_to_u8(np.asarray(lenses[i]), fmt, autostretch=True)
            lap = cv2.Laplacian(u8, cv2.CV_32F)
            focus_scores[i] = float(lap.var())
            corners = None
            method_used = "none"
            for src_name, src_img in (
                ("flat_u8", u8),
                ("flat_u8_autostretch", u8_stretch),
                ("raw_u8", raw_u8),
                ("raw_u8_autostretch", raw_u8_stretch),
            ):
                self.status_var.set(f"Geometry detect: lens {lens_no}/16, source={src_name}, method=robust(classic+sb)")
                self.geometry_progress_text_var.set(
                    f"Lens {lens_no}/16: source={src_name}, method=robust(classic+sb)"
                )
                try:
                    self.update_idletasks()
                except Exception:
                    pass
                method_ui_ts = 0.0

                def _method_progress(method_name: str) -> None:
                    nonlocal method_ui_ts
                    now_ts = time.monotonic()
                    if (now_ts - method_ui_ts) < 0.12:
                        return
                    method_ui_ts = now_ts
                    self.geometry_progress_text_var.set(
                        f"Lens {lens_no}/16: source={src_name}, method={method_name}"
                    )
                    try:
                        self.update_idletasks()
                    except Exception:
                        pass

                c_try, m_try = self._find_chessboard_corners_geometry(
                    src_img,
                    cols,
                    rows,
                    progress_cb=_method_progress,
                )
                if c_try is not None and c_try.shape[0] == cols * rows:
                    corners = c_try
                    method_used = f"{src_name}:{m_try}"
                    break
            if corners is not None and corners.shape[0] == cols * rows:
                corners_dict[i] = corners.astype(np.float32)
                if self.debug:
                    self._dbg(f"geom lens {lens_no}: found by {method_used}")
            else:
                self.status_var.set(f"Chessboard not found on lens {lens_no}. Frame rejected.")
                self.geometry_progress_text_var.set(f"Lens {lens_no}/16 failed. Capture rejected.")
                if self.debug:
                    self._dbg(
                        f"geom lens {lens_no}: chessboard not found; "
                        f"focus={focus_scores[i]:.2f} u8[min,max]=[{int(u8.min())},{int(u8.max())}] "
                        f"raw_u8[min,max]=[{int(raw_u8.min())},{int(raw_u8.max())}]"
                    )
                return
            self.geometry_progress_text_var.set(f"Lens {lens_no}/16 OK ({method_used})")
        self.geometry_corners.append(corners_dict)
        self.geometry_focus_scores.append(focus_scores)
        self.geometry_capture_idx += 1
        if self.geometry_capture_idx < target:
            self.geometry_btn_var.set(f"Capture chess frame {self.geometry_capture_idx + 1}/{target}")
            self.status_var.set(f"Valid geometry frame captured: {self.geometry_capture_idx}/{target}")
            return
        self.geometry_btn_var.set("Solving geometry...")
        self.geometry_progress_var.set(5.0)
        self.geometry_progress_text_var.set("Estimating homographies...")
        self.status_var.set("Geometry solve started")
        threading.Thread(target=self._solve_geometry_worker, args=(cols, rows), daemon=True).start()

    def _solve_geometry_worker(self, cols: int, rows: int) -> None:
        if cv2 is None:
            self._push_ui_event("status", "OpenCV is unavailable")
            return
        try:
            focus_stack = np.stack(self.geometry_focus_scores, axis=0).astype(np.float32)  # [n,16]
            focus_mean = focus_stack.mean(axis=0)
            order = np.argsort(focus_mean)
            ref = int(order[len(order) // 2])
            H = np.full((16, 3, 3), np.nan, dtype=np.float64)
            H[ref] = np.eye(3, dtype=np.float64)
            lens_info: list[dict[str, object]] = []
            for i in range(16):
                if i == ref:
                    lens_info.append({"lens_index": i, "status": "ok", "reference_lens": True})
                    self._push_ui_event("geometry_progress", {"value": 10 + (i + 1) * 5, "text": f"Lens {i + 1}/16"})
                    continue
                src_all: list[np.ndarray] = []
                dst_all: list[np.ndarray] = []
                for cap in self.geometry_corners:
                    src = cap.get(i)
                    dst = cap.get(ref)
                    if src is None or dst is None:
                        continue
                    if src.shape[0] != dst.shape[0] or src.shape[0] < 4:
                        continue
                    src_all.append(src)
                    dst_all.append(dst)
                if not src_all:
                    lens_info.append({"lens_index": i, "status": "insufficient_data"})
                    self._push_ui_event("geometry_progress", {"value": 10 + (i + 1) * 5, "text": f"Lens {i + 1}/16"})
                    continue
                src_pts = np.vstack(src_all).astype(np.float32)
                dst_pts = np.vstack(dst_all).astype(np.float32)
                h_mat, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 3.0)
                if h_mat is None:
                    lens_info.append({"lens_index": i, "status": "failed"})
                else:
                    H[i] = h_mat.astype(np.float64)
                    inliers = int(mask.sum()) if mask is not None else int(src_pts.shape[0])
                    lens_info.append({"lens_index": i, "status": "ok", "inliers": inliers, "points_used": int(src_pts.shape[0])})
                self._push_ui_event("geometry_progress", {"value": 10 + (i + 1) * 5, "text": f"Lens {i + 1}/16"})

            self.geometry_h = H
            self.geometry_correction_enabled = True
            self.reference_lens = ref
            if self.session_root is not None:
                np.save(self.session_root / "H_lens.npy", H)
                np.save(self.session_root / "focus_scores.npy", focus_stack)
                self._save_json(
                    self.session_root / "focus_scores.json",
                    {
                        "focus_score_metric": "variance_of_laplacian",
                        "mean_scores": focus_mean.tolist(),
                        "reference_lens": ref,
                    },
                )
                self._save_json(
                    self.session_root / "geometry_calibration.json",
                    {
                        "stage": "geometry_calibration",
                        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
                        "reference_lens": ref,
                        "skipped": False,
                        "inner_corners": {"cols": cols, "rows": rows},
                        "captures_count": len(self.geometry_corners),
                        "lenses": lens_info,
                    },
                )
            self._push_ui_event("geometry_done", {"reference_lens": ref})
        except Exception as exc:
            self._push_ui_event("status", f"Geometry solve failed: {exc}")
            self._push_ui_event("geometry_progress", {"value": 0.0, "text": ""})

    # ----------------------------- Main snapshot -----------------------------
    def _on_main_lens_change(self, _event: tk.Event) -> None:
        try:
            self.main_lens_var.set(int(self.main_lens_combo.get()))
        except Exception:
            self.main_lens_var.set(1)
        self._update_analyze_button_state()
        if self._analysis_mode:
            return
        self._render_preview(force=True)
        self._draw_main_overlay()

    def _toggle_white_point_pick(self) -> None:
        if self.current_page != "main":
            self.status_var.set("White point can be set only on the main page")
            return
        if self._analysis_mode:
            self.status_var.set("Close analysis mode before setting white point")
            return
        if self._white_pick_active:
            self._white_pick_active = False
            self._white_pick_start_xy = None
            self._white_pick_end_xy = None
            self.white_point_button_var.set("Задать точку белого")
            self.status_var.set("White point selection cancelled")
            self._update_analyze_button_state()
            self._draw_main_overlay()
            return
        if str(self.main_view_mode.get()) != "single":
            self.status_var.set("Switch to Single Lens mode to set white point")
            return
        self._white_pick_active = True
        self._white_pick_start_xy = None
        self._white_pick_end_xy = None
        self.white_point_button_var.set("Отмена выбора точки белого")
        self.status_var.set("Select white ROI on preview (drag rectangle)")
        self._update_analyze_button_state()
        self._draw_main_overlay()

    def _main_canvas_to_image(self, x: float, y: float, clamp: bool = False) -> tuple[float, float] | None:
        m = self._main_display_map
        shape = self._main_rgb_shape
        if m is None or shape is None:
            return None
        ox, oy, s = m
        if s <= 0:
            return None
        h, w = int(shape[0]), int(shape[1])
        if h <= 0 or w <= 0:
            return None
        ix = (float(x) - float(ox)) / float(s)
        iy = (float(y) - float(oy)) / float(s)
        if clamp:
            ix = min(max(ix, 0.0), max(0.0, float(w) - 1e-6))
            iy = min(max(iy, 0.0), max(0.0, float(h) - 1e-6))
            return ix, iy
        if ix < 0.0 or iy < 0.0 or ix >= float(w) or iy >= float(h):
            return None
        return ix, iy

    def _on_main_canvas_press(self, event: tk.Event) -> None:
        if self._analysis_mode:
            self._on_analysis_canvas_click(event)
            return
        if self.current_page != "main" or not self._white_pick_active:
            return
        if str(self.main_view_mode.get()) != "single":
            return
        xy = self._main_canvas_to_image(float(event.x), float(event.y), clamp=False)
        if xy is None:
            return
        self._white_pick_start_xy = xy
        self._white_pick_end_xy = xy
        self._draw_main_overlay()

    def _on_main_canvas_drag(self, event: tk.Event) -> None:
        if self._analysis_mode:
            return
        if self.current_page != "main" or not self._white_pick_active:
            return
        if self._white_pick_start_xy is None:
            return
        xy = self._main_canvas_to_image(float(event.x), float(event.y), clamp=True)
        if xy is None:
            return
        self._white_pick_end_xy = xy
        self._draw_main_overlay()

    def _compute_white_reference_from_roi(self, roi_xyxy: tuple[int, int, int, int]) -> "np.ndarray | None":
        got = self._get_latest_raw()
        if got is None:
            return None
        if self.crop_boxes is None or len(self.crop_boxes) != 16:
            self.status_var.set("Cannot set white point: crop calibration is missing")
            return None
        raw_arr, fmt = got
        raw_lenses = self._apply_crop_to_raw(raw_arr)
        if raw_lenses is None or int(raw_lenses.shape[0]) != 16:
            self.status_var.set("Cannot set white point: invalid cropped lenses")
            return None
        corrected = self._apply_dark_flat(raw_lenses)
        h_lens = int(corrected.shape[1])
        w_lens = int(corrected.shape[2])
        x0, y0, x1, y1 = roi_xyxy
        x0 = max(0, min(w_lens - 1, int(x0)))
        y0 = max(0, min(h_lens - 1, int(y0)))
        x1 = max(x0 + 1, min(w_lens, int(x1)))
        y1 = max(y0 + 1, min(h_lens, int(y1)))
        if x1 <= x0 or y1 <= y0:
            self.status_var.set("White point ROI is empty")
            return None
        boxes_sorted = self._sorted_crop_boxes()
        ref = np.empty((48,), dtype=np.float32)
        for i in range(16):
            pattern_i = None
            if i < len(boxes_sorted):
                bi = boxes_sorted[i]
                pattern_i = bayer_pattern_for_crop(fmt, int(bi["x"]), int(bi["y"]))
            rgb_u8 = debayer_menon_rgb(
                np.clip(corrected[i], 0, 65535).astype(np.uint16),
                fmt,
                pattern_override=pattern_i,
            )
            patch = rgb_u8[y0:y1, x0:x1]
            if patch.size == 0:
                self.status_var.set("White point ROI does not intersect lens images")
                return None
            means = patch.reshape(-1, 3).mean(axis=0).astype(np.float32)
            ref[(i * 3) : (i * 3 + 3)] = means
        return np.maximum(ref, 1e-6)

    def _on_main_canvas_release(self, event: tk.Event) -> None:
        if self._analysis_mode:
            return
        if self.current_page != "main" or not self._white_pick_active:
            return
        if str(self.main_view_mode.get()) != "single":
            self.status_var.set("Switch to Single Lens mode to complete white point selection")
            return
        if self._white_pick_start_xy is None:
            return
        end_xy = self._main_canvas_to_image(float(event.x), float(event.y), clamp=True)
        if end_xy is None:
            end_xy = self._white_pick_end_xy
        if end_xy is None:
            return
        self._white_pick_end_xy = end_xy
        x0f, y0f = self._white_pick_start_xy
        x1f, y1f = self._white_pick_end_xy
        x0 = int(math.floor(min(x0f, x1f)))
        y0 = int(math.floor(min(y0f, y1f)))
        x1 = int(math.ceil(max(x0f, x1f)))
        y1 = int(math.ceil(max(y0f, y1f)))
        shape = self._main_rgb_shape
        if shape is None:
            self.status_var.set("Cannot set white point: preview is not ready")
            return
        h, w = int(shape[0]), int(shape[1])
        x0 = max(0, min(w - 1, x0))
        y0 = max(0, min(h - 1, y0))
        x1 = max(x0 + 1, min(w, x1))
        y1 = max(y0 + 1, min(h, y1))
        if (x1 - x0) < 2 or (y1 - y0) < 2:
            self.status_var.set("ROI is too small. Select a larger area.")
            self._draw_main_overlay()
            return
        ref = self._compute_white_reference_from_roi((x0, y0, x1, y1))
        if ref is None:
            self._draw_main_overlay()
            return
        lens_idx = max(1, min(16, int(self.main_lens_var.get()))) - 1
        self.white_point_roi = (x0, y0, x1, y1)
        self.white_point_lens = int(lens_idx)
        self.white_point_ref = ref.astype(np.float32)
        self.white_point_created_utc = dt.datetime.now(dt.timezone.utc).isoformat()
        self._save_white_point_to_session()
        self._white_pick_active = False
        self._white_pick_start_xy = None
        self._white_pick_end_xy = None
        self.white_point_button_var.set("Задать точку белого")
        self._update_white_point_info()
        self.status_var.set(
            f"White point set from lens {lens_idx + 1}: ROI {x1 - x0}x{y1 - y0} at ({x0}, {y0})"
        )
        self._update_analyze_button_state()
        self._draw_main_overlay()

    def _draw_main_overlay(self) -> None:
        if self.current_page != "main":
            return
        if self._main_display_map is None:
            return
        self.main_canvas.delete("main_overlay")
        ox, oy, s = self._main_display_map
        selected_lens_idx = max(1, min(16, int(self.main_lens_var.get()))) - 1
        if (
            self.white_point_roi is not None
            and str(self.main_view_mode.get()) == "single"
            and self.white_point_lens is not None
            and int(self.white_point_lens) == selected_lens_idx
        ):
            x0, y0, x1, y1 = self.white_point_roi
            self.main_canvas.create_rectangle(
                ox + float(x0) * s,
                oy + float(y0) * s,
                ox + float(x1) * s,
                oy + float(y1) * s,
                outline="#44dd88",
                width=2,
                tags="main_overlay",
            )
        if self._white_pick_active and self._white_pick_start_xy is not None and self._white_pick_end_xy is not None:
            x0f, y0f = self._white_pick_start_xy
            x1f, y1f = self._white_pick_end_xy
            self.main_canvas.create_rectangle(
                ox + float(x0f) * s,
                oy + float(y0f) * s,
                ox + float(x1f) * s,
                oy + float(y1f) * s,
                outline="#ffd24a",
                width=2,
                tags="main_overlay",
            )

    def _on_face_seg_lens_change(self, _event: tk.Event | None = None) -> None:
        try:
            self.face_seg_lens_var.set(int(self.face_seg_lens_combo.get()))
        except Exception:
            self.face_seg_lens_var.set(0)
        self._update_analyze_button_state()

    def _start_face_seg_model_preload(self) -> None:
        if self._face_seg_preload_started:
            return
        self._face_seg_preload_started = True
        threading.Thread(target=self._face_seg_preload_worker, name="hydra-face-seg-preload", daemon=True).start()

    def _face_seg_preload_worker(self) -> None:
        with self._face_seg_model_lock:
            if self._face_seg_model_ready or self._face_seg_model_loading:
                return
            self._face_seg_model_loading = True
        self._push_ui_event("face_seg_status", "Face segmentation: loading model...")
        try:
            import torch as torch_mod_local  # type: ignore

            try:
                from modelscope import Sam3Model, Sam3Processor  # type: ignore
            except ImportError:
                from transformers import Sam3Model, Sam3Processor  # type: ignore

            device = "cuda:0" if bool(torch_mod_local.cuda.is_available()) else "cpu"
            model = Sam3Model.from_pretrained("facebook/sam3").to(device)
            processor = Sam3Processor.from_pretrained("facebook/sam3")
            with self._face_seg_model_lock:
                self._face_seg_torch = torch_mod_local
                self._face_seg_device = device
                self._face_seg_model = model
                self._face_seg_processor = processor
                self._face_seg_model_ready = True
                self._face_seg_model_error = None
                self._face_seg_model_loading = False
            self._push_ui_event("face_seg_status", f"Face segmentation: model ready on {device}")
            self._push_ui_event("analysis_state", {"analyze_ready_check": True})
        except Exception as exc:
            with self._face_seg_model_lock:
                self._face_seg_model_ready = False
                self._face_seg_model_error = str(exc)
                self._face_seg_model_loading = False
            self._push_ui_event("face_seg_status", f"Face segmentation unavailable: {exc}")
            self._push_ui_event("analysis_state", {"analyze_ready_check": True})

    def _face_cls_model_path(self) -> Path:
        return self._weights_dir() / "face_cls_model.json"

    def _start_face_cls_model_preload(self) -> None:
        if self._face_cls_preload_started:
            return
        self._face_cls_preload_started = True
        threading.Thread(target=self._face_cls_preload_worker, name="hydra-face-cls-preload", daemon=True).start()

    def _load_face_cls_model(self, model_path: Path) -> dict[str, object]:
        if not model_path.exists():
            raise RuntimeError(f"classifier model not found: {model_path}")
        payload = json.loads(model_path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise RuntimeError("classifier model JSON must be an object")
        mean = np.asarray(payload.get("mean", []), dtype=np.float32).reshape(-1)
        std = np.asarray(payload.get("std", []), dtype=np.float32).reshape(-1)
        weights = np.asarray(payload.get("weights", []), dtype=np.float32).reshape(-1)
        bias = float(payload.get("bias", 0.0))
        feature_count = int(payload.get("feature_count", int(weights.size)))
        threshold = float(payload.get("threshold_live", 0.5))
        if int(mean.size) != feature_count or int(std.size) != feature_count or int(weights.size) != feature_count:
            raise RuntimeError(
                "classifier model has inconsistent feature sizes: "
                f"feature_count={feature_count}, mean={int(mean.size)}, std={int(std.size)}, weights={int(weights.size)}"
            )
        if feature_count <= 0:
            raise RuntimeError("classifier model has empty feature vector")
        std = np.where(std < 1e-8, 1.0, std).astype(np.float32)
        return {
            "feature_count": int(feature_count),
            "mean": mean,
            "std": std,
            "weights": weights,
            "bias": float(bias),
            "threshold_live": float(threshold),
            "model_path": str(model_path),
        }

    def _face_cls_preload_worker(self) -> None:
        if self._face_cls_model_ready or self._face_cls_model_loading:
            return
        self._face_cls_model_loading = True
        try:
            model = self._load_face_cls_model(self._face_cls_model_path())
            self._face_cls_model = model
            self._face_cls_model_ready = True
            self._face_cls_model_error = None
            self._face_cls_model_loading = False
            self._push_ui_event("face_cls_status", f"Face classification: model ready ({Path(str(model.get('model_path'))).name})")
            self._push_ui_event("analysis_state", {"analyze_ready_check": True})
        except Exception as exc:
            self._face_cls_model = None
            self._face_cls_model_ready = False
            self._face_cls_model_error = str(exc)
            self._face_cls_model_loading = False
            self._push_ui_event("face_cls_status", f"Face classification unavailable: {exc}")
            self._push_ui_event("analysis_state", {"analyze_ready_check": True})

    def _start_analysis_recon_precheck(self) -> None:
        if self._analysis_recon_checked:
            return
        threading.Thread(target=self._analysis_recon_precheck_worker, name="hydra-analysis-precheck", daemon=True).start()

    def _recon_worker_script_path(self) -> Path:
        return Path(__file__).resolve().with_name("hsi_recon_worker.py")

    def _readline_with_timeout(self, stream, timeout_s: float) -> str | None:
        if stream is None:
            return None
        try:
            fd = stream.fileno()
        except Exception:
            return None
        try:
            ready, _, _ = select.select([fd], [], [], max(0.01, float(timeout_s)))
        except Exception:
            return None
        if not ready:
            return None
        try:
            line = stream.readline()
        except Exception:
            return None
        if line is None:
            return None
        return str(line)

    def _read_json_message_with_timeout(self, stream, timeout_s: float) -> dict[str, object] | None:
        deadline = time.monotonic() + max(0.01, float(timeout_s))
        while time.monotonic() < deadline:
            left = max(0.01, deadline - time.monotonic())
            line = self._readline_with_timeout(stream, timeout_s=left)
            if line is None:
                return None
            s = line.strip()
            if not s:
                continue
            try:
                msg = json.loads(s)
                if isinstance(msg, dict):
                    return msg
            except Exception:
                if self.debug:
                    self._dbg(f"recon worker non-json output: {s[:200]}")
                continue
        return None

    def _stop_recon_worker_locked(self) -> None:
        proc = self._recon_worker_proc
        self._recon_worker_proc = None
        self._recon_worker_ready_info = None
        if proc is None:
            return
        try:
            if proc.poll() is None and proc.stdin is not None:
                req = {"id": -1, "cmd": "shutdown"}
                proc.stdin.write(json.dumps(req, ensure_ascii=False) + "\n")
                proc.stdin.flush()
                _ = self._readline_with_timeout(proc.stdout, timeout_s=1.0)
        except Exception:
            pass
        try:
            if proc.poll() is None:
                proc.terminate()
                proc.wait(timeout=1.0)
        except Exception:
            try:
                proc.kill()
            except Exception:
                pass

    def _start_recon_worker_locked(self) -> None:
        proc = self._recon_worker_proc
        if proc is not None and proc.poll() is None:
            return
        if proc is not None:
            self._stop_recon_worker_locked()

        infer_py = Path("/Users/mac/Desktop/HSIRestore/infer.py")
        weights_dir = self._weights_dir()
        config_path = weights_dir / "config.yaml"
        ckpt_path = weights_dir / "weights.ckpt"
        worker_script = self._recon_worker_script_path()
        if not worker_script.exists():
            raise RuntimeError(f"recon worker script not found: {worker_script}")
        if not infer_py.exists():
            raise RuntimeError(f"infer.py not found: {infer_py}")
        if not config_path.exists():
            raise RuntimeError(f"config.yaml not found: {config_path}")
        if not ckpt_path.exists():
            raise RuntimeError(f"weights.ckpt not found: {ckpt_path}")

        cmd = [
            sys.executable,
            "-u",
            str(worker_script),
            "--infer-py",
            str(infer_py),
            "--config",
            str(config_path),
            "--checkpoint",
            str(ckpt_path),
            "--device",
            "auto",
            "--precision",
            "fp16",
        ]
        proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        msg = self._read_json_message_with_timeout(proc.stdout, timeout_s=60.0)
        if msg is None:
            err_tail = ""
            try:
                if proc.poll() is not None and proc.stderr is not None:
                    err_tail = (proc.stderr.read() or "").strip()
            except Exception:
                pass
            try:
                proc.terminate()
            except Exception:
                pass
            raise RuntimeError(f"recon worker startup timeout. {err_tail[-500:]}")
        if not bool(msg.get("ok", False)):
            try:
                proc.terminate()
            except Exception:
                pass
            raise RuntimeError(str(msg.get("error", "recon worker failed to initialize")))

        self._recon_worker_proc = proc
        self._recon_worker_ready_info = dict(msg)
        self._recon_worker_req_id = 0
        if self.debug:
            self._dbg(f"recon worker ready: {json.dumps(msg, ensure_ascii=False)}")

    def _recon_worker_infer_paths_locked(self, input_path: Path, output_path: Path, timeout_s: float = 300.0) -> dict[str, object]:
        self._start_recon_worker_locked()
        proc = self._recon_worker_proc
        if proc is None or proc.stdin is None or proc.stdout is None:
            raise RuntimeError("recon worker is not available")
        self._recon_worker_req_id += 1
        req_id = int(self._recon_worker_req_id)
        req = {
            "id": req_id,
            "cmd": "infer",
            "input": str(input_path),
            "output": str(output_path),
        }
        try:
            proc.stdin.write(json.dumps(req, ensure_ascii=False) + "\n")
            proc.stdin.flush()
        except Exception as exc:
            self._stop_recon_worker_locked()
            raise RuntimeError(f"Failed to send request to recon worker: {exc}")

        msg = self._read_json_message_with_timeout(proc.stdout, timeout_s=timeout_s)
        if msg is None:
            self._stop_recon_worker_locked()
            raise RuntimeError("Timed out waiting for reconstruction response")
        if int(msg.get("id", -1)) != req_id:
            raise RuntimeError(f"Unexpected recon worker response id={msg.get('id')} expected={req_id}")
        if not bool(msg.get("ok", False)):
            raise RuntimeError(str(msg.get("error", "Reconstruction failed in worker")))
        return dict(msg)

    def _analysis_recon_precheck_worker(self) -> None:
        try:
            with self._recon_worker_lock:
                self._start_recon_worker_locked()
            self._analysis_recon_checked = True
            self._analysis_recon_ready = True
            self._analysis_recon_error = None
        except Exception as exc:
            self._analysis_recon_checked = True
            self._analysis_recon_ready = False
            self._analysis_recon_error = str(exc)
            self._push_ui_event("status", f"Analysis reconstruction unavailable: {exc}")
        finally:
            self._push_ui_event("analysis_state", {"analyze_ready_check": True})

    def _update_analyze_button_state(self) -> None:
        if not hasattr(self, "close_analysis_btn"):
            return
        white_ok = self.white_point_ref is not None
        model_ok = bool(self._face_seg_model_ready)
        cls_ok = bool(self._face_cls_model_ready)
        lens_ok = 1 <= int(self.face_seg_lens_var.get()) <= 16
        recon_ok = bool(self._analysis_recon_ready)
        can_analyze = white_ok and model_ok and lens_ok and recon_ok and (not self._analysis_busy) and (not self._analysis_mode)
        can_classify = can_analyze and cls_ok
        if self._analysis_busy:
            self.analyze_button_var.set("Анализ...")
        elif self._analysis_mode:
            self.analyze_button_var.set("Анализ")
        elif can_analyze:
            self.analyze_button_var.set("Анализ")
        elif not white_ok:
            self.analyze_button_var.set("Анализ (нужна точка белого)")
        elif not model_ok:
            self.analyze_button_var.set("Анализ (модель не готова)")
        elif not lens_ok:
            self.analyze_button_var.set("Анализ (выберите линзу)")
        elif not recon_ok:
            self.analyze_button_var.set("Анализ (реконструкция не готова)")
        else:
            self.analyze_button_var.set("Анализ")
        analyze_clickable = (not self._analysis_busy) and (not self._analysis_mode)
        if hasattr(self, "analyze_btn_fake"):
            self.analyze_btn_fake.state(["!disabled"] if analyze_clickable else ["disabled"])
        if hasattr(self, "analyze_btn_live"):
            self.analyze_btn_live.state(["!disabled"] if analyze_clickable else ["disabled"])
        if hasattr(self, "classify_btn"):
            if self._analysis_busy:
                self.classify_button_var.set("Classification...")
            elif can_classify:
                self.classify_button_var.set("Classification")
            elif not cls_ok:
                self.classify_button_var.set("Classification (model not ready)")
            else:
                self.classify_button_var.set("Classification")
            self.classify_btn.state(["!disabled"] if analyze_clickable else ["disabled"])
        if self._analysis_mode:
            self.close_analysis_btn.state(["!disabled"])
            if hasattr(self, "analysis_rotate_btn"):
                self.analysis_rotate_btn.state(["!disabled"])
        else:
            self.close_analysis_btn.state(["disabled"])
            if hasattr(self, "analysis_rotate_btn"):
                self.analysis_rotate_btn.state(["disabled"])

    def _current_frame_id(self) -> int:
        if self.last_frame is None:
            return -1
        try:
            return int((self.last_frame.meta or {}).get("frame_id", -1))
        except Exception:
            return -1

    def _wait_for_frame_advance(self, previous_frame_id: int, timeout_s: float) -> bool:
        deadline = time.monotonic() + max(0.05, float(timeout_s))
        while time.monotonic() < deadline:
            try:
                self.update()
            except Exception:
                pass
            current = self._current_frame_id()
            if current > previous_frame_id:
                return True
            time.sleep(0.01)
        return False

    def _close_analysis_mode(self) -> None:
        self._analysis_mode = False
        self._analysis_base_rgb = None
        self._analysis_view_rgb = None
        self._analysis_face_mask = None
        self._analysis_hsi_hwc = None
        self._analysis_wavelengths_nm = None
        self._analysis_spectrum = None
        self._analysis_pick_xy = None
        self._analysis_fas_label = None
        self._analysis_fas_is_live = None
        self._analysis_image_width = 0
        self._analysis_rotation_quarters = 0
        self._update_analyze_button_state()
        self._render_preview(force=True)

    def _on_analysis_plot_controls_changed(self, _event: tk.Event | None = None) -> None:
        self._refresh_analysis_view()

    def _analysis_plot_spectrum(self, spectrum: "np.ndarray | None") -> "np.ndarray | None":
        if spectrum is None:
            return None
        ys = np.asarray(spectrum, dtype=np.float32).reshape(-1)
        if ys.size <= 2:
            return ys
        if not bool(self.analysis_smooth_enabled_var.get()):
            return ys
        try:
            sigma = float(self.analysis_smooth_sigma_var.get())
        except Exception:
            sigma = 1.0
        sigma = max(0.01, min(20.0, sigma))
        if gaussian_filter is not None:
            try:
                return np.asarray(gaussian_filter(ys, sigma=sigma), dtype=np.float32)
            except Exception:
                pass
        # Fallback smoothing: moving average.
        try:
            w = int(self.analysis_smooth_window_var.get())
        except Exception:
            w = 5
        w = max(1, min(31, w))
        if (w % 2) == 0:
            w += 1
        if w <= 1:
            return ys
        kernel = np.ones((w,), dtype=np.float32) / float(w)
        return np.asarray(np.convolve(ys, kernel, mode="same"), dtype=np.float32)

    def _refresh_analysis_view(self) -> None:
        if not self._analysis_mode:
            return
        if self._analysis_hsi_hwc is None:
            return
        base_rgb = self._analysis_base_rgb
        if base_rgb is None:
            base_rgb = self._synthesize_rgb_from_hsi(self._analysis_hsi_hwc, wavelengths=self._analysis_wavelengths_nm)
            self._analysis_base_rgb = base_rgb
        spec_plot = self._analysis_plot_spectrum(self._analysis_spectrum)
        wl = self._analysis_wavelengths_nm if self._analysis_wavelengths_nm is not None else np.arange(int(self._analysis_hsi_hwc.shape[2]), dtype=np.float32)
        self._analysis_view_rgb = self._compose_analysis_view(
            base_rgb,
            self._analysis_face_mask,
            spec_plot,
            wl,
            self._analysis_pick_xy,
            self._analysis_fas_label,
            self._analysis_fas_is_live,
            auto_y=bool(self.analysis_auto_y_var.get()),
        )
        self._draw_analysis_to_main_canvas()

    def _rotate_analysis_90(self) -> None:
        if not self._analysis_mode:
            return
        if self._analysis_base_rgb is None or self._analysis_hsi_hwc is None:
            return
        old_h, old_w = int(self._analysis_hsi_hwc.shape[0]), int(self._analysis_hsi_hwc.shape[1])
        self._analysis_base_rgb = np.ascontiguousarray(np.rot90(self._analysis_base_rgb, k=1))
        self._analysis_hsi_hwc = np.ascontiguousarray(np.rot90(self._analysis_hsi_hwc, k=1, axes=(0, 1)))
        if self._analysis_face_mask is not None:
            self._analysis_face_mask = np.ascontiguousarray(np.rot90(self._analysis_face_mask, k=1))
        if self._analysis_pick_xy is not None:
            x, y = int(self._analysis_pick_xy[0]), int(self._analysis_pick_xy[1])
            # np.rot90(k=1): (x, y) -> (y, old_w - 1 - x)
            self._analysis_pick_xy = (int(y), int(max(0, old_w - 1 - x)))
        self._analysis_rotation_quarters = int((self._analysis_rotation_quarters + 1) % 4)
        self._refresh_analysis_view()

    def _weights_dir(self) -> Path:
        return Path(__file__).resolve().parents[1] / "weights"

    def _start_classification(self) -> None:
        self._start_analysis(fas_force_live=None, use_real_classifier=True)

    def _start_analysis(self, fas_force_live: bool | None = None, use_real_classifier: bool = False) -> None:
        if self._analysis_busy:
            self.status_var.set("Analysis is already running")
            return
        if self._analysis_mode:
            self.status_var.set("Close analysis mode before running a new analysis")
            return
        if use_real_classifier and (not self._face_cls_model_ready):
            msg = self._face_cls_model_error or "face classifier model is not ready"
            self.status_var.set(f"Classification unavailable: {msg}")
            self._update_analyze_button_state()
            return
        if self.white_point_ref is None:
            self.status_var.set("Analysis unavailable: set white point first")
            self._update_analyze_button_state()
            return
        if not self._face_seg_model_ready:
            self.status_var.set("Analysis unavailable: face segmentation model is not ready")
            self._update_analyze_button_state()
            return
        if not self._analysis_recon_checked:
            self._start_analysis_recon_precheck()
            self.status_var.set("Analysis unavailable: reconstruction precheck is running, try again")
            self._update_analyze_button_state()
            return
        if not self._analysis_recon_ready:
            msg = self._analysis_recon_error or "reconstruction stack is unavailable"
            self.status_var.set(f"Analysis unavailable: {msg}")
            try:
                messagebox.showerror("Analysis Error", f"Cannot start analysis.\n\n{msg}")
            except Exception:
                pass
            self._update_analyze_button_state()
            return
        lens_idx = int(self.face_seg_lens_var.get())
        if lens_idx < 1 or lens_idx > 16:
            self.status_var.set("Analysis unavailable: set segmentation lens (1..16)")
            self._update_analyze_button_state()
            return
        if self.last_frame is None:
            self.status_var.set("Analysis unavailable: no live frame")
            return
        self._analysis_busy = True
        self._update_analyze_button_state()
        if use_real_classifier:
            self.status_var.set("Classification started: building reflectance cube...")
        else:
            self.status_var.set("Analysis started: building reflectance cube...")
        threading.Thread(
            target=self._analysis_worker,
            args=(fas_force_live, use_real_classifier),
            name="hydra-analysis",
            daemon=True,
        ).start()

    def _analysis_worker(self, fas_force_live: bool | None = None, use_real_classifier: bool = False) -> None:
        try:
            inputs = self._build_analysis_inputs()
            seg_rgb = inputs["seg_rgb"]
            reflectance = inputs["reflectance"]
            self._push_ui_event("status", "Analysis: running face segmentation...")
            mask = self._segment_face_single(seg_rgb, confidence=0.5)
            self._push_ui_event("status", "Analysis: running HSI reconstruction...")
            restored_hsi, analysis_tmp_dir = self._run_hsi_reconstruction(reflectance)
            wavelengths = self._load_wavelengths_for_hsi(restored_hsi.shape[2])
            base_rgb = self._synthesize_rgb_from_hsi(restored_hsi, wavelengths=wavelengths)
            if mask is not None and (mask.shape[0] != restored_hsi.shape[0] or mask.shape[1] != restored_hsi.shape[1]):
                if cv2 is not None:
                    mask = cv2.resize(mask, (restored_hsi.shape[1], restored_hsi.shape[0]), interpolation=cv2.INTER_NEAREST)
                else:
                    pil = Image.fromarray(mask, mode="L")
                    pil = pil.resize((restored_hsi.shape[1], restored_hsi.shape[0]), Image.Resampling.NEAREST)
                    mask = np.asarray(pil, dtype=np.uint8)
            spectrum = None
            fas_label: str | None = None
            fas_is_live: bool | None = None
            fas_score: float | None = None
            face_spectrum_json_path: str | None = None
            if mask is not None and bool(np.any(mask > 127)):
                spectrum = self._spectrum_from_mask(restored_hsi, mask > 127)
                if use_real_classifier:
                    fas_label, fas_is_live, fas_score = self._fas_predict_from_model(spectrum)
                else:
                    fas_label, fas_is_live = self._fas_mock_predict_from_spectrum(spectrum, force_live=fas_force_live)
                face_spectrum_json_path = str(
                    self._save_face_spectrum_artifact(
                        analysis_tmp_dir=analysis_tmp_dir,
                        spectrum=spectrum,
                        wavelengths=wavelengths,
                        fas_label=fas_label,
                        fas_is_live=fas_is_live,
                        fas_score=fas_score,
                    )
                )
            spec_plot = self._analysis_plot_spectrum(spectrum)
            view_rgb = self._compose_analysis_view(
                base_rgb,
                mask,
                spec_plot,
                wavelengths,
                None,
                fas_label,
                fas_is_live,
                auto_y=bool(self.analysis_auto_y_var.get()),
            )
            self._push_ui_event(
                "analysis_done",
                {
                    "base_rgb": base_rgb,
                    "view_rgb": view_rgb,
                    "mask": mask,
                    "hsi_hwc": restored_hsi,
                    "wavelengths": wavelengths,
                    "spectrum": spectrum,
                    "face_found": bool(mask is not None and np.any(mask > 127)),
                    "fas_label": fas_label,
                    "fas_is_live": fas_is_live,
                    "fas_score": fas_score,
                    "face_spectrum_json": face_spectrum_json_path,
                    "run_mode": "classification" if use_real_classifier else "analysis",
                },
            )
        except Exception as exc:
            self._push_ui_event("analysis_error", str(exc))
        finally:
            self._push_ui_event("analysis_state", {"busy": False, "analyze_ready_check": True})

    def _build_analysis_inputs(self) -> dict[str, np.ndarray]:
        if self.last_frame is None:
            raise RuntimeError("No frame for analysis")
        if self.crop_boxes is None or len(self.crop_boxes) != 16:
            raise RuntimeError("Calibration is incomplete: crop data missing")
        if self.geometry_correction_enabled and self.geometry_h is None:
            raise RuntimeError("Calibration is incomplete: geometry data missing")
        if self.white_point_ref is None:
            raise RuntimeError("White point is not set")

        got = self._get_latest_raw()
        if got is None:
            raise RuntimeError("Failed to decode latest frame")
        raw_arr, fmt = got
        raw_lenses = self._apply_crop_to_raw(raw_arr)
        if raw_lenses is None:
            raise RuntimeError("Invalid crop data")
        corrected = self._apply_dark_flat(raw_lenses)
        boxes_sorted = self._sorted_crop_boxes()

        lens_rgb = np.empty((16, corrected.shape[1], corrected.shape[2], 3), dtype=np.float32)
        for i in range(16):
            pattern_i = None
            if i < len(boxes_sorted):
                bi = boxes_sorted[i]
                pattern_i = bayer_pattern_for_crop(fmt, int(bi["x"]), int(bi["y"]))
            rgb_u8 = debayer_menon_rgb(
                np.clip(corrected[i], 0, 65535).astype(np.uint16),
                fmt,
                pattern_override=pattern_i,
            )
            lens_rgb[i] = rgb_u8.astype(np.float32)

        if self.geometry_correction_enabled and self.geometry_h is not None:
            ref_idx = max(0, min(15, int(self.reference_lens)))
            ref_h = int(lens_rgb[ref_idx].shape[0])
            ref_w = int(lens_rgb[ref_idx].shape[1])
            warped = np.empty((16, ref_h, ref_w, 3), dtype=np.float32)
            for i in range(16):
                h_mat = self.geometry_h[i]
                if np.isnan(h_mat).any():
                    warped[i] = (
                        cv2.resize(lens_rgb[i], (ref_w, ref_h), interpolation=cv2.INTER_NEAREST)
                        if cv2 is not None
                        else lens_rgb[i]
                    )
                    continue
                if cv2 is not None:
                    warped[i] = cv2.warpPerspective(lens_rgb[i], h_mat.astype(np.float32), (ref_w, ref_h), flags=cv2.INTER_LINEAR)
                else:
                    warped[i] = lens_rgb[i]
        else:
            warped = lens_rgb.copy()

        cube = np.empty((48, warped.shape[1], warped.shape[2]), dtype=np.float32)
        for i in range(16):
            base = i * 3
            cube[base + 0] = warped[i, :, :, 0]
            cube[base + 1] = warped[i, :, :, 1]
            cube[base + 2] = warped[i, :, :, 2]

        mapping_idx, _mapping_wl, _mapping_labels, _mapping_source = self._ordered_wavelength_mapping_for_snapshot()
        cube_out = cube[mapping_idx, :, :] if mapping_idx else cube
        ref = np.asarray(self.white_point_ref, dtype=np.float32).reshape(-1)
        if ref.size == 3:
            ref = np.tile(ref, 16)
        if ref.size != 48:
            raise RuntimeError(f"Invalid white point size: {ref.size}")
        ref_out = ref[mapping_idx] if mapping_idx else ref
        if int(ref_out.size) != int(cube_out.shape[0]):
            raise RuntimeError(f"White point channels ({int(ref_out.size)}) != cube channels ({int(cube_out.shape[0])})")
        safe_ref = np.maximum(ref_out.astype(np.float32), 1e-6)[:, None, None]
        reflectance = np.clip(cube_out.astype(np.float32) / safe_ref, 0.0, 1.0).astype(np.float32)

        lens_num = int(self.face_seg_lens_var.get())
        if lens_num < 1 or lens_num > 16:
            raise RuntimeError("Segmentation lens is not set")
        lens_idx = lens_num - 1
        seg_rgb = np.clip(lens_rgb[lens_idx], 0, 255).astype(np.uint8)
        return {"reflectance": reflectance, "seg_rgb": seg_rgb}

    def _segment_face_single(self, rgb: np.ndarray, confidence: float = 0.5) -> "np.ndarray | None":
        with self._face_seg_model_lock:
            model = self._face_seg_model
            processor = self._face_seg_processor
            torch_mod = self._face_seg_torch
            device = self._face_seg_device
        if model is None or processor is None or torch_mod is None:
            raise RuntimeError("Face segmentation model is not loaded")
        img = Image.fromarray(rgb, mode="RGB")
        inputs = processor(images=img, text="Faces", return_tensors="pt")
        if hasattr(inputs, "to"):
            inputs = inputs.to(device)
        with torch_mod.no_grad():
            outputs = model(**inputs)
        target_sizes = inputs.get("original_sizes")
        ts = target_sizes.tolist() if hasattr(target_sizes, "tolist") else [[int(rgb.shape[0]), int(rgb.shape[1])]]
        results = processor.post_process_instance_segmentation(
            outputs,
            threshold=float(confidence),
            mask_threshold=float(confidence),
            target_sizes=ts,
        )
        return self._extract_face_mask_from_segmentation_result(
            results,
            target_h=int(rgb.shape[0]),
            target_w=int(rgb.shape[1]),
            torch_mod=torch_mod,
        )

    def _fas_mock_predict_from_spectrum(self, spectrum: np.ndarray, force_live: bool | None = None) -> tuple[str, bool]:
        if force_live is not None:
            is_live = bool(force_live)
            return ("Live Face" if is_live else "Fake Face"), is_live
        is_live = bool(self._fas_mock_next_live)
        self._fas_mock_next_live = not is_live
        return ("Live Face" if is_live else "Fake Face"), is_live

    def _fas_predict_from_model(self, spectrum: np.ndarray) -> tuple[str, bool, float]:
        model = self._face_cls_model
        if not self._face_cls_model_ready or not isinstance(model, dict):
            raise RuntimeError("Face classifier model is not loaded")
        x = np.asarray(spectrum, dtype=np.float32).reshape(-1)
        feat_n = int(model.get("feature_count", 0))
        if int(x.size) != feat_n:
            raise RuntimeError(f"Face classifier expected {feat_n} channels, got {int(x.size)}")
        mean = np.asarray(model.get("mean", []), dtype=np.float32).reshape(-1)
        std = np.asarray(model.get("std", []), dtype=np.float32).reshape(-1)
        weights = np.asarray(model.get("weights", []), dtype=np.float32).reshape(-1)
        bias = float(model.get("bias", 0.0))
        threshold = float(model.get("threshold_live", 0.5))
        z = (x - mean) / np.maximum(std, 1e-8)
        score = float(np.dot(z, weights) + bias)
        prob_live = float(1.0 / (1.0 + math.exp(-max(-60.0, min(60.0, score)))))
        is_live = bool(prob_live >= threshold)
        return ("Live Face" if is_live else "Fake Face"), is_live, prob_live

    def _save_face_spectrum_artifact(
        self,
        analysis_tmp_dir: Path,
        spectrum: np.ndarray,
        wavelengths: np.ndarray,
        fas_label: str | None,
        fas_is_live: bool | None,
        fas_score: float | None = None,
    ) -> Path:
        analysis_tmp_dir.mkdir(parents=True, exist_ok=True)
        ys = np.asarray(spectrum, dtype=np.float32).reshape(-1)
        xs = np.asarray(wavelengths, dtype=np.float32).reshape(-1)
        n = int(min(int(xs.size), int(ys.size)))
        if n <= 0:
            raise RuntimeError("Cannot save face spectrum artifact: empty spectrum")
        xs = xs[:n]
        ys = ys[:n]
        payload = {
            "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            "fas_label": str(fas_label) if fas_label is not None else None,
            "fas_is_live": bool(fas_is_live) if fas_is_live is not None else None,
            "fas_score_live": float(fas_score) if fas_score is not None else None,
            "channels": int(n),
            "wavelength_nm": [float(v) for v in xs.tolist()],
            "intensity": [float(v) for v in ys.tolist()],
        }
        out_path = analysis_tmp_dir / "face_spectrum.json"
        out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return out_path

    def _run_hsi_reconstruction(self, reflectance: np.ndarray) -> tuple[np.ndarray, Path]:
        if self.session_root is None:
            ts = dt.datetime.now().strftime("%Y-%m-%d_%H-%M-%S_%f")
            self.session_root = self.output_dir / f"session_{ts}_analysis"
            self.session_root.mkdir(parents=True, exist_ok=True)
        work_dir = self.session_root / "analysis_tmp" / dt.datetime.now().strftime("%Y-%m-%d_%H-%M-%S_%f")
        work_dir.mkdir(parents=True, exist_ok=True)
        input_path = work_dir / "reflectance.npy"
        output_path = work_dir / "restored.npy"
        np.save(input_path, reflectance.astype(np.float32))

        resp: dict[str, object]
        with self._recon_worker_lock:
            try:
                resp = self._recon_worker_infer_paths_locked(input_path, output_path, timeout_s=300.0)
            except Exception:
                # One automatic restart/retry path if worker got stale.
                self._stop_recon_worker_locked()
                resp = self._recon_worker_infer_paths_locked(input_path, output_path, timeout_s=300.0)

        if not output_path.exists():
            raise RuntimeError("HSI reconstruction did not produce output file")
        if self.debug and resp:
            infer_ms = resp.get("infer_ms")
            self._dbg(f"recon worker infer done: {infer_ms} ms")
        restored = np.load(output_path)
        return self._cube_to_hwc(restored), work_dir

    def _cube_to_hwc(self, cube: np.ndarray) -> np.ndarray:
        if cube.ndim != 3:
            raise RuntimeError(f"Restored cube must be 3D, got {cube.shape}")
        band_axis = int(np.argmin(cube.shape))
        if band_axis == 2:
            hwc = cube
        elif band_axis == 0:
            hwc = np.transpose(cube, (1, 2, 0))
        else:
            hwc = np.transpose(cube, (0, 2, 1))
        return np.asarray(hwc, dtype=np.float32)

    def _load_wavelengths_for_hsi(self, channels: int) -> np.ndarray:
        p = self._weights_dir() / "wavelengths.txt"
        if p.exists():
            values: list[float] = []
            for line in p.read_text(encoding="utf-8").splitlines():
                s = line.strip()
                if not s:
                    continue
                try:
                    values.append(float(s))
                except Exception:
                    continue
            arr = np.asarray(values, dtype=np.float32)
            if int(arr.size) == int(channels):
                return arr
            if int(arr.size) > int(channels):
                return arr[:channels]
        return np.arange(channels, dtype=np.float32)

    def _synthesize_rgb_from_hsi(self, hsi_hwc: np.ndarray, wavelengths: "np.ndarray | None" = None) -> np.ndarray:
        h, w, c = hsi_hwc.shape
        if c <= 0:
            return np.zeros((h, w, 3), dtype=np.uint8)

        # Build display RGB from nearest bands to target wavelengths:
        # R~630nm, G~550nm, B~450nm.
        target_nm = np.asarray([630.0, 550.0, 450.0], dtype=np.float32)
        idxs_rgb: list[int] = []
        chosen_wls: list[float] = []
        used: set[int] = set()
        wl_arr = np.asarray(wavelengths, dtype=np.float32).reshape(-1) if wavelengths is not None else np.zeros((0,), dtype=np.float32)
        if int(wl_arr.size) == int(c) and bool(np.all(np.isfinite(wl_arr))):
            for t in target_nm:
                order = np.argsort(np.abs(wl_arr - float(t)))
                pick = None
                for cand in order.tolist():
                    ci = int(cand)
                    if ci not in used:
                        pick = ci
                        break
                if pick is None:
                    pick = int(order[0]) if int(order.size) > 0 else 0
                used.add(int(pick))
                idxs_rgb.append(int(pick))
                chosen_wls.append(float(wl_arr[int(pick)]))
            if self.debug:
                self._dbg(
                    "analysis rgb synthesis: "
                    f"R@{chosen_wls[0]:.1f}nm(i={idxs_rgb[0]}), "
                    f"G@{chosen_wls[1]:.1f}nm(i={idxs_rgb[1]}), "
                    f"B@{chosen_wls[2]:.1f}nm(i={idxs_rgb[2]})"
                )
        else:
            # Fallback for missing wavelength grid: preserve legacy anchor bands,
            # but map them to RGB order (R,G,B) as (high, mid, low).
            low = min(c - 1, 5)
            mid = min(c - 1, 19)
            high = min(c - 1, 33)
            idxs_rgb = [high, mid, low]
            if self.debug:
                self._dbg(
                    "analysis rgb synthesis fallback indices: "
                    f"R(i={idxs_rgb[0]}), G(i={idxs_rgb[1]}), B(i={idxs_rgb[2]})"
                )

        rgb = np.stack([hsi_hwc[:, :, idxs_rgb[0]], hsi_hwc[:, :, idxs_rgb[1]], hsi_hwc[:, :, idxs_rgb[2]]], axis=2).astype(np.float32)
        p1 = float(np.percentile(rgb, 1.0))
        p99 = float(np.percentile(rgb, 99.0))
        if p99 <= p1 + 1e-6:
            p1, p99 = float(np.min(rgb)), float(np.max(rgb) + 1e-6)
        rgb = np.clip((rgb - p1) / (p99 - p1), 0.0, 1.0)
        return np.clip(rgb * 255.0, 0, 255).astype(np.uint8)

    def _spectrum_from_mask(self, hsi_hwc: np.ndarray, mask_bool: np.ndarray) -> np.ndarray:
        if mask_bool.shape[0] != hsi_hwc.shape[0] or mask_bool.shape[1] != hsi_hwc.shape[1]:
            raise RuntimeError("Mask/HIS shape mismatch")
        pix = hsi_hwc[mask_bool]
        if pix.size == 0:
            raise RuntimeError("Face mask is empty")
        return np.asarray(pix.mean(axis=0), dtype=np.float32)

    def _spectrum_from_point(self, hsi_hwc: np.ndarray, x: int, y: int, half: int = 2) -> np.ndarray:
        h, w, _c = hsi_hwc.shape
        x0 = max(0, int(x) - int(half))
        x1 = min(w, int(x) + int(half) + 1)
        y0 = max(0, int(y) - int(half))
        y1 = min(h, int(y) + int(half) + 1)
        patch = hsi_hwc[y0:y1, x0:x1, :]
        if patch.size == 0:
            raise RuntimeError("Empty patch for spectrum")
        return np.asarray(patch.reshape(-1, patch.shape[2]).mean(axis=0), dtype=np.float32)

    def _compose_analysis_view(
        self,
        base_rgb: np.ndarray,
        face_mask: "np.ndarray | None",
        spectrum: "np.ndarray | None",
        wavelengths: np.ndarray,
        pick_xy: tuple[int, int] | None,
        fas_label: str | None = None,
        fas_is_live: bool | None = None,
        auto_y: bool = False,
    ) -> np.ndarray:
        rgb = base_rgb.copy()
        if face_mask is not None and bool(np.any(face_mask > 127)):
            rgb = self._overlay_face_mask_on_rgb(rgb, face_mask)
        if pick_xy is not None and cv2 is not None:
            cv2.drawMarker(
                rgb,
                (int(pick_xy[0]), int(pick_xy[1])),
                color=(255, 180, 20),
                markerType=cv2.MARKER_CROSS,
                markerSize=12,
                thickness=2,
            )

        h, w = rgb.shape[:2]
        panel_w = max(460, int(round(w * 0.62)))
        panel = np.zeros((h, panel_w, 3), dtype=np.uint8)
        panel[:, :, :] = 22

        if ImageDraw is not None:
            pil = Image.fromarray(panel, mode="RGB")
            dr = ImageDraw.Draw(pil)
            dr.text((12, 12), "Spectrum", fill=(230, 230, 230))
            if spectrum is None:
                dr.text((12, 36), "No face mask.\nClick image to sample 5x5 spectrum.", fill=(210, 210, 210))
            panel = np.array(pil, dtype=np.uint8, copy=True)

        # OpenCV drawing APIs require writable contiguous output arrays.
        panel = np.ascontiguousarray(panel)

        x0 = 64
        x1 = panel_w - 20
        chart_h = max(220, int(round(h * 0.56)))
        y1 = 56
        y0 = min(h - 36, y1 + chart_h)
        if cv2 is not None:
            cv2.rectangle(panel, (x0, y1), (x1, y0), (80, 80, 80), 1)

        x_min = 0.0
        x_max = 1.0
        y_min = 0.0
        y_max = 1.0
        has_spec = spectrum is not None and int(spectrum.size) > 1
        ys = np.zeros((0,), dtype=np.float32)
        xs = np.zeros((0,), dtype=np.float32)

        if has_spec:
            ys = np.asarray(spectrum, dtype=np.float32).reshape(-1)
            xs = np.asarray(wavelengths, dtype=np.float32).reshape(-1)
            if int(xs.size) != int(ys.size):
                xs = np.arange(int(ys.size), dtype=np.float32)
            x_min, x_max = float(xs.min()), float(xs.max())
            if x_max <= x_min:
                x_max = x_min + 1.0
            if auto_y:
                y_min, y_max = float(np.min(ys)), float(np.max(ys))
                if y_max <= y_min + 1e-6:
                    y_max = y_min + 1e-6

        if cv2 is not None:
            axis_color = (170, 170, 170)
            label_color = (210, 210, 210)
            tick_color = (140, 140, 140)

            # Axis labels
            cv2.putText(panel, "Wavelength (nm)", (max(8, (x0 + x1) // 2 - 78), min(h - 8, y0 + 28)), cv2.FONT_HERSHEY_SIMPLEX, 0.48, label_color, 1, cv2.LINE_AA)
            cv2.putText(panel, "Intensity", (12, max(16, y1 - 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.48, label_color, 1, cv2.LINE_AA)

            # X ticks in wavelength units.
            for i in range(5):
                t = float(i) / 4.0
                px = int(round(x0 + t * (x1 - x0)))
                cv2.line(panel, (px, y0), (px, y0 + 5), tick_color, 1, cv2.LINE_AA)
                wl = x_min + t * (x_max - x_min)
                cv2.putText(panel, f"{wl:.0f}", (px - 16, y0 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, axis_color, 1, cv2.LINE_AA)

            # Y ticks in spectrum value units.
            for i in range(5):
                t = float(i) / 4.0
                py = int(round(y0 - t * (y0 - y1)))
                cv2.line(panel, (x0 - 5, py), (x0, py), tick_color, 1, cv2.LINE_AA)
                val = y_min + t * (y_max - y_min)
                cv2.putText(panel, f"{val:.2f}", (6, py + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.38, axis_color, 1, cv2.LINE_AA)

        if has_spec:
            pts: list[tuple[int, int]] = []
            for xv, yv in zip(xs, ys):
                px = int(round(x0 + (float(xv) - x_min) / (x_max - x_min) * (x1 - x0)))
                py = int(round(y0 - (float(yv) - y_min) / (y_max - y_min) * (y0 - y1)))
                pts.append((px, py))
            if cv2 is not None and len(pts) >= 2:
                cv2.polylines(panel, [np.asarray(pts, dtype=np.int32)], False, (70, 220, 120), 2, lineType=cv2.LINE_AA)
        if cv2 is not None and fas_label:
            # panel is kept in RGB for Tk/PIL display, so colors here are RGB.
            fas_color = (70, 220, 120) if bool(fas_is_live) else (230, 70, 60)
            fx = max(12, x0)
            fy = min(h - 10, y0 + 52)
            cv2.putText(panel, str(fas_label), (fx, fy), cv2.FONT_HERSHEY_SIMPLEX, 0.72, fas_color, 2, cv2.LINE_AA)

        combined = np.hstack([rgb, panel])
        combined = np.ascontiguousarray(combined)
        self._analysis_image_width = int(w)
        return combined

    def _on_analysis_canvas_click(self, event: tk.Event) -> None:
        if not self._analysis_mode or self._analysis_hsi_hwc is None:
            return
        xy = self._main_canvas_to_image(float(event.x), float(event.y), clamp=False)
        if xy is None:
            return
        xi = int(round(float(xy[0])))
        yi = int(round(float(xy[1])))
        if xi < 0 or yi < 0 or xi >= int(self._analysis_image_width):
            return
        hsi = self._analysis_hsi_hwc
        h, w, _c = hsi.shape
        xh = int(round((float(xi) / max(1.0, float(self._analysis_image_width - 1))) * float(max(1, w - 1))))
        yh = int(round((float(yi) / max(1.0, float(h - 1))) * float(max(1, h - 1))))
        try:
            spectrum = self._spectrum_from_point(hsi, xh, yh, half=2)
            self._analysis_pick_xy = (xh, yh)
            self._analysis_spectrum = spectrum
            self._refresh_analysis_view()
            self.status_var.set(f"Point spectrum sampled at ({xh}, {yh})")
        except Exception as exc:
            self.status_var.set(f"Spectrum sampling failed: {exc}")

    def _draw_analysis_to_main_canvas(self) -> None:
        if self._analysis_view_rgb is None:
            return
        rgb = self._analysis_view_rgb
        photo, m = self._fit_rgb_for_canvas(rgb, self.main_canvas, nearest=True)
        self._main_photo = photo
        self._main_display_map = m
        self._main_rgb_shape = (int(rgb.shape[0]), int(rgb.shape[1]))
        self.main_canvas.delete("all")
        self.main_canvas.create_image(m[0], m[1], anchor="nw", image=self._main_photo)

    def _ordered_wavelength_mapping_for_snapshot(self) -> tuple[list[int], list[float], list[str], str]:
        if not self.wavelength_mapping_entries:
            return [], [], [], "none"
        entries = list(self.wavelength_mapping_entries)
        order_idx: list[int] = []
        order_wl: list[float] = []
        order_labels: list[str] = []
        seen: set[int] = set()
        sortable: list[tuple[float, int, str]] = []
        for e in entries:
            try:
                ch_idx = int(e.get("channel_index", -1))
                wl = float(e.get("wavelength_nm"))
                label = str(e.get("label", ""))
            except Exception:
                continue
            if ch_idx < 0 or ch_idx >= 48:
                continue
            if ch_idx in seen:
                continue
            seen.add(ch_idx)
            sortable.append((wl, ch_idx, label))
        # Snapshot channels must be saved in ascending wavelength order.
        sortable.sort(key=lambda x: (x[0], x[1]))
        for wl, ch_idx, label in sortable:
            order_idx.append(ch_idx)
            order_wl.append(wl)
            order_labels.append(label)
        source = str(self.wavelength_mapping_source or "session_wavelength_mapping") + " (sorted_by_wavelength_asc)"
        return order_idx, order_wl, order_labels, source

    def _snapshot(self) -> None:
        if self.last_frame is None:
            self.status_var.set("No frame for snapshot")
            return
        if self.crop_boxes is None or len(self.crop_boxes) != 16:
            self.status_var.set("Calibration is incomplete: crop data missing")
            return
        if self.geometry_correction_enabled and self.geometry_h is None:
            self.status_var.set("Calibration is incomplete: geometry data missing")
            return
        geometry_applied = bool(self.geometry_correction_enabled and self.geometry_h is not None)
        live_gain_before = float(self.gain_var.get())
        live_exposure_before = float(self.exposure_var.get())
        snapshot_gain = float(live_gain_before)
        snapshot_exposure = float(live_exposure_before)
        try:
            got = self._get_latest_raw()
            if got is None:
                return
            raw_arr, fmt = got
            raw_lenses = self._apply_crop_to_raw(raw_arr)
            if raw_lenses is None:
                self.status_var.set("Invalid crop data")
                return
            corrected = self._apply_dark_flat(raw_lenses)

            boxes_sorted = self._sorted_crop_boxes()
            lens_rgb = np.empty((16, corrected.shape[1], corrected.shape[2], 3), dtype=np.float32)
            for i in range(16):
                pattern_i = None
                if i < len(boxes_sorted):
                    bi = boxes_sorted[i]
                    pattern_i = bayer_pattern_for_crop(fmt, int(bi["x"]), int(bi["y"]))
                rgb_u8 = debayer_menon_rgb(
                    np.clip(corrected[i], 0, 65535).astype(np.uint16),
                    fmt,
                    pattern_override=pattern_i,
                )
                lens_rgb[i] = rgb_u8.astype(np.float32)

            if geometry_applied:
                ref_idx = max(0, min(15, int(self.reference_lens)))
                ref_h = int(lens_rgb[ref_idx].shape[0])
                ref_w = int(lens_rgb[ref_idx].shape[1])
                warped = np.empty((16, ref_h, ref_w, 3), dtype=np.float32)
                for i in range(16):
                    h_mat = self.geometry_h[i]
                    if np.isnan(h_mat).any():
                        warped[i] = (
                            cv2.resize(lens_rgb[i], (ref_w, ref_h), interpolation=cv2.INTER_NEAREST)
                            if cv2 is not None
                            else lens_rgb[i]
                        )
                        continue
                    if cv2 is not None:
                        warped[i] = cv2.warpPerspective(
                            lens_rgb[i], h_mat.astype(np.float32), (ref_w, ref_h), flags=cv2.INTER_LINEAR
                        )
                    else:
                        warped[i] = lens_rgb[i]
            else:
                ref_h = int(lens_rgb[0].shape[0])
                ref_w = int(lens_rgb[0].shape[1])
                warped = lens_rgb.copy()

            cube = np.empty((48, ref_h, ref_w), dtype=np.float32)
            for i in range(16):
                base = i * 3
                cube[base + 0] = warped[i, :, :, 0]
                cube[base + 1] = warped[i, :, :, 1]
                cube[base + 2] = warped[i, :, :, 2]

            mapping_idx, mapping_wl, mapping_labels, mapping_source = self._ordered_wavelength_mapping_for_snapshot()
            cube_out = cube
            mapping_applied = False
            mapping_note = "default_channels_order"
            if mapping_idx:
                cube_out = cube[mapping_idx, :, :]
                mapping_applied = True
                if len(mapping_idx) == 48:
                    mapping_note = "full_48ch_reorder"
                else:
                    mapping_note = f"partial_reorder_{len(mapping_idx)}ch"

            reflectance_saved = False
            reflectance_error: str | None = None
            white_meta: dict[str, object] = {
                "configured": bool(self.white_point_ref is not None),
                "applied": False,
                "lens_index": int(self.white_point_lens) if self.white_point_lens is not None else None,
                "lens_number": (int(self.white_point_lens) + 1) if self.white_point_lens is not None else None,
                "roi_xyxy": [int(v) for v in self.white_point_roi] if self.white_point_roi is not None else None,
                "created_utc": str(self.white_point_created_utc) if self.white_point_created_utc else None,
                "channels": 0,
            }
            if self.white_point_ref is not None:
                try:
                    ref = np.asarray(self.white_point_ref, dtype=np.float32).reshape(-1)
                    if ref.size == 3:
                        ref = np.tile(ref, 16)
                    if ref.size != 48:
                        raise RuntimeError(f"invalid white point size: {ref.size}")
                    white_meta["channels"] = int(ref.size)
                    ref_out = ref[mapping_idx] if mapping_idx else ref
                    if int(ref_out.size) != int(cube_out.shape[0]):
                        raise RuntimeError(
                            f"white point channels ({int(ref_out.size)}) != cube channels ({int(cube_out.shape[0])})"
                        )
                    safe_ref = np.maximum(ref_out.astype(np.float32), 1e-6)[:, None, None]
                    reflectance = cube_out.astype(np.float32) / safe_ref
                    white_meta["applied"] = True
                    reflectance_saved = True
                except Exception as exc:
                    reflectance_error = str(exc)
                    if self.debug:
                        self._dbg(f"reflectance generation skipped: {exc}")

            if self.session_root is None:
                ts = dt.datetime.now().strftime("%Y-%m-%d_%H-%M-%S_%f")
                self.session_root = self.output_dir / f"session_{ts}_snapshot"
                self.session_root.mkdir(parents=True, exist_ok=True)
            self._save_white_point_to_session()
            cap_dir = self.session_root / "captures"
            cap_dir.mkdir(parents=True, exist_ok=True)
            stem = dt.datetime.now().strftime("snapshot_%Y-%m-%d_%H-%M-%S_%f")
            snap_dir = cap_dir / stem
            snap_dir.mkdir(parents=True, exist_ok=True)
            cube_path = snap_dir / "cube.npy"
            np.save(cube_path, cube_out)
            np.save(snap_dir / "raw_full.npy", raw_arr)
            np.save(snap_dir / "raw_lenses.npy", raw_lenses)
            np.save(snap_dir / "corrected_lenses.npy", corrected.astype(np.float32))
            np.save(snap_dir / "rgb_lenses.npy", warped.astype(np.float32))
            if reflectance_saved:
                np.save(snap_dir / "reflectance.npy", reflectance.astype(np.float32))
                # Backward-compatible typo variant requested by some pipelines.
                np.save(snap_dir / "reflactance.npy", reflectance.astype(np.float32))
            if mapping_idx:
                np.save(snap_dir / "wavelengths_nm.npy", np.asarray(mapping_wl, dtype=np.float32))
                (snap_dir / "wavelengths_order.txt").write_text(
                    "\n".join(f"{wl:.6f}" for wl in mapping_wl) + "\n",
                    encoding="utf-8",
                )
                self._save_json(
                    snap_dir / "wavelengths.json",
                    {
                        "order_labels": mapping_labels,
                        "wavelengths_nm": mapping_wl,
                        "channel_indices_from_default_cube": mapping_idx,
                        "source": mapping_source,
                    },
                )
            self._save_json(
                snap_dir / "snapshot.json",
                {
                    "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
                    "format": "cube_48_h_w",
                    "channels_order_default": "[L1_R, L1_G, L1_B, ... L16_B]",
                    "shape": list(cube_out.shape),
                    "dtype": str(cube_out.dtype),
                    "reference_lens": int(self.reference_lens),
                    "geometry_correction_applied": bool(geometry_applied),
                    "geometry_correction_enabled": bool(self.geometry_correction_enabled),
                    "source_pixel_format": fmt,
                    "snapshot_capture_gain_db": snapshot_gain,
                    "snapshot_capture_exposure_us": snapshot_exposure,
                    "live_gain_before_snapshot_db": live_gain_before,
                    "live_exposure_before_snapshot_us": live_exposure_before,
                    "wavelength_mapping": {
                        "applied": mapping_applied,
                        "source": mapping_source if mapping_applied else None,
                        "count": int(len(mapping_idx)),
                        "note": mapping_note,
                        "labels_preview": mapping_labels[:8] if mapping_applied else [],
                    },
                    "white_point": {
                        **white_meta,
                        "error": reflectance_error,
                    },
                    "files": {
                        "cube": "cube.npy",
                        "reflactance": "reflactance.npy" if reflectance_saved else None,
                        "reflectance": "reflectance.npy" if reflectance_saved else None,
                        "raw_full": "raw_full.npy",
                        "raw_lenses": "raw_lenses.npy",
                        "corrected_lenses": "corrected_lenses.npy",
                        "rgb_lenses": "rgb_lenses.npy",
                        "wavelengths_nm": "wavelengths_nm.npy" if mapping_applied else None,
                        "wavelengths_txt": "wavelengths_order.txt" if mapping_applied else None,
                        "wavelengths_json": "wavelengths.json" if mapping_applied else None,
                    },
                },
            )
            if reflectance_saved:
                self.status_var.set(f"Snapshot saved (+reflectance): {snap_dir}")
            elif reflectance_error:
                self.status_var.set(f"Snapshot saved (reflectance skipped): {snap_dir}")
            else:
                self.status_var.set(f"Snapshot saved: {snap_dir}")
        except Exception as exc:
            self.status_var.set(f"Snapshot failed: {exc}")

    # ----------------------------- Render -----------------------------
    def _fit_rgb_for_canvas(self, rgb: "np.ndarray", canvas: tk.Canvas, nearest: bool = True) -> tuple[tk.PhotoImage, tuple[float, float, float]]:
        cw = max(1, int(canvas.winfo_width()))
        ch = max(1, int(canvas.winfo_height()))
        h, w = int(rgb.shape[0]), int(rgb.shape[1])
        scale = min(cw / max(1, w), ch / max(1, h))
        dw = max(1, int(round(w * scale)))
        dh = max(1, int(round(h * scale)))
        img = Image.fromarray(rgb, mode="RGB")
        resample = Image.Resampling.NEAREST if nearest else Image.Resampling.BILINEAR
        if dw != w or dh != h:
            img = img.resize((dw, dh), resample=resample)
        photo = ImageTk.PhotoImage(img)
        x0 = 0.5 * (cw - dw)
        y0 = 0.5 * (ch - dh)
        return photo, (x0, y0, scale)

    def _draw_crop_overlay(self) -> None:
        if self.current_page != "calib" or self.calib_stage != "crop":
            return
        if self._calib_display_map is None:
            return
        self.calib_canvas.delete("overlay")
        x0, y0, s = self._calib_display_map
        if self.crop_centers is not None:
            for i, pt in enumerate(self.crop_centers):
                cx = x0 + float(pt[0]) * s
                cy = y0 + float(pt[1]) * s
                self.calib_canvas.create_oval(cx - 3, cy - 3, cx + 3, cy + 3, outline="#00ff88", width=2, tags="overlay")
                self.calib_canvas.create_text(cx + 8, cy - 8, text=str(i + 1), fill="#00ff88", anchor="sw", tags="overlay")
        for b in self.crop_boxes:
            rx0 = x0 + float(b["x"]) * s
            ry0 = y0 + float(b["y"]) * s
            rx1 = x0 + float(b["x"] + b["width"]) * s
            ry1 = y0 + float(b["y"] + b["height"]) * s
            self.calib_canvas.create_rectangle(rx0, ry0, rx1, ry1, outline="#ffaa00", width=1, tags="overlay")

    def _queue_render_request(self) -> None:
        if self.last_frame is None:
            return
        self._render_submit_seq += 1
        req = {
            "seq": int(self._render_submit_seq),
            "frame": self.last_frame,
            "page": str(self.current_page),
            "stage": str(self.calib_stage),
            "main_view_mode": str(self.main_view_mode.get()),
            "main_lens": int(self.main_lens_var.get()),
            "crop_boxes": [dict(b) for b in self._sorted_crop_boxes()],
            "flat_norm": self.flat_norm,
            "force_u8_mode": bool(self.force_u8_mode),
        }
        while True:
            try:
                self._render_in_q.get_nowait()
            except queue.Empty:
                break
        try:
            self._render_in_q.put_nowait(req)
        except Exception:
            pass

    @staticmethod
    def _overlay_face_mask_on_rgb(rgb: "np.ndarray", mask: "np.ndarray") -> "np.ndarray":
        if rgb.ndim != 3 or rgb.shape[2] != 3:
            return rgb
        m = mask
        if m.ndim == 3:
            m = m[:, :, 0]
        if m.shape[0] != rgb.shape[0] or m.shape[1] != rgb.shape[1]:
            if cv2 is not None:
                m = cv2.resize(m, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_NEAREST)
            else:
                pil = Image.fromarray(np.asarray(m > 0, dtype=np.uint8) * 255, mode="L")
                pil = pil.resize((rgb.shape[1], rgb.shape[0]), Image.Resampling.NEAREST)
                m = np.asarray(pil, dtype=np.uint8)
        m_bin = m > 127
        if not bool(np.any(m_bin)):
            return rgb
        out = rgb.astype(np.float32, copy=True)
        color = np.array([40.0, 225.0, 90.0], dtype=np.float32)
        alpha = 0.36
        out[m_bin] = out[m_bin] * (1.0 - alpha) + color * alpha
        out_u8 = np.clip(out, 0, 255).astype(np.uint8)
        if cv2 is not None:
            edge = cv2.Canny((m_bin.astype(np.uint8) * 255), 64, 128)
            out_u8[edge > 0] = np.array([255, 140, 40], dtype=np.uint8)
        return out_u8

    @staticmethod
    def _extract_face_mask_from_segmentation_result(
        results: object,
        target_h: int,
        target_w: int,
        torch_mod: object | None,
    ) -> "np.ndarray | None":
        if not isinstance(results, list) or not results:
            return None
        first = results[0]
        if not isinstance(first, dict):
            return None
        masks_obj = first.get("masks")
        if masks_obj is None:
            return None

        def to_numpy_mask(x: object) -> "np.ndarray | None":
            try:
                if torch_mod is not None and hasattr(torch_mod, "is_tensor") and torch_mod.is_tensor(x):
                    arr = x.detach().float().cpu().numpy()
                else:
                    arr = np.asarray(x, dtype=np.float32)
                if arr.ndim == 2:
                    return arr
                return None
            except Exception:
                return None

        masks_list: list[np.ndarray] = []
        if torch_mod is not None and hasattr(torch_mod, "is_tensor") and torch_mod.is_tensor(masks_obj):
            try:
                arr = masks_obj.detach().float().cpu().numpy()
                if arr.ndim == 2:
                    masks_list = [arr]
                elif arr.ndim >= 3:
                    masks_list = [arr[i] for i in range(int(arr.shape[0]))]
            except Exception:
                masks_list = []
        elif isinstance(masks_obj, (list, tuple)):
            for m in masks_obj:
                m_np = to_numpy_mask(m)
                if m_np is not None:
                    masks_list.append(m_np)
        else:
            m_np = to_numpy_mask(masks_obj)
            if m_np is not None:
                masks_list = [m_np]

        if not masks_list:
            return None

        best_mask = None
        best_area = -1
        for m in masks_list:
            m_bin = (m > 0.5).astype(np.uint8)
            area = int(m_bin.sum())
            if area > best_area:
                best_area = area
                best_mask = m_bin
        if best_mask is None or best_area <= 0:
            return None

        if best_mask.shape[0] != target_h or best_mask.shape[1] != target_w:
            if cv2 is not None:
                best_mask = cv2.resize(best_mask, (target_w, target_h), interpolation=cv2.INTER_NEAREST)
            else:
                pil = Image.fromarray((best_mask * 255).astype(np.uint8), mode="L")
                pil = pil.resize((target_w, target_h), Image.Resampling.NEAREST)
                best_mask = (np.asarray(pil, dtype=np.uint8) > 127).astype(np.uint8)
        return (best_mask * 255).astype(np.uint8)

    def _render_worker_loop(self) -> None:
        while not self._render_stop_event.is_set():
            try:
                req = self._render_in_q.get(timeout=0.12)
            except queue.Empty:
                continue
            if req is None:
                continue
            try:
                result = self._build_preview_rgb_for_request(req)
            except Exception as exc:
                result = {
                    "seq": int(req.get("seq", 0)),
                    "page": str(req.get("page", "")),
                    "stage": str(req.get("stage", "")),
                    "error": f"{exc}",
                }
            while True:
                try:
                    self._render_out_q.get_nowait()
                except queue.Empty:
                    break
            try:
                self._render_out_q.put_nowait(result)
            except Exception:
                pass

    def _build_preview_rgb_for_request(self, req: dict[str, object]) -> dict[str, object]:
        frame = req.get("frame")
        if not isinstance(frame, FramePacket):
            return {"seq": int(req.get("seq", 0)), "page": str(req.get("page", "")), "stage": str(req.get("stage", ""))}
        meta = frame.meta or {}
        fmt = meta.get("pixel_format_name") or meta.get("pixel_format") or frame.pixel_format
        fmt_name = pixel_format_to_name(fmt)
        raw_arr = decode_buffer_to_ndarray(frame.raw, frame.width, frame.height, fmt)
        if bool(req.get("force_u8_mode", True)) and raw_arr.dtype != np.uint8:
            raw_arr = raw_to_u8(raw_arr, fmt_name, autostretch=False)

        page = str(req.get("page", ""))
        stage = str(req.get("stage", ""))
        rgb: np.ndarray | None = None
        frame_id = int(meta.get("frame_id", -1))
        crop_boxes = req.get("crop_boxes") if isinstance(req.get("crop_boxes"), list) else []
        boxes_sorted = sorted(crop_boxes, key=lambda z: int(z["index"])) if crop_boxes else []
        flat_norm = req.get("flat_norm")

        def _apply_dark_flat_req(raw_lenses: "np.ndarray") -> "np.ndarray":
            work = raw_lenses.astype(np.float32)
            if isinstance(flat_norm, np.ndarray) and flat_norm.shape == raw_lenses.shape:
                safe = np.where(np.abs(flat_norm) < 1e-6, 1e-6, flat_norm)
                work = work / safe
            return work

        if page == "calib":
            if stage == "crop":
                rgb = self._preview_rgb_main_fast(raw_arr, fmt_name, autostretch=False)
            else:
                lenses = self._apply_crop_to_raw_boxes(raw_arr, boxes_sorted)
                if lenses is None:
                    rgb = self._preview_rgb_main_fast(raw_arr, fmt_name, autostretch=False)
                else:
                    # In calibration, use raw for crop stage and dark/flat corrected data for the rest.
                    work = _apply_dark_flat_req(lenses)
                    rgb_lenses = np.empty((16, work.shape[1], work.shape[2], 3), dtype=np.uint8)
                    for i in range(16):
                        pattern_i = None
                        if i < len(boxes_sorted):
                            bi = boxes_sorted[i]
                            pattern_i = bayer_pattern_for_crop(fmt_name, int(bi["x"]), int(bi["y"]))
                        rgb_lenses[i] = self._preview_rgb_main_fast(
                            np.clip(work[i], 0, 65535).astype(np.uint16),
                            fmt_name,
                            autostretch=False,
                            pattern_override=pattern_i,
                        )
                    rgb = compose_grid16_rgb(rgb_lenses, gap=2)
        elif page == "main":
            lenses = self._apply_crop_to_raw_boxes(raw_arr, boxes_sorted)
            if lenses is None:
                rgb = self._preview_rgb_main_fast(raw_arr, fmt_name)
            else:
                work = _apply_dark_flat_req(lenses)
                rgb_lenses = np.empty((16, work.shape[1], work.shape[2], 3), dtype=np.uint8)
                for i in range(16):
                    pattern_i = None
                    if i < len(boxes_sorted):
                        bi = boxes_sorted[i]
                        pattern_i = bayer_pattern_for_crop(fmt_name, int(bi["x"]), int(bi["y"]))
                    rgb_lenses[i] = self._preview_rgb_main_fast(
                        np.clip(work[i], 0, 65535).astype(np.uint16),
                        fmt_name,
                        pattern_override=pattern_i,
                    )
                if str(req.get("main_view_mode", "grid")) == "single":
                    idx = max(1, min(16, int(req.get("main_lens", 1)))) - 1
                    rgb = rgb_lenses[idx]
                else:
                    rgb = compose_grid16_rgb(rgb_lenses, gap=2)

        dark_hint = None
        try:
            raw_max_hint = float(raw_arr.max()) if raw_arr.size else 0.0
            if page == "calib" and stage == "crop" and raw_max_hint <= 4.0:
                dark_hint = "Crop preview is very dark (RAW max<=4). Increase Exposure/Gain."
        except Exception:
            pass

        return {
            "seq": int(req.get("seq", 0)),
            "page": page,
            "stage": stage,
            "rgb": rgb,
            "fmt": fmt_name,
            "raw_dtype": str(raw_arr.dtype),
            "raw_shape": tuple(raw_arr.shape),
            "raw_min": float(raw_arr.min()) if raw_arr.size else 0.0,
            "raw_max": float(raw_arr.max()) if raw_arr.size else 0.0,
            "dark_hint": dark_hint,
        }

    def _drain_render_results(self) -> None:
        if self._analysis_mode and self.current_page == "main":
            while True:
                try:
                    self._render_out_q.get_nowait()
                except queue.Empty:
                    break
            return
        latest: dict[str, object] | None = None
        while True:
            try:
                latest = self._render_out_q.get_nowait()
            except queue.Empty:
                break
        if latest is None:
            return
        seq = int(latest.get("seq", 0))
        if seq < self._render_applied_seq:
            return
        self._render_applied_seq = seq

        err = latest.get("error")
        if err:
            self.status_var.set(f"Preview render error: {err}")
            return
        page = str(latest.get("page", ""))
        stage = str(latest.get("stage", ""))
        if page != self.current_page:
            return
        if page == "calib" and stage != self.calib_stage:
            return
        rgb = latest.get("rgb")
        if not isinstance(rgb, np.ndarray):
            return

        if page == "calib":
            photo, m = self._fit_rgb_for_canvas(rgb, self.calib_canvas, nearest=True)
            self._calib_photo = photo
            self._calib_display_map = m
            self.calib_canvas.delete("all")
            self.calib_canvas.create_image(m[0], m[1], anchor="nw", image=self._calib_photo)
            self._draw_crop_overlay()
        elif page == "main":
            photo, m = self._fit_rgb_for_canvas(rgb, self.main_canvas, nearest=True)
            self._main_photo = photo
            self._main_display_map = m
            self._main_rgb_shape = (int(rgb.shape[0]), int(rgb.shape[1]))
            self.main_canvas.delete("all")
            self.main_canvas.create_image(m[0], m[1], anchor="nw", image=self._main_photo)
            self._draw_main_overlay()

        hint = latest.get("dark_hint")
        if hint and (time.monotonic() - self._dark_preview_hint_ts) >= 2.0:
            self._dark_preview_hint_ts = time.monotonic()
            self.status_var.set(str(hint))

        if self.debug:
            now_log = time.monotonic()
            if (now_log - self._debug_last_render_log_ts) >= 1.0:
                self._debug_last_render_log_ts = now_log
                self._dbg(
                    f"render: page={page} stage={stage} fmt={latest.get('fmt')} "
                    f"dtype={latest.get('raw_dtype')} shape={latest.get('raw_shape')} "
                    f"raw_min={float(latest.get('raw_min', 0.0)):.2f} raw_max={float(latest.get('raw_max', 0.0)):.2f}"
                )
            try:
                rgb_min = int(rgb.min())
                rgb_max = int(rgb.max())
                if rgb_max <= 1:
                    self._dbg(f"render warning: rgb appears dark/flat (min={rgb_min}, max={rgb_max})")
            except Exception:
                pass

    def _render_preview(self, force: bool = False) -> None:
        if self._analysis_mode:
            return
        if self.last_frame is None:
            return
        frame_id = int((self.last_frame.meta or {}).get("frame_id", -1))
        if (not force) and frame_id >= 0 and frame_id == self._last_rendered_frame_id:
            return
        now = time.monotonic()
        render_interval = self.render_interval_s
        if self.current_page == "calib" and self.calib_stage == "crop":
            render_interval = max(render_interval, 0.10)
        if (not force) and (now - self.last_render_ts) < render_interval:
            return
        self.last_render_ts = now
        if frame_id >= 0:
            self._last_rendered_frame_id = frame_id
        self._queue_render_request()

    # ----------------------------- Crop manual drag -----------------------------
    def _canvas_to_raw(self, x: float, y: float) -> tuple[float, float] | None:
        m = self._calib_display_map
        if m is None:
            return None
        ox, oy, s = m
        if s <= 0:
            return None
        return (x - ox) / s, (y - oy) / s

    def _on_calib_canvas_press(self, event: tk.Event) -> None:
        if self.calib_stage != "crop" or not self.manual_points_var.get() or self.crop_centers is None:
            return
        xy = self._canvas_to_raw(float(event.x), float(event.y))
        if xy is None:
            return
        x, y = xy
        d2 = ((self.crop_centers[:, 0] - x) ** 2 + (self.crop_centers[:, 1] - y) ** 2).astype(np.float32)
        idx = int(np.argmin(d2))
        if float(d2[idx]) <= 900.0:  # 30 px radius
            self._drag_center_idx = idx

    def _on_calib_canvas_drag(self, event: tk.Event) -> None:
        if self._drag_center_idx is None or self.crop_centers is None or self.last_frame is None:
            return
        xy = self._canvas_to_raw(float(event.x), float(event.y))
        if xy is None:
            return
        x = max(0.0, min(float(self.last_frame.width - 1), float(xy[0])))
        y = max(0.0, min(float(self.last_frame.height - 1), float(xy[1])))
        self.crop_centers[self._drag_center_idx, 0] = x
        self.crop_centers[self._drag_center_idx, 1] = y
        pts = self.crop_centers[np.argsort(self.crop_centers[:, 1])]
        rows = []
        for r in range(4):
            row = pts[r * 4 : (r + 1) * 4]
            row = row[np.argsort(row[:, 0])]
            rows.append(row)
        self.crop_centers = np.vstack(rows).astype(np.float32)
        self._on_crop_offsets_changed()

    def _on_calib_canvas_release(self, _event: tk.Event) -> None:
        self._drag_center_idx = None

    # ----------------------------- Event loop -----------------------------
    def _push_ui_event(self, kind: str, payload: object | None = None) -> None:
        try:
            self.event_q.put_nowait((kind, payload))
        except Exception:
            pass

    def _poll_events(self) -> None:
        handled = 0
        frame_updated = False
        while handled < 24:
            try:
                kind, payload = self.event_q.get_nowait()
            except queue.Empty:
                break
            handled += 1
            if kind == "status":
                self.status_var.set(str(payload))
            elif kind == "zaber_status":
                self.zaber_status_var.set(str(payload))
            elif kind == "zaber_pos":
                try:
                    pos = float(payload)
                    self._zaber_last_pos_mm = pos
                    self.zaber_pos_mm_var.set(f"{pos:.4f}")
                except Exception:
                    self.zaber_pos_mm_var.set(str(payload))
            elif kind == "auto_fix_set_ip":
                self.camera_var.set(str(payload))
            elif kind == "auto_fix_done":
                self.auto_fix_running = False
            elif kind == "camera_scan_finish":
                self.camera_scan_running = False
            elif kind == "camera_scan_done":
                payload_d = dict(payload) if isinstance(payload, dict) else {}
                ok = bool(payload_d.get("ok"))
                if not ok:
                    self.status_var.set(f"Camera scan failed: {payload_d.get('error', 'unknown error')}")
                else:
                    devices = payload_d.get("devices") if isinstance(payload_d.get("devices"), list) else []
                    disp: list[str] = []
                    mapping: dict[str, str] = {}
                    for item in devices:
                        if not (isinstance(item, (list, tuple)) and len(item) >= 1):
                            continue
                        dev_id = str(item[0])
                        transport = str(item[1]) if len(item) > 1 else "unknown"
                        label = f"{dev_id} ({transport})"
                        disp.append(label)
                        endpoint = transport if is_ipv4_literal(transport) else dev_id
                        mapping[label] = endpoint
                    self.camera_scan_display_to_id = mapping
                    self.camera_pick_combo.configure(values=disp)
                    if disp:
                        self.camera_pick_var.set(disp[0])
                        self.camera_var.set(mapping[disp[0]])
                        self.status_var.set(f"Found {len(disp)} camera(s). Selected: {mapping[disp[0]]}")
                    else:
                        self.camera_pick_var.set("")
                        self.status_var.set("No cameras found by Aravis scan")
            elif kind == "connected":
                self.connected = True
                info = dict(payload) if isinstance(payload, dict) else {}
                self.camera_info = info
                if self.debug:
                    self._dbg(f"connected event: {json.dumps(info, ensure_ascii=False, default=str)}")
                controls = info.get("controls", {}) if isinstance(info, dict) else {}
                if isinstance(controls, dict):
                    if "gain" in controls:
                        self.gain_var.set(float(controls["gain"]))
                        self.gain_entry_var.set(f"{float(controls['gain']):.3f}")
                    if "exposure" in controls:
                        self.exposure_var.set(float(controls["exposure"]))
                        self.exposure_entry_var.set(f"{float(controls['exposure']):.1f}")
                    if "gain_min" in controls and "gain_max" in controls:
                        self.gain_bounds = (float(controls["gain_min"]), float(controls["gain_max"]))
                    if "exposure_min" in controls and "exposure_max" in controls:
                        self.exposure_bounds = (float(controls["exposure_min"]), float(controls["exposure_max"]))
                vendor = str(info.get("vendor", ""))
                model = str(info.get("model", ""))
                serial = str(info.get("serial", ""))
                pix = str(info.get("pixel_format", ""))
                self.connection_info_var.set(f"{vendor} {model}\nSerial: {serial}\nPixelFormat: {pix}")
                self.status_var.set("Connected")
                self._show_page("choice")
            elif kind == "controls":
                controls = dict(payload) if isinstance(payload, dict) else {}
                if "gain" in controls:
                    self.gain_var.set(float(controls["gain"]))
                    self.gain_entry_var.set(f"{float(controls['gain']):.3f}")
                if "exposure" in controls:
                    self.exposure_var.set(float(controls["exposure"]))
                    self.exposure_entry_var.set(f"{float(controls['exposure']):.1f}")
            elif kind == "frame" and isinstance(payload, FramePacket):
                self.last_frame = payload
                frame_updated = True
                if self.debug:
                    self._debug_frame_count += 1
                    if (self._debug_frame_count % 30) == 1:
                        pmeta = payload.meta or {}
                        self._dbg(
                            f"frame event: #{self._debug_frame_count} "
                            f"w={payload.width} h={payload.height} pf=0x{payload.pixel_format:08x} "
                            f"bytes={len(payload.raw)} frame_id={pmeta.get('frame_id')} "
                            f"status={pmeta.get('status_int')}"
                        )
                self._consume_dark_frame(payload)
                self._consume_flat_frame(payload)
            elif kind == "geometry_progress":
                if isinstance(payload, dict):
                    self.geometry_progress_var.set(float(payload.get("value", 0.0)))
                    self.geometry_progress_text_var.set(str(payload.get("text", "")))
            elif kind == "geometry_done":
                self.geometry_progress_var.set(100.0)
                self.geometry_progress_text_var.set("Done")
                self.status_var.set(f"Geometry calibration complete. Reference lens: {self.reference_lens + 1}")
                self._show_page("main")
            elif kind == "face_seg_status":
                msg = str(payload)
                self.face_seg_status_var.set(msg)
                self._update_analyze_button_state()
            elif kind == "face_cls_status":
                msg = str(payload)
                cur = str(self.face_seg_status_var.get()).strip()
                if cur:
                    self.face_seg_status_var.set(f"{cur}\n{msg}")
                else:
                    self.face_seg_status_var.set(msg)
                self._update_analyze_button_state()
            elif kind == "analysis_state":
                payload_d = dict(payload) if isinstance(payload, dict) else {}
                if "busy" in payload_d:
                    self._analysis_busy = bool(payload_d.get("busy"))
                self._update_analyze_button_state()
            elif kind == "analysis_done":
                payload_d = dict(payload) if isinstance(payload, dict) else {}
                self._analysis_mode = True
                self._analysis_busy = False
                self._analysis_rotation_quarters = 0
                self._analysis_base_rgb = payload_d.get("base_rgb") if isinstance(payload_d.get("base_rgb"), np.ndarray) else None
                self._analysis_view_rgb = payload_d.get("view_rgb") if isinstance(payload_d.get("view_rgb"), np.ndarray) else None
                self._analysis_face_mask = payload_d.get("mask") if isinstance(payload_d.get("mask"), np.ndarray) else None
                self._analysis_hsi_hwc = payload_d.get("hsi_hwc") if isinstance(payload_d.get("hsi_hwc"), np.ndarray) else None
                self._analysis_wavelengths_nm = payload_d.get("wavelengths") if isinstance(payload_d.get("wavelengths"), np.ndarray) else None
                self._analysis_spectrum = payload_d.get("spectrum") if isinstance(payload_d.get("spectrum"), np.ndarray) else None
                self._analysis_fas_label = str(payload_d.get("fas_label")) if payload_d.get("fas_label") is not None else None
                self._analysis_fas_is_live = bool(payload_d.get("fas_is_live")) if payload_d.get("fas_is_live") is not None else None
                fas_score = float(payload_d.get("fas_score")) if payload_d.get("fas_score") is not None else None
                run_mode = str(payload_d.get("run_mode", "analysis"))
                self._analysis_pick_xy = None
                self._refresh_analysis_view()
                face_found = bool(payload_d.get("face_found", False))
                if face_found:
                    fas_txt = self._analysis_fas_label or "Face status unavailable"
                    if fas_score is not None:
                        fas_txt = f"{fas_txt} (p_live={fas_score:.3f})"
                    js = str(payload_d.get("face_spectrum_json", "") or "")
                    prefix = "Classification" if run_mode == "classification" else "Analysis"
                    if js:
                        self.status_var.set(f"{prefix} done: {fas_txt}. Spectrum saved: {js}")
                    else:
                        self.status_var.set(f"{prefix} done: {fas_txt}. Showing mean face spectrum")
                else:
                    prefix = "Classification" if run_mode == "classification" else "Analysis"
                    self.status_var.set(f"{prefix} done: face not detected. Click image to sample 5x5 spectrum.")
                self._update_analyze_button_state()
            elif kind == "analysis_error":
                self._analysis_busy = False
                self._analysis_mode = False
                self._analysis_fas_label = None
                self._analysis_fas_is_live = None
                msg = str(payload)
                self.status_var.set(f"Analysis failed: {msg}")
                self.face_seg_status_var.set(f"Analysis failed: {msg}")
                try:
                    messagebox.showerror("Analysis Error", msg)
                except Exception:
                    pass
                self._update_analyze_button_state()
            elif kind == "error":
                self.status_var.set(f"Camera error: {payload}")
            elif kind == "disconnected":
                if self.connected:
                    self.status_var.set("Camera disconnected")
                self.connected = False

        if frame_updated:
            self._render_preview()
        self._drain_render_results()
        self._serial_tick()

        self.after(self.ui_poll_ms, self._poll_events)

    def _snapshot_capture_count(self) -> int:
        if self.session_root is None:
            return 0
        cap_dir = self.session_root / "captures"
        if not cap_dir.exists():
            return 0
        n = 0
        try:
            for p in cap_dir.iterdir():
                if p.is_dir() and p.name.startswith("snapshot_"):
                    n += 1
        except Exception:
            return 0
        return n

    def _latest_snapshot_dir(self) -> Path | None:
        if self.session_root is None:
            return None
        cap_dir = self.session_root / "captures"
        if not cap_dir.exists():
            return None
        best: tuple[float, Path] | None = None
        try:
            for p in cap_dir.iterdir():
                if not (p.is_dir() and p.name.startswith("snapshot_")):
                    continue
                try:
                    mt = p.stat().st_mtime
                except Exception:
                    continue
                if best is None or mt > best[0]:
                    best = (mt, p)
        except Exception:
            return None
        return best[1] if best else None

    def _serial_stop(self) -> None:
        self._serial_running = False
        self._serial_positions_mm = []
        self._serial_idx = 0
        self._serial_wait_deadline = 0.0
        self.zaber_status_var.set("Zaber: serials stopped")

    def _serial_start(self) -> None:
        if self._serial_running:
            self.zaber_status_var.set("Zaber: serials already running")
            return
        if self._zaber_axis is None or self._zaber_conn is None:
            self.zaber_status_var.set("Zaber: not connected")
            return
        if self.last_frame is None:
            self.status_var.set("No frame (need live preview) for serial snapshots")
            return
        cur = self._zaber_last_pos_mm
        if cur is None:
            self.zaber_status_var.set("Zaber: position unknown yet (wait 1s)")
            return
        try:
            step = float(self.zaber_step_mm_var.get())
            target = float(self.zaber_target_mm_var.get())
        except Exception:
            self.zaber_status_var.set("Zaber: invalid step/target")
            return
        step_abs = abs(step)
        if step_abs <= 0:
            self.zaber_status_var.set("Zaber: step must be > 0")
            return

        direction = 1.0 if target >= cur else -1.0
        step_signed = step_abs * direction
        positions: list[float] = [float(cur)]
        v = float(cur)
        for _ in range(200000):
            if (direction > 0 and v + step_signed >= target) or (direction < 0 and v + step_signed <= target):
                break
            v = v + step_signed
            positions.append(float(v))
        if not positions or abs(positions[-1] - target) > 1e-9:
            positions.append(float(target))

        self._serial_running = True
        self._serial_positions_mm = positions
        self._serial_idx = 0
        self._serial_last_capture_count = self._snapshot_capture_count()
        self._serial_wait_deadline = 0.0
        self.zaber_status_var.set(f"Zaber: serials started ({len(positions)} pos)")

    def _serial_tick(self) -> None:
        if not self._serial_running:
            return
        if self._serial_idx >= len(self._serial_positions_mm):
            self._serial_running = False
            self.zaber_status_var.set("Zaber: serials done")
            return
        if self._zaber_axis is None:
            self._serial_stop()
            return

        target = float(self._serial_positions_mm[self._serial_idx])
        cur = self._zaber_last_pos_mm
        if cur is None:
            return

        tol = 0.01  # mm
        now = time.monotonic()
        if self._serial_wait_deadline <= 0.0:
            self._zaber_submit("goto", {"mm": target})
            self._serial_wait_deadline = now + 20.0
            self.zaber_status_var.set(f"Zaber: moving to {target:.4f} mm ({self._serial_idx+1}/{len(self._serial_positions_mm)})")
            return

        if now > self._serial_wait_deadline:
            self._serial_stop()
            self.zaber_status_var.set(f"Zaber: timeout to reach {target:.4f} mm")
            return

        if abs(float(cur) - target) > tol:
            return

        before = self._snapshot_capture_count()
        self._snapshot()
        after = self._snapshot_capture_count()
        ok = after > before
        latest = self._latest_snapshot_dir()
        if latest is not None:
            try:
                ok = ok and (latest / "cube.npy").exists()
            except Exception:
                ok = False
        if not ok:
            self._serial_stop()
            self.status_var.set("Serials aborted: snapshot did not save")
            return

        self._serial_last_capture_count = after
        self._serial_idx += 1
        self._serial_wait_deadline = 0.0

    # ----------------------------- Lifecycle -----------------------------
    def _on_close(self) -> None:
        try:
            with self._recon_worker_lock:
                self._stop_recon_worker_locked()
            self._render_stop_event.set()
            try:
                self._render_in_q.put_nowait(None)
            except Exception:
                pass
            if self._render_thread.is_alive():
                self._render_thread.join(timeout=0.25)
            self._zaber_stop_event.set()
            try:
                self._zaber_cmd_q.put_nowait(("shutdown", None))
            except Exception:
                pass
            if self._zaber_thread.is_alive():
                self._zaber_thread.join(timeout=0.4)
            try:
                self._zaber_shutdown()
            except Exception:
                pass
            self._disconnect()
        finally:
            self.after(120, self.destroy)

    # ----------------------------- Zaber motion -----------------------------
    def _zaber_submit(self, cmd: str, payload: object | None = None) -> None:
        try:
            self._zaber_cmd_q.put_nowait((cmd, payload))
        except Exception:
            self._push_ui_event("zaber_status", "Zaber: command queue full")

    def _zaber_connect(self) -> None:
        self._zaber_submit("connect", {"port": self.zaber_port_var.get().strip(), "axis": int(self.zaber_axis_var.get())})

    def _zaber_disconnect(self) -> None:
        self._zaber_submit("disconnect", None)

    def _zaber_home(self) -> None:
        self._zaber_submit("home", None)

    def _zaber_stop(self) -> None:
        self._zaber_submit("stop", None)

    def _zaber_goto(self) -> None:
        self._zaber_submit("goto", {"mm": float(self.zaber_target_mm_var.get())})

    def _zaber_jog(self, direction: int) -> None:
        step = float(self.zaber_step_mm_var.get())
        self._zaber_submit("jog", {"mm": step * (1.0 if direction >= 0 else -1.0)})

    def _zaber_shutdown(self) -> None:
        try:
            if self._zaber_conn is not None:
                self._zaber_conn.close()
        finally:
            self._zaber_conn = None
            self._zaber_axis = None

    def _zaber_worker_loop(self) -> None:
        last_pos_poll = 0.0
        while not self._zaber_stop_event.is_set():
            now = time.time()
            try:
                cmd, payload = self._zaber_cmd_q.get(timeout=0.1)
            except queue.Empty:
                cmd, payload = None, None

            if cmd == "shutdown":
                break

            if cmd is not None:
                try:
                    if cmd == "connect":
                        pd = dict(payload) if isinstance(payload, dict) else {}
                        port = str(pd.get("port", "")).strip()
                        axis_idx = int(pd.get("axis", 1))
                        if not port:
                            self._push_ui_event("zaber_status", "Zaber: empty port")
                            continue
                        self._zaber_shutdown()
                        self._push_ui_event("zaber_status", f"Zaber: connecting to {port}...")
                        conn = Connection.open_serial_port(port)
                        devs = conn.detect_devices()
                        if not devs:
                            conn.close()
                            raise RuntimeError("No devices detected")
                        dev = devs[0]
                        axis = dev.get_axis(axis_idx)
                        self._zaber_conn = conn
                        self._zaber_axis = axis
                        self._push_ui_event("zaber_status", f"Zaber: connected ({dev.identity.device_name})")
                    elif cmd == "disconnect":
                        self._zaber_shutdown()
                        self._push_ui_event("zaber_status", "Zaber: disconnected")
                    else:
                        axis = self._zaber_axis
                        if axis is None:
                            self._push_ui_event("zaber_status", "Zaber: not connected")
                            continue
                        if cmd == "home":
                            axis.home()
                            self._push_ui_event("zaber_status", "Zaber: home started")
                        elif cmd == "stop":
                            try:
                                axis.stop()
                            finally:
                                self._push_ui_event("zaber_status", "Zaber: stop")
                        elif cmd == "goto":
                            pd = dict(payload) if isinstance(payload, dict) else {}
                            mm = float(pd.get("mm", 0.0))
                            axis.move_absolute(mm, Units.LENGTH_MILLIMETRES)
                            self._push_ui_event("zaber_status", f"Zaber: goto {mm:g} mm")
                        elif cmd == "jog":
                            pd = dict(payload) if isinstance(payload, dict) else {}
                            mm = float(pd.get("mm", 0.0))
                            axis.move_relative(mm, Units.LENGTH_MILLIMETRES)
                            self._push_ui_event("zaber_status", f"Zaber: jog {mm:g} mm")
                except Exception as exc:
                    self._push_ui_event("zaber_status", f"Zaber error: {exc}")

            if self._zaber_axis is not None and (now - last_pos_poll) >= 0.25:
                try:
                    pos = float(self._zaber_axis.get_position(Units.LENGTH_MILLIMETRES))
                    self._push_ui_event("zaber_pos", pos)
                except Exception:
                    pass
                last_pos_poll = now


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Hydra Baumer wizard GUI")
    parser.add_argument("--interface", default="en10", help="GigE interface name")
    parser.add_argument("--camera", default="", help="Camera IP or Aravis device id")
    parser.add_argument("--output-dir", default="capture", help="Output directory for sessions and snapshots")
    parser.add_argument("--packet-size", type=int, default=DEFAULT_PACKET_SIZE, help="GevSCPSPacketSize")
    parser.add_argument("--packet-delay", type=int, default=DEFAULT_PACKET_DELAY, help="GevSCPD")
    parser.add_argument("--preview-fps", type=float, default=DEFAULT_PREVIEW_FPS, help="Target preview FPS")
    parser.add_argument("--ui-poll-ms", type=int, default=DEFAULT_UI_POLL_MS, help="UI poll interval")
    parser.add_argument("--debug", action="store_true", help="Enable verbose terminal debug prints")
    u8_group = parser.add_mutually_exclusive_group()
    u8_group.add_argument("--force-u8", dest="force_u8", action="store_true", help="Force uint8 processing in GUI pipeline")
    u8_group.add_argument("--no-force-u8", dest="force_u8", action="store_false", help="Keep source bit depth in GUI pipeline")
    parser.set_defaults(force_u8=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.debug:
        print(
            f"[hydra-debug] launch args: interface={args.interface} camera={args.camera} "
            f"output_dir={args.output_dir} preview_fps={args.preview_fps} "
            f"force_u8={args.force_u8}",
            flush=True,
        )
    app = HydraWizardApp(
        interface=args.interface,
        camera_ip=args.camera,
        output_dir=Path(args.output_dir),
        packet_size=args.packet_size,
        packet_delay=args.packet_delay,
        preview_fps=args.preview_fps,
        ui_poll_ms=args.ui_poll_ms,
        debug=bool(args.debug),
        force_u8_mode=bool(args.force_u8),
    )
    app.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
