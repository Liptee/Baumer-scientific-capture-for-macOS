#!/usr/bin/env python3
"""Camera control helpers for scientific capture mode."""

from __future__ import annotations

import datetime as dt
from typing import Callable

from capture_profiles import CaptureProfile, profile_to_dict

LogFn = Callable[[str], None]


def _as_node_value(value: object) -> object:
    if isinstance(value, bool):
        return bool(value)
    if isinstance(value, int):
        return int(value)
    if isinstance(value, float):
        return float(value)
    return str(value)


def _safe_set_node(camera, node: str, value: object) -> tuple[bool, str]:
    try:
        typed = _as_node_value(value)
        if isinstance(typed, bool):
            camera.set_boolean(node, typed)
        elif isinstance(typed, int):
            camera.set_integer(node, typed)
        elif isinstance(typed, float):
            camera.set_float(node, typed)
        else:
            camera.set_string(node, str(typed))
        return True, ""
    except Exception as exc:
        return False, str(exc)


def _safe_get_node(camera, node: str) -> tuple[bool, object | None, str]:
    for getter in ("get_float", "get_integer", "get_boolean", "get_string"):
        try:
            fn = getattr(camera, getter)
            return True, fn(node), ""
        except Exception:
            continue
    return False, None, f"Node {node} is unavailable"


def _record_setting(
    applied: list[dict[str, object]],
    unavailable: list[dict[str, object]],
    name: str,
    value: object,
    ok: bool,
    error: str = "",
) -> None:
    target = applied if ok else unavailable
    entry: dict[str, object] = {"name": name, "value": value}
    if error:
        entry["error"] = error
    target.append(entry)


def _try_call(logger: LogFn, text: str) -> None:
    try:
        logger(text)
    except Exception:
        pass


def configure_scientific_mode(camera, profile: CaptureProfile, logger: LogFn | None = None) -> dict[str, object]:
    if logger is None:
        logger = lambda _msg: None

    applied: list[dict[str, object]] = []
    unavailable: list[dict[str, object]] = []
    warnings: list[str] = []

    def add_warn(msg: str) -> None:
        warnings.append(msg)
        _try_call(logger, f"WARNING: {msg}")

    # Pixel format selection with scientific priority.
    fmt_candidates: list[str] = []
    if profile.pixel_format:
        fmt_candidates.append(profile.pixel_format)
    for fallback in ("BayerRG12", "BayerRG8"):
        if fallback not in fmt_candidates:
            fmt_candidates.append(fallback)

    selected_fmt = None
    for fmt in fmt_candidates:
        ok = False
        err = ""
        try:
            camera.set_pixel_format_from_string(fmt)
            ok = True
        except Exception as exc:
            err = str(exc)
        _record_setting(applied, unavailable, "PixelFormat", fmt, ok, err)
        if ok:
            selected_fmt = fmt
            break

    if selected_fmt is None:
        try:
            selected_fmt = str(camera.get_pixel_format_as_string())
            add_warn("Failed to set requested PixelFormat, using current camera value")
        except Exception:
            selected_fmt = "unknown"
            add_warn("PixelFormat is unavailable")

    # Core scientific settings.
    def set_call(name: str, fn, value: object) -> None:
        try:
            fn(value)
            _record_setting(applied, unavailable, name, value, True)
        except Exception as exc:
            _record_setting(applied, unavailable, name, value, False, str(exc))
            add_warn(f"{name}: {exc}")

    # Auto/trigger controls.
    set_call("ExposureAuto", camera.set_exposure_time_auto, 0)
    ok, err = _safe_set_node(camera, "ExposureAuto", "Off")
    _record_setting(applied, unavailable, "ExposureAuto(string)", "Off", ok, err)
    if not ok:
        add_warn(f"ExposureAuto(string): {err}")

    set_call("GainAuto", camera.set_gain_auto, 0)
    ok, err = _safe_set_node(camera, "GainAuto", "Off")
    _record_setting(applied, unavailable, "GainAuto(string)", "Off", ok, err)
    if not ok:
        add_warn(f"GainAuto(string): {err}")

    ok, err = _safe_set_node(camera, "TriggerMode", "Off")
    _record_setting(applied, unavailable, "TriggerMode", "Off", ok, err)
    if not ok:
        add_warn(f"TriggerMode: {err}")

    # Exposure/Gain.
    set_call("ExposureTime", camera.set_exposure_time, float(profile.exposure_us))
    set_call("Gain", camera.set_gain, float(profile.gain_db))

    # Frame rate.
    set_call("AcquisitionFrameRateEnable", camera.set_frame_rate_enable, bool(profile.frame_rate_enable))
    if profile.frame_rate_enable and profile.frame_rate is not None:
        set_call("AcquisitionFrameRate", camera.set_frame_rate, float(profile.frame_rate))

    # Optional black level.
    if profile.black_level is not None:
        try:
            camera.set_black_level(float(profile.black_level))
            _record_setting(applied, unavailable, "BlackLevel", float(profile.black_level), True)
        except Exception as exc:
            _record_setting(applied, unavailable, "BlackLevel", float(profile.black_level), False, str(exc))
            add_warn(f"BlackLevel: {exc}")

    # ISP-like controls (best effort).
    for node, value in (
        ("GammaEnable", False),
        ("Gamma", 1.0),
        ("BalanceWhiteAuto", "Off"),
        ("ColorTransformationEnable", False),
        ("ColorCorrectionEnable", False),
        ("LUTEnable", False),
    ):
        ok, err = _safe_set_node(camera, node, value)
        _record_setting(applied, unavailable, node, value, ok, err)
        if not ok:
            add_warn(f"{node}: {err}")

    current = read_camera_runtime_metadata(camera)
    result = {
        "selected_pixel_format": selected_fmt,
        "applied_settings": applied,
        "unavailable_settings": unavailable,
        "warnings": warnings,
        "runtime": current,
        "profile": profile_to_dict(profile),
        "configured_at": dt.datetime.now(dt.timezone.utc).isoformat(),
    }
    return result


def read_camera_runtime_metadata(camera) -> dict[str, object]:
    out: dict[str, object] = {}
    for key, fn in (
        ("vendor", getattr(camera, "get_vendor_name", None)),
        ("model", getattr(camera, "get_model_name", None)),
        ("serial_number", getattr(camera, "get_device_serial_number", None)),
        ("pixel_format", getattr(camera, "get_pixel_format_as_string", None)),
        ("exposure_us", getattr(camera, "get_exposure_time", None)),
        ("gain_db", getattr(camera, "get_gain", None)),
        ("black_level", getattr(camera, "get_black_level", None)),
        ("frame_rate", getattr(camera, "get_frame_rate", None)),
        ("frame_rate_enable", getattr(camera, "get_frame_rate_enable", None)),
    ):
        if fn is None:
            continue
        try:
            out[key] = fn()
        except Exception:
            pass

    # Optional node reads where Camera wrapper may not expose dedicated methods.
    for node in (
        "Gamma",
        "GammaEnable",
        "BalanceWhiteAuto",
        "ColorTransformationEnable",
        "AcquisitionFrameRateEnable",
        "AcquisitionFrameRate",
        "BlackLevel",
    ):
        ok, value, _err = _safe_get_node(camera, node)
        if ok:
            out[node] = value

    return out


def read_buffer_metadata(buffer) -> dict[str, object]:
    out: dict[str, object] = {}
    for key, fn in (
        ("frame_id", getattr(buffer, "get_frame_id", None)),
        ("camera_timestamp", getattr(buffer, "get_timestamp", None)),
        ("system_timestamp", getattr(buffer, "get_system_timestamp", None)),
        ("has_chunks", getattr(buffer, "has_chunks", None)),
    ):
        if fn is None:
            continue
        try:
            out[key] = fn()
        except Exception:
            pass

    try:
        out["status_int"] = int(buffer.get_status())
    except Exception:
        pass

    try:
        payload_type = buffer.get_payload_type()
        out["payload_type_int"] = int(payload_type)
        out["payload_type"] = str(payload_type)
    except Exception:
        pass

    try:
        out["n_parts"] = int(buffer.get_n_parts())
    except Exception:
        pass

    return out

