#!/usr/bin/env python3
"""Чтение метаданных камеры и кадров для Hydra GUI."""

from __future__ import annotations


def _safe_get_node(camera, node: str) -> tuple[bool, object | None, str]:
    for getter in ("get_float", "get_integer", "get_boolean", "get_string"):
        try:
            fn = getattr(camera, getter)
            return True, fn(node), ""
        except Exception:
            continue
    return False, None, f"Node {node} is unavailable"


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

    for node in (
        "Gamma",
        "GammaEnable",
        "BalanceWhiteAuto",
        "ColorTransformationEnable",
        "AcquisitionFrameRateEnable",
        "AcquisitionFrameRate",
        "BlackLevel",
    ):
        ok, value, _error = _safe_get_node(camera, node)
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
