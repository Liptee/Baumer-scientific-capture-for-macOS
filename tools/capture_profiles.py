#!/usr/bin/env python3
"""Capture profile definitions shared by GUI and CLI."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace


@dataclass(frozen=True)
class CaptureProfile:
    name: str
    pixel_format: str | None
    exposure_us: float
    gain_db: float
    frame_rate_enable: bool
    frame_rate: float | None
    save_preview: bool
    save_raw: bool
    frames_count: int
    black_level: float | None = None
    notes: str = ""


DEFAULT_CAPTURE_PROFILES: dict[str, CaptureProfile] = {
    "scene_capture": CaptureProfile(
        name="scene_capture",
        pixel_format="BayerRG12",
        exposure_us=40000.0,
        gain_db=0.0,
        frame_rate_enable=False,
        frame_rate=None,
        save_preview=True,
        save_raw=True,
        frames_count=1,
        notes="General scene acquisition for Hydra offline processing.",
    ),
    "dark_frame": CaptureProfile(
        name="dark_frame",
        pixel_format="BayerRG12",
        exposure_us=40000.0,
        gain_db=0.0,
        frame_rate_enable=False,
        frame_rate=None,
        save_preview=False,
        save_raw=True,
        frames_count=16,
        notes="Lens covered; used for dark subtraction / averaging.",
    ),
    "flat_field": CaptureProfile(
        name="flat_field",
        pixel_format="BayerRG12",
        exposure_us=10000.0,
        gain_db=0.0,
        frame_rate_enable=False,
        frame_rate=None,
        save_preview=True,
        save_raw=True,
        frames_count=16,
        notes="Uniform field illumination for per-pixel gain calibration.",
    ),
    "white_reference": CaptureProfile(
        name="white_reference",
        pixel_format="BayerRG12",
        exposure_us=10000.0,
        gain_db=0.0,
        frame_rate_enable=False,
        frame_rate=None,
        save_preview=True,
        save_raw=True,
        frames_count=8,
        notes="Spectralon or equivalent white reference frames.",
    ),
    "chessboard_calibration": CaptureProfile(
        name="chessboard_calibration",
        pixel_format="BayerRG12",
        exposure_us=8000.0,
        gain_db=0.0,
        frame_rate_enable=False,
        frame_rate=None,
        save_preview=True,
        save_raw=True,
        frames_count=8,
        notes="Geometry/calibration target captures.",
    ),
    "burst_repeat": CaptureProfile(
        name="burst_repeat",
        pixel_format="BayerRG12",
        exposure_us=40000.0,
        gain_db=0.0,
        frame_rate_enable=False,
        frame_rate=None,
        save_preview=False,
        save_raw=True,
        frames_count=32,
        notes="Repeatability burst capture of the same scene.",
    ),
}


def list_profile_names() -> list[str]:
    return list(DEFAULT_CAPTURE_PROFILES.keys())


def get_capture_profile(name: str) -> CaptureProfile:
    key = name.strip()
    if key not in DEFAULT_CAPTURE_PROFILES:
        raise KeyError(f"Unknown capture profile: {name}")
    return DEFAULT_CAPTURE_PROFILES[key]


def profile_to_dict(profile: CaptureProfile) -> dict[str, object]:
    return asdict(profile)


def profile_from_dict(data: dict[str, object]) -> CaptureProfile:
    base_name = str(data.get("name", "scene_capture"))
    try:
        base = get_capture_profile(base_name)
    except KeyError:
        base = DEFAULT_CAPTURE_PROFILES["scene_capture"]
    return with_overrides(
        base,
        pixel_format=str(data["pixel_format"]) if data.get("pixel_format") is not None else None,
        exposure_us=float(data["exposure_us"]) if data.get("exposure_us") is not None else base.exposure_us,
        gain_db=float(data["gain_db"]) if data.get("gain_db") is not None else base.gain_db,
        frame_rate_enable=bool(data.get("frame_rate_enable", base.frame_rate_enable)),
        frame_rate=float(data["frame_rate"]) if data.get("frame_rate") is not None else base.frame_rate,
        save_preview=bool(data.get("save_preview", base.save_preview)),
        save_raw=bool(data.get("save_raw", base.save_raw)),
        frames_count=int(data.get("frames_count", base.frames_count)),
        black_level=float(data["black_level"]) if data.get("black_level") is not None else base.black_level,
        notes=str(data.get("notes", base.notes)),
    )


def with_overrides(profile: CaptureProfile, **kwargs: object) -> CaptureProfile:
    return replace(profile, **kwargs)

