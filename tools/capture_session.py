#!/usr/bin/env python3
"""Session-oriented scientific RAW frame storage."""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import threading
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from capture_profiles import CaptureProfile, profile_to_dict
from raw_decode import make_preview_from_raw, pixel_format_to_name, save_preview_png

APP_VERSION = "scientific-capture-v1"


@dataclass
class SessionPaths:
    root: Path
    frames: Path
    logs: Path


class SessionWriter:
    """Writes scientific capture sessions with per-frame metadata."""

    def __init__(
        self,
        base_dir: Path,
        profile: CaptureProfile,
        camera_metadata: dict[str, object],
        notes: str = "",
    ) -> None:
        self.base_dir = base_dir.expanduser()
        self.profile = profile
        self.camera_metadata = dict(camera_metadata)
        self.notes = notes.strip()
        self._lock = threading.Lock()
        self._warnings: list[str] = []
        self._frame_files: list[str] = []

        ts = dt.datetime.now().strftime("%Y-%m-%d_%H-%M-%S_%f")
        root = self.base_dir / f"session_{ts}"
        self.paths = SessionPaths(root=root, frames=root / "frames", logs=root / "logs")
        self.paths.frames.mkdir(parents=True, exist_ok=True)
        self.paths.logs.mkdir(parents=True, exist_ok=True)

        self._session_data: dict[str, object] = {
            "app_version": APP_VERSION,
            "session_start_timestamp_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            "selected_capture_profile": profile_to_dict(profile),
            "camera": self.camera_metadata,
            "notes": self.notes,
            "pixel_format": camera_metadata.get("pixel_format", profile.pixel_format),
            "width": camera_metadata.get("width"),
            "height": camera_metadata.get("height"),
            "applied_camera_settings": [],
            "unavailable_settings": [],
            "warnings": self._warnings,
            "frames": self._frame_files,
            "hydra_placeholders": {
                "mapping_version": None,
                "spectral_wavelengths": None,
                "offline_pipeline_version": None,
            },
        }
        self._write_session_json()

    @property
    def root_dir(self) -> Path:
        return self.paths.root

    def _write_session_json(self) -> None:
        session_json = self.paths.root / "session.json"
        session_json.write_text(json.dumps(self._session_data, indent=2, ensure_ascii=False), encoding="utf-8")

    def set_configuration_result(self, result: dict[str, object]) -> None:
        with self._lock:
            self._session_data["camera_configuration"] = result
            self._session_data["pixel_format"] = result.get("selected_pixel_format", self._session_data.get("pixel_format"))
            self._write_session_json()

    def add_warning(self, text: str) -> None:
        with self._lock:
            self._warnings.append(text)
            log_path = self.paths.logs / "warnings.log"
            with log_path.open("a", encoding="utf-8") as f:
                f.write(text.rstrip() + "\n")
            self._write_session_json()

    def write_frame(
        self,
        frame_index: int,
        raw_array: np.ndarray,
        raw_bytes: bytes,
        frame_metadata: dict[str, object],
        save_preview: bool,
    ) -> Path:
        stem = f"frame_{frame_index:06d}"
        npy_path = self.paths.frames / f"{stem}.npy"
        json_path = self.paths.frames / f"{stem}.json"
        np.save(npy_path, raw_array)

        px_fmt = pixel_format_to_name(frame_metadata.get("pixel_format", "unknown"))
        info = {
            "frame_index": frame_index,
            "host_timestamp_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            "camera_timestamp": frame_metadata.get("camera_timestamp"),
            "frame_id": frame_metadata.get("frame_id"),
            "width": int(frame_metadata.get("width", raw_array.shape[1] if raw_array.ndim == 2 else 0)),
            "height": int(frame_metadata.get("height", raw_array.shape[0] if raw_array.ndim == 2 else 0)),
            "pixel_format": px_fmt,
            "pixel_format_int": frame_metadata.get("pixel_format_int"),
            "dtype": str(raw_array.dtype),
            "shape": list(raw_array.shape),
            "raw_byte_length": len(raw_bytes),
            "exposure_us": frame_metadata.get("exposure_us"),
            "gain_db": frame_metadata.get("gain_db"),
            "black_level": frame_metadata.get("black_level"),
            "status_int": frame_metadata.get("status_int"),
            "dropped_frame": bool(frame_metadata.get("dropped_frame", False)),
            "sha256": hashlib.sha256(raw_bytes).hexdigest(),
        }

        preview_name = None
        if save_preview:
            try:
                rgb = make_preview_from_raw(raw_array, px_fmt)
                preview_path = self.paths.frames / f"{stem}_preview.png"
                save_preview_png(preview_path, rgb)
                preview_name = preview_path.name
            except Exception as exc:
                self.add_warning(f"{stem}: preview export failed: {exc}")

        if preview_name:
            info["preview_file"] = preview_name

        json_path.write_text(json.dumps(info, indent=2, ensure_ascii=False), encoding="utf-8")

        with self._lock:
            self._frame_files.append(npy_path.name)
            self._session_data["frames_count"] = int(self._session_data.get("frames_count", 0)) + 1
            self._write_session_json()

        return npy_path

    def finalize(self) -> None:
        with self._lock:
            self._session_data["session_end_timestamp_utc"] = dt.datetime.now(dt.timezone.utc).isoformat()
            self._write_session_json()

