#!/usr/bin/env python3
"""Постоянный worker встроенной HSI-реконструкции."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from hsi_restore_runtime import (
    choose_device,
    load_config,
    load_cube,
    load_model,
    predict_cube,
    save_cube,
)


def _emit(message: dict[str, object]) -> None:
    sys.stdout.write(json.dumps(message, ensure_ascii=False) + "\n")
    sys.stdout.flush()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Persistent HSI reconstruction worker")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--precision", default="fp16", choices=["fp16", "fp32"])
    parser.add_argument("--tile-size", type=int, default=256)
    parser.add_argument("--tile-overlap", type=int, default=32)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    config_path = Path(args.config).expanduser().resolve()
    checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    if not config_path.exists():
        _emit({"type": "ready", "ok": False, "error": f"config not found: {config_path}"})
        return 2
    if not checkpoint_path.exists():
        _emit({"type": "ready", "ok": False, "error": f"checkpoint not found: {checkpoint_path}"})
        return 2

    try:
        config = load_config(config_path)
        device = choose_device(str(args.device))
        model = load_model(config, checkpoint_path, device)
        use_fp16 = bool(str(args.precision).lower() == "fp16" and device.type == "cuda")
        expected_channels = int(config.get("model", {}).get("in_channels", 44))
        _emit(
            {
                "type": "ready",
                "ok": True,
                "device": str(device),
                "precision": "fp16" if use_fp16 else "fp32",
                "expected_channels": expected_channels,
            }
        )
    except Exception as exc:
        _emit({"type": "ready", "ok": False, "error": str(exc)})
        return 3

    for line in sys.stdin:
        request_text = line.strip()
        if not request_text:
            continue
        request_id = -1
        try:
            request = json.loads(request_text)
            request_id = int(request.get("id", -1))
            command = str(request.get("cmd", ""))
            if command == "shutdown":
                _emit({"type": "result", "id": request_id, "ok": True, "shutdown": True})
                return 0
            if command != "infer":
                raise RuntimeError(f"Unknown command: {command}")

            input_path = Path(str(request.get("input", ""))).expanduser().resolve()
            output_path = Path(str(request.get("output", ""))).expanduser().resolve()
            if not input_path.exists():
                raise RuntimeError(f"Input reflectance missing: {input_path}")

            cube, band_axis = load_cube(input_path, expected_channels=expected_channels)
            started = time.perf_counter()
            prediction_hwc, _gate = predict_cube(
                model=model,
                source_hwc=cube,
                device=device,
                tile_size=int(args.tile_size),
                tile_overlap=int(args.tile_overlap),
                use_fp16=use_fp16,
                return_gate=False,
            )
            inference_ms = (time.perf_counter() - started) * 1000.0
            save_cube(output_path, prediction_hwc, band_axis)
            _emit(
                {
                    "type": "result",
                    "id": request_id,
                    "ok": True,
                    "output": str(output_path),
                    "shape_hwc": [
                        int(prediction_hwc.shape[0]),
                        int(prediction_hwc.shape[1]),
                        int(prediction_hwc.shape[2]),
                    ],
                    "infer_ms": round(float(inference_ms), 2),
                }
            )
        except Exception as exc:
            _emit({"type": "result", "id": request_id, "ok": False, "error": str(exc)})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
