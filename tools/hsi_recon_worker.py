#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np


def _emit(msg: dict[str, object]) -> None:
    sys.stdout.write(json.dumps(msg, ensure_ascii=False) + "\n")
    sys.stdout.flush()


def _load_infer_module(infer_py: Path):
    infer_root = str(infer_py.parent)
    if infer_root not in sys.path:
        # infer.py expects local package imports like "from hsirestore ...".
        sys.path.insert(0, infer_root)
    spec = importlib.util.spec_from_file_location("hsi_restore_infer_runtime", str(infer_py))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import infer module from {infer_py}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Persistent HSI reconstruction worker")
    p.add_argument("--infer-py", required=True)
    p.add_argument("--config", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--device", default="auto")
    p.add_argument("--precision", default="fp16", choices=["fp16", "fp32"])
    p.add_argument("--tile-size", type=int, default=256)
    p.add_argument("--tile-overlap", type=int, default=32)
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    infer_py = Path(args.infer_py).expanduser().resolve()
    config_path = Path(args.config).expanduser().resolve()
    checkpoint_path = Path(args.checkpoint).expanduser().resolve()

    if not infer_py.exists():
        _emit({"type": "ready", "ok": False, "error": f"infer.py not found: {infer_py}"})
        return 2
    if not config_path.exists():
        _emit({"type": "ready", "ok": False, "error": f"config not found: {config_path}"})
        return 2
    if not checkpoint_path.exists():
        _emit({"type": "ready", "ok": False, "error": f"checkpoint not found: {checkpoint_path}"})
        return 2

    try:
        infer_mod = _load_infer_module(infer_py)
        cfg = infer_mod.load_config(str(config_path))
        device = infer_mod.choose_device(str(args.device))
        model = infer_mod.load_model(cfg, checkpoint_path, device)
        use_fp16 = bool(str(args.precision).lower() == "fp16")
        if use_fp16 and str(device.type) != "cuda":
            use_fp16 = False
        expected_channels = int(cfg.get("model", {}).get("in_channels", 44))
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
        s = line.strip()
        if not s:
            continue
        req_id = -1
        try:
            req = json.loads(s)
            req_id = int(req.get("id", -1))
            cmd = str(req.get("cmd", ""))
            if cmd == "shutdown":
                _emit({"type": "result", "id": req_id, "ok": True, "shutdown": True})
                return 0
            if cmd != "infer":
                raise RuntimeError(f"Unknown command: {cmd}")

            input_path = Path(str(req.get("input", ""))).expanduser().resolve()
            output_path = Path(str(req.get("output", ""))).expanduser().resolve()
            if not input_path.exists():
                raise RuntimeError(f"Input reflectance missing: {input_path}")

            cube, band_axis = infer_mod.load_cube(input_path, expected_channels=expected_channels)
            t0 = time.perf_counter()
            pred_hwc, _gate = infer_mod.predict_cube(
                model=model,
                source_hwc=cube,
                device=device,
                tile_size=int(args.tile_size),
                tile_overlap=int(args.tile_overlap),
                use_fp16=bool(use_fp16),
                return_gate=False,
            )
            infer_ms = (time.perf_counter() - t0) * 1000.0
            infer_mod.save_cube(output_path, pred_hwc, band_axis)
            _emit(
                {
                    "type": "result",
                    "id": req_id,
                    "ok": True,
                    "output": str(output_path),
                    "shape_hwc": [int(pred_hwc.shape[0]), int(pred_hwc.shape[1]), int(pred_hwc.shape[2])],
                    "infer_ms": round(float(infer_ms), 2),
                }
            )
        except Exception as exc:
            _emit({"type": "result", "id": req_id, "ok": False, "error": str(exc)})

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
