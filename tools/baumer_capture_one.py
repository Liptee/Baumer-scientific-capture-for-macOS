#!/usr/bin/env python3
"""
Capture one frame from a GigE Vision camera using Aravis.

Typical usage:
  /opt/homebrew/bin/python3.14 tools/baumer_capture_one.py \
    --camera 169.254.223.111 --interface en10 --out frame.pgm
"""

from __future__ import annotations

import argparse
import fcntl
import json
import socket
import struct
import subprocess
import time
from pathlib import Path

from camera_control import configure_scientific_mode, read_buffer_metadata, read_camera_runtime_metadata
from capture_profiles import get_capture_profile, list_profile_names, with_overrides
from capture_session import SessionWriter
from raw_decode import decode_buffer_to_ndarray, pixel_format_to_name

SIOCGIFADDR = 0xC0206921


def save_pgm(path: Path, width: int, height: int, image_bytes: bytes) -> None:
    header = f"P5\n{width} {height}\n255\n".encode("ascii")
    path.write_bytes(header + image_bytes)


def get_interface_ipv4(interface: str) -> str | None:
    entries = get_interface_ipv4_entries(interface)
    if entries:
        return entries[0][0]
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        packed = struct.pack("256s", interface[:15].encode("ascii", errors="ignore"))
        data = fcntl.ioctl(sock.fileno(), SIOCGIFADDR, packed)
        return socket.inet_ntoa(data[20:24])
    except OSError:
        return None
    finally:
        sock.close()


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


def _decode_ifconfig_netmask(token: str) -> str:
    if token.startswith("0x"):
        try:
            return socket.inet_ntoa(struct.pack(">I", int(token, 16)))
        except Exception:
            return "255.255.255.0"
    return token


def get_interface_ipv4_entries(interface: str) -> list[tuple[str, str]]:
    try:
        out = subprocess.check_output(["ifconfig", interface], text=True, stderr=subprocess.DEVNULL)
    except Exception:
        return []
    entries: list[tuple[str, str]] = []
    for line in out.splitlines():
        parts = line.strip().split()
        if len(parts) < 4 or parts[0] != "inet" or parts[2] != "netmask":
            continue
        ip = parts[1]
        if ip.startswith("127."):
            continue
        entries.append((ip, _decode_ifconfig_netmask(parts[3])))
    return entries


def get_interface_ipv4_for_peer(interface: str, peer_ip: str) -> str | None:
    entries = get_interface_ipv4_entries(interface)
    if not entries:
        return get_interface_ipv4(interface)
    if is_ipv4_literal(peer_ip):
        for ip, mask in entries:
            if same_subnet(ip, peer_ip, mask):
                return ip
    return entries[0][0]


def is_ipv4_literal(value: str) -> bool:
    try:
        socket.inet_aton(value)
        return value.count(".") == 3
    except OSError:
        return False


def open_camera_with_fallback(Aravis, camera_id: str, interface: str) -> tuple[object | None, str]:
    errors: list[str] = []

    try:
        camera = Aravis.Camera.new(camera_id)
        if camera is not None:
            return camera, ""
        errors.append("Aravis.Camera.new returned None")
    except Exception as exc:  # noqa: BLE001
        errors.append(f"Aravis.Camera.new failed: {exc}")

    if not is_ipv4_literal(camera_id):
        return None, "; ".join(errors)

    try:
        from gi.repository import Gio  # type: ignore
    except Exception as exc:  # noqa: BLE001
        errors.append(f"Gio import failed for GvDevice fallback: {exc}")
        return None, "; ".join(errors)

    if_ip = get_interface_ipv4_for_peer(interface, camera_id)
    if not if_ip:
        errors.append(f"Interface {interface} has no IPv4 for GvDevice fallback")
        return None, "; ".join(errors)

    try:
        iface_addr = Gio.InetAddress.new_from_string(if_ip)
        dev_addr = Gio.InetAddress.new_from_string(camera_id)
        if iface_addr is None or dev_addr is None:
            raise RuntimeError("Failed to parse interface/camera IPv4")
        device = Aravis.GvDevice.new(iface_addr, dev_addr)
        camera = Aravis.Camera.new_with_device(device)
        if camera is None:
            raise RuntimeError("Aravis.Camera.new_with_device returned None")
        return camera, "opened via GvDevice fallback"
    except Exception as exc:  # noqa: BLE001
        errors.append(f"GvDevice fallback failed: {exc}")
        return None, "; ".join(errors)


def configure_aravis_gige_interface(Aravis, interface: str) -> None:
    try:
        Aravis.GvInterface.set_discovery_interface_name(interface)
    except Exception:
        pass
    # Some cameras send discovery replies as broadcast; without this Aravis
    # may miss device enumeration or choose unstable paths on macOS.
    try:
        flags = int(getattr(Aravis.GvInterfaceFlags, "ACK", 0))
        if flags:
            Aravis.set_interface_flags("GigEVision", flags)
    except Exception:
        pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="One-shot frame capture via Aravis")
    parser.add_argument("--camera", required=True, help="Camera IP or Aravis camera id")
    parser.add_argument("--interface", default="en10", help="GigE interface (default: en10)")
    parser.add_argument("--out", default="capture/frame.pgm", help="Output image path")
    parser.add_argument("--timeout-ms", type=int, default=5000, help="Frame wait timeout")
    parser.add_argument("--buffers", type=int, default=8, help="Stream buffer count")
    parser.add_argument(
        "--poll-ms",
        type=int,
        default=750,
        help="Single wait interval while polling for a valid frame",
    )
    parser.add_argument(
        "--packet-size",
        type=int,
        default=1440,
        help="GV packet size override (default: 1440 for MTU 1500 links)",
    )
    parser.add_argument(
        "--packet-delay",
        type=int,
        default=1000,
        help="Optional GevSCPD packet delay ticks (default: 1000)",
    )
    parser.add_argument(
        "--scientific-session",
        action="store_true",
        help="Enable scientific RAW session export (npy + json metadata)",
    )
    parser.add_argument(
        "--profile",
        default="scene_capture",
        choices=list_profile_names(),
        help="Capture profile for scientific session mode",
    )
    parser.add_argument("--session-dir", default="capture", help="Base directory for session outputs")
    parser.add_argument("--frames-count", type=int, default=0, help="Override profile frames_count (0 = profile default)")
    parser.add_argument("--save-preview", action="store_true", help="Save lightweight preview PNG in scientific mode")
    parser.add_argument("--pixel-format", default="", help="Override PixelFormat in scientific mode")
    parser.add_argument("--exposure-us", type=float, default=-1.0, help="Override ExposureTime for profile")
    parser.add_argument("--gain-db", type=float, default=-1.0, help="Override Gain for profile")
    parser.add_argument("--notes", default="", help="Optional session notes")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    try:
        import gi
    except ModuleNotFoundError:
        print("ERROR: Python 'gi' module is missing for this interpreter.")
        print("Run with: /opt/homebrew/bin/python3.14 tools/baumer_capture_one.py ...")
        return 2

    gi.require_version("Aravis", "0.8")
    from gi.repository import Aravis  # type: ignore

    configure_aravis_gige_interface(Aravis, args.interface)
    try:
        Aravis.update_device_list()
    except Exception:
        pass

    camera, open_note = open_camera_with_fallback(Aravis, args.camera, args.interface)
    if camera is None:
        print(f"ERROR: failed to create camera object: {open_note}")
        return 4
    if open_note:
        print(f"Connect note: {open_note}")

    # Avoid packet socket path on systems where raw socket usage is restricted.
    camera.gv_set_stream_options(Aravis.GvStreamOption.PACKET_SOCKET_DISABLED)
    camera.gv_set_packet_size_adjustment(Aravis.GvPacketSizeAdjustment.NEVER)

    if args.packet_size > 0:
        try:
            camera.gv_set_packet_size(args.packet_size)
        except Exception as exc:  # noqa: BLE001
            print(f"WARNING: failed to set packet size {args.packet_size}: {exc}")

    vendor = ""
    model = ""
    pixel_format = ""
    try:
        vendor = camera.get_vendor_name()
        model = camera.get_model_name()
        payload = camera.get_payload()
        pixel_format = camera.get_pixel_format_as_string()
        print(f"Camera: {vendor} {model}")
        print(f"PixelFormat: {pixel_format}")
        print(f"Payload bytes: {payload}")
    except Exception as exc:  # noqa: BLE001
        print(f"WARNING: failed to query camera metadata: {exc}")
        payload = 0

    try:
        stream = camera.create_stream(None, None)
    except Exception as exc:  # noqa: BLE001
        print(f"ERROR: failed to create stream: {exc}")
        return 5

    if stream is None:
        print("ERROR: stream is None")
        return 6

    # Force stream destination to selected interface address/port.
    if args.interface:
        if_ip = get_interface_ipv4_for_peer(args.interface, args.camera)
        if if_ip:
            try:
                if hasattr(stream, "get_port"):
                    stream_port = int(stream.get_port())
                    camera.set_integer("GevSCPHostPort", stream_port)
                    print(f"Set GevSCPHostPort={stream_port}")
            except Exception as exc:  # noqa: BLE001
                print(f"WARNING: failed to set GevSCPHostPort: {exc}")
            try:
                camera.set_integer("GevSCDA", ipv4_to_u32(if_ip))
                print(f"Set GevSCDA={if_ip}")
            except Exception as exc:  # noqa: BLE001
                print(f"WARNING: failed to set GevSCDA: {exc}")
        else:
            print(f"WARNING: interface {args.interface} has no IPv4, skipping GevSCDA setup")

    if payload <= 0:
        try:
            payload = camera.get_payload()
        except Exception:
            pass
    if payload <= 0:
        print("ERROR: invalid payload size")
        return 7

    for _ in range(args.buffers):
        stream.push_buffer(Aravis.Buffer.new_allocate(payload))

    buffer = None
    try:
        camera.set_string("TriggerMode", "Off")
        print("Set TriggerMode=Off")
    except Exception:
        pass

    if args.packet_size > 0:
        try:
            camera.set_integer("GevSCPSPacketSize", int(args.packet_size))
            print(f"Set GevSCPSPacketSize={int(args.packet_size)}")
        except Exception as exc:  # noqa: BLE001
            print(f"WARNING: failed to set GevSCPSPacketSize: {exc}")

    if args.packet_delay > 0:
        try:
            camera.set_integer("GevSCPD", int(args.packet_delay))
            print(f"Set GevSCPD={int(args.packet_delay)}")
        except Exception as exc:  # noqa: BLE001
            print(f"WARNING: failed to set GevSCPD: {exc}")

    try:
        camera.set_acquisition_mode(Aravis.AcquisitionMode.CONTINUOUS)
    except Exception:
        pass

    scientific_requested = bool(
        args.scientific_session
        or args.frames_count != 0
        or args.profile != "scene_capture"
        or args.save_preview
        or bool(args.pixel_format.strip())
        or args.exposure_us >= 0
        or args.gain_db >= 0
        or bool(args.notes.strip())
    )

    if scientific_requested:
        try:
            profile = get_capture_profile(args.profile)
        except Exception as exc:
            print(f"ERROR: unknown profile {args.profile}: {exc}")
            return 10

        overrides: dict[str, object] = {}
        if args.frames_count != 0:
            overrides["frames_count"] = int(args.frames_count)
        if args.save_preview:
            overrides["save_preview"] = True
        if args.pixel_format.strip():
            overrides["pixel_format"] = args.pixel_format.strip()
        if args.exposure_us >= 0:
            overrides["exposure_us"] = float(args.exposure_us)
        if args.gain_db >= 0:
            overrides["gain_db"] = float(args.gain_db)
        if overrides:
            profile = with_overrides(profile, **overrides)

        config_result = configure_scientific_mode(camera, profile, logger=lambda msg: print(f"[scientific] {msg}"))

        camera_meta = {
            "vendor": vendor,
            "model": model,
            "serial_number": str(camera.get_device_serial_number()) if hasattr(camera, "get_device_serial_number") else "",
            "camera_id": args.camera,
            "interface": args.interface,
            "packet_size": args.packet_size,
            "packet_delay": args.packet_delay,
        }
        camera_meta.update(read_camera_runtime_metadata(camera))

        writer = SessionWriter(
            base_dir=Path(args.session_dir).expanduser(),
            profile=profile,
            camera_metadata=camera_meta,
            notes=args.notes,
        )
        writer.set_configuration_result(config_result)
        print(f"Scientific session directory: {writer.root_dir}")

        frames_target = int(profile.frames_count)
        if frames_target == 0:
            frames_target = 1

        poll_ms = max(1, int(args.poll_ms))
        timeout_window_s = max(1, int(args.timeout_ms)) / 1000.0
        success_status = int(Aravis.BufferStatus.SUCCESS)
        captured = 0
        dropped = 0
        last_success_ts = time.monotonic()
        timed_out = False
        interrupted = False
        try:
            camera.start_acquisition()
            while True:
                if frames_target > 0 and captured >= frames_target:
                    break
                candidate = stream.timeout_pop_buffer(poll_ms)
                if candidate is None:
                    if (time.monotonic() - last_success_ts) > timeout_window_s:
                        timed_out = True
                        break
                    continue

                try:
                    candidate_status = int(candidate.get_status())
                except Exception:
                    candidate_status = -1

                if candidate_status != success_status:
                    dropped += 1
                    stream.push_buffer(candidate)
                    continue

                try:
                    width = int(candidate.get_image_width())
                    height = int(candidate.get_image_height())
                    pixel_format_int = int(candidate.get_image_pixel_format())
                    raw = bytes(candidate.get_image_data())
                    fmt_name = pixel_format_to_name(pixel_format_int)
                    raw_array = decode_buffer_to_ndarray(raw, width, height, fmt_name)

                    frame_meta = read_buffer_metadata(candidate)
                    frame_meta["width"] = width
                    frame_meta["height"] = height
                    frame_meta["pixel_format_int"] = pixel_format_int
                    frame_meta["pixel_format"] = fmt_name
                    runtime_now = read_camera_runtime_metadata(camera)
                    for key in ("exposure_us", "gain_db", "black_level"):
                        if key in runtime_now:
                            frame_meta[key] = runtime_now.get(key)

                    writer.write_frame(
                        frame_index=captured + 1,
                        raw_array=raw_array,
                        raw_bytes=raw,
                        frame_metadata=frame_meta,
                        save_preview=bool(profile.save_preview),
                    )
                    captured += 1
                    last_success_ts = time.monotonic()
                    print(f"Captured frame {captured}")
                except Exception as exc:
                    dropped += 1
                    writer.add_warning(f"Frame {captured + 1}: {exc}")
                finally:
                    stream.push_buffer(candidate)
        except KeyboardInterrupt:
            interrupted = True
            print("Capture interrupted by user.")
        finally:
            try:
                camera.stop_acquisition()
            except Exception:
                pass
            writer.finalize()

        print(f"Session complete: {writer.root_dir}")
        print(f"Captured: {captured}, dropped/non-success: {dropped}")
        if interrupted:
            return 0
        if captured <= 0:
            print("ERROR: no successful frames captured.")
            return 11
        if timed_out and (frames_target > 0 and captured < frames_target):
            print("WARNING: capture stopped due to timeout before reaching target frame count.")
            return 12
        return 0

    last_non_success = None
    deadline = time.monotonic() + max(args.timeout_ms, 1) / 1000.0
    poll_ms = max(1, int(args.poll_ms))
    try:
        camera.start_acquisition()
        while time.monotonic() < deadline:
            remaining_ms = int(max(1.0, (deadline - time.monotonic()) * 1000.0))
            wait_ms = min(poll_ms, remaining_ms)
            candidate = stream.timeout_pop_buffer(wait_ms)
            if candidate is None:
                continue
            candidate_status = candidate.get_status()
            if int(candidate_status) == int(Aravis.BufferStatus.SUCCESS):
                buffer = candidate
                break
            last_non_success = candidate
            print(f"Discarding buffer status {int(candidate_status)} ({candidate_status})")
            stream.push_buffer(candidate)
    finally:
        try:
            camera.stop_acquisition()
        except Exception:
            pass

    if buffer is None:
        if last_non_success is None:
            print("ERROR: timeout waiting for frame")
            return 8
        buffer = last_non_success

    status = buffer.get_status()
    if int(status) != int(Aravis.BufferStatus.SUCCESS):
        print(f"ERROR: frame status is {int(status)} ({status})")
        out_path = Path(args.out).expanduser()
        out_path.parent.mkdir(parents=True, exist_ok=True)

        debug_meta: dict[str, object] = {"status_int": int(status), "status": str(status)}

        try:
            payload_type = buffer.get_payload_type()
            debug_meta["payload_type_int"] = int(payload_type)
            debug_meta["payload_type"] = str(payload_type)
            print(f"PayloadType: {debug_meta['payload_type_int']} ({debug_meta['payload_type']})")
        except Exception:
            pass

        try:
            n_parts = int(buffer.get_n_parts())
            debug_meta["n_parts"] = n_parts
            print(f"Parts: {n_parts}")
        except Exception:
            n_parts = 0

        # Fallback: try to dump multipart/raw payload to aid debugging.
        dumped_any = False
        if n_parts > 0:
            parts: list[dict[str, object]] = []
            for i in range(n_parts):
                part_meta: dict[str, object] = {"index": i}
                part_type_int = None
                try:
                    part_type_int = int(buffer.get_part_data_type(i))
                    part_meta["data_type_int"] = part_type_int
                except Exception:
                    pass
                if part_type_int in (1, 2, 3, 4, 5, 6, 7, 8, 9):
                    try:
                        part_meta["width"] = int(buffer.get_part_width(i))
                        part_meta["height"] = int(buffer.get_part_height(i))
                    except Exception:
                        pass
                    try:
                        part_meta["pixel_format_int"] = int(buffer.get_part_pixel_format(i))
                    except Exception:
                        pass
                try:
                    pdata = bytes(buffer.get_part_data(i))
                    if pdata:
                        ppath = out_path.with_suffix(f".part{i}.bin")
                        ppath.write_bytes(pdata)
                        part_meta["bytes"] = len(pdata)
                        part_meta["file"] = str(ppath)
                        dumped_any = True
                except Exception:
                    pass
                parts.append(part_meta)
            debug_meta["parts"] = parts

        if not dumped_any:
            try:
                raw_all = bytes(buffer.get_data())
                if raw_all:
                    raw_path = out_path.with_suffix(".raw")
                    raw_path.write_bytes(raw_all)
                    debug_meta["bytes"] = len(raw_all)
                    debug_meta["file"] = str(raw_path)
                    dumped_any = True
            except Exception:
                pass

        debug_path = out_path.with_suffix(".debug.json")
        debug_path.write_text(json.dumps(debug_meta, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"Saved debug metadata: {debug_path}")
        if dumped_any:
            print("Saved raw payload dump for offline decode.")
        return 9

    width = int(buffer.get_image_width())
    height = int(buffer.get_image_height())
    pixel_format = int(buffer.get_image_pixel_format())
    raw = bytes(buffer.get_image_data())

    out_path = Path(args.out).expanduser()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # If this is 8-bit mono-like payload, save directly as PGM.
    if len(raw) == width * height:
        if out_path.suffix.lower() not in (".pgm", ".raw"):
            out_path = out_path.with_suffix(".pgm")
        if out_path.suffix.lower() == ".pgm":
            save_pgm(out_path, width, height, raw)
        else:
            out_path.write_bytes(raw)
    else:
        out_path = out_path.with_suffix(".raw")
        out_path.write_bytes(raw)
        meta = {
            "width": width,
            "height": height,
            "pixel_format_int": pixel_format,
            "bytes": len(raw),
            "note": "saved as RAW because payload is not width*height 8-bit",
        }
        out_path.with_suffix(".json").write_text(
            json.dumps(meta, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    print(f"Saved frame: {out_path}")
    print(f"Frame: {width}x{height}, bytes={len(raw)}, pixel_format=0x{pixel_format:08x}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
