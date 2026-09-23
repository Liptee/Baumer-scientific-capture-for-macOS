#!/usr/bin/env python3
"""GVCP-обнаружение GigE Vision камер для Hydra GUI на macOS."""

from __future__ import annotations

import socket
import struct
import subprocess
import time
from dataclasses import dataclass
from typing import List, Optional

import fcntl

SIOCGIFADDR = 0xC0206921
IP_BOUND_IF = 25  # macOS: bind socket to interface index.

GVCP_DISCOVERY_CMD = 0x0002
GVCP_DISCOVERY_ACK = 0x0003
GVCP_PORT = 3956


@dataclass
class CameraInfo:
    source_ip: str
    status: int
    ack: int
    req_id: int
    gev_version: str
    device_mode_hex: str
    mac: str
    ip_cfg_options_hex: str
    ip_cfg_current_hex: str
    current_ip: str
    current_subnet: str
    current_gateway: str
    persistent_ip: str
    persistent_gateway: str
    persistent_subnet: str
    manufacturer_name: str
    model_name: str
    device_version: str
    manufacturer_info: str
    serial_number: str
    user_defined_name: str


def get_ipv4_for_interface(interface: str) -> Optional[str]:
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        packed = struct.pack("256s", interface[:15].encode("ascii", errors="ignore"))
        data = fcntl.ioctl(sock.fileno(), SIOCGIFADDR, packed)
        return socket.inet_ntoa(data[20:24])
    except OSError:
        return None
    finally:
        sock.close()


def get_broadcast_for_interface(interface: str) -> Optional[str]:
    try:
        out = subprocess.check_output(["ifconfig", interface], text=True, stderr=subprocess.DEVNULL)
    except Exception:
        return None
    for line in out.splitlines():
        line = line.strip()
        if " broadcast " not in line:
            continue
        parts = line.split()
        if "broadcast" in parts:
            idx = parts.index("broadcast")
            if idx + 1 < len(parts):
                return parts[idx + 1]
    return None


def ipv4_from_bytes(raw: bytes) -> str:
    if len(raw) != 4:
        return "0.0.0.0"
    return ".".join(str(b) for b in raw)


def decode_ascii_field(payload: bytes, offset: int, length: int) -> str:
    if len(payload) <= offset:
        return ""
    chunk = payload[offset : offset + length]
    return chunk.split(b"\x00", 1)[0].decode("ascii", errors="ignore").strip()


def parse_discovery_ack(data: bytes, source_ip: str) -> Optional[CameraInfo]:
    if len(data) < 8:
        return None

    status, ack, payload_len, req_id = struct.unpack(">HHHH", data[:8])
    if ack != GVCP_DISCOVERY_ACK:
        return None

    payload = data[8:]
    if payload_len > len(payload):
        return None

    major = int.from_bytes(payload[0:2], "big") if len(payload) >= 2 else 0
    minor = int.from_bytes(payload[2:4], "big") if len(payload) >= 4 else 0
    device_mode = int.from_bytes(payload[4:8], "big") if len(payload) >= 8 else 0

    # Bootstrap block (common layout in GigE Vision discovery ACK).
    mac = ""
    if len(payload) >= 16:
        mac_bytes = payload[10:16]
        if len(mac_bytes) == 6:
            mac = ":".join(f"{b:02x}" for b in mac_bytes)

    ip_cfg_options = int.from_bytes(payload[16:20], "big") if len(payload) >= 20 else 0
    ip_cfg_current = int.from_bytes(payload[20:24], "big") if len(payload) >= 24 else 0

    # GigE Vision discovery ACK layouts vary across vendors/firmware.
    # Observed on Baumer MXGC40c: current IP at [36:40], subnet at [52:56].
    cur_ip_a = ipv4_from_bytes(payload[24:28]) if len(payload) >= 28 else "0.0.0.0"
    cur_mask_a = ipv4_from_bytes(payload[28:32]) if len(payload) >= 32 else "0.0.0.0"
    cur_gw_a = ipv4_from_bytes(payload[32:36]) if len(payload) >= 36 else "0.0.0.0"

    cur_ip_b = ipv4_from_bytes(payload[36:40]) if len(payload) >= 40 else "0.0.0.0"
    cur_mask_b = ipv4_from_bytes(payload[52:56]) if len(payload) >= 56 else "0.0.0.0"
    cur_gw_b = ipv4_from_bytes(payload[40:44]) if len(payload) >= 44 else "0.0.0.0"

    if cur_ip_b != "0.0.0.0":
        current_ip, current_subnet, current_gateway = cur_ip_b, cur_mask_b, cur_gw_b
        persistent_ip, persistent_subnet, persistent_gateway = cur_ip_a, cur_mask_a, cur_gw_a
    else:
        current_ip, current_subnet, current_gateway = cur_ip_a, cur_mask_a, cur_gw_a
        persistent_ip = ipv4_from_bytes(payload[36:40]) if len(payload) >= 40 else "0.0.0.0"
        persistent_gateway = ipv4_from_bytes(payload[40:44]) if len(payload) >= 44 else "0.0.0.0"
        persistent_subnet = ipv4_from_bytes(payload[48:52]) if len(payload) >= 52 else "0.0.0.0"

    # String block offsets used by Baumer discovery ACK payload observed on macOS.
    manufacturer_name = decode_ascii_field(payload, 72, 32)
    model_name = decode_ascii_field(payload, 104, 32)
    device_version = decode_ascii_field(payload, 136, 32)
    manufacturer_info = decode_ascii_field(payload, 168, 48)
    serial_number = decode_ascii_field(payload, 216, 16)
    user_defined_name = decode_ascii_field(payload, 232, 16)

    return CameraInfo(
        source_ip=source_ip,
        status=status,
        ack=ack,
        req_id=req_id,
        gev_version=f"{major}.{minor}",
        device_mode_hex=f"0x{device_mode:08x}",
        mac=mac,
        ip_cfg_options_hex=f"0x{ip_cfg_options:08x}",
        ip_cfg_current_hex=f"0x{ip_cfg_current:08x}",
        current_ip=current_ip,
        current_subnet=current_subnet,
        current_gateway=current_gateway,
        persistent_ip=persistent_ip,
        persistent_gateway=persistent_gateway,
        persistent_subnet=persistent_subnet,
        manufacturer_name=manufacturer_name,
        model_name=model_name,
        device_version=device_version,
        manufacturer_info=manufacturer_info,
        serial_number=serial_number,
        user_defined_name=user_defined_name,
    )


def discover(interface: str, duration: float, interval: float) -> List[CameraInfo]:
    source_ip = get_ipv4_for_interface(interface)
    if not source_ip:
        raise RuntimeError(
            f"Interface '{interface}' has no IPv4 address. Configure it first."
        )

    interface_index = socket.if_nametoindex(interface)
    request_id = int(time.time()) & 0xFFFF
    discovery_packet = struct.pack(">BBHHH", 0x42, 0x11, GVCP_DISCOVERY_CMD, 0, request_id)
    interface_broadcast = get_broadcast_for_interface(interface)
    destinations = [d for d in (interface_broadcast, "255.255.255.255", "169.254.255.255") if d]
    # Keep destination order deterministic and unique.
    destinations = list(dict.fromkeys(destinations))

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
    sock.setsockopt(socket.IPPROTO_IP, IP_BOUND_IF, struct.pack("I", interface_index))
    # Bind to INADDR_ANY so broadcast replies to 255.255.255.255 / subnet
    # broadcast are received reliably on macOS.
    try:
        sock.bind(("", 0))
    except OSError:
        sock.bind((source_ip, 0))
    sock.settimeout(0.15)

    discovered: dict[str, CameraInfo] = {}
    start = time.monotonic()
    next_send = 0.0

    try:
        while True:
            now = time.monotonic()
            if now - start >= duration:
                break

            if now >= next_send:
                for dst in destinations:
                    try:
                        sock.sendto(discovery_packet, (dst, GVCP_PORT))
                    except OSError:
                        # Keep probing; some routes can be temporarily unavailable.
                        pass
                next_send = now + interval

            try:
                data, addr = sock.recvfrom(8192)
            except socket.timeout:
                continue
            except OSError:
                continue

            cam = parse_discovery_ack(data, addr[0])
            if not cam:
                continue
            discovered[cam.source_ip] = cam
    finally:
        sock.close()

    return list(discovered.values())
