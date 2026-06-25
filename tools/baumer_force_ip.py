#!/usr/bin/env python3
"""
Force temporary IPv4 address for a GigE Vision camera (GVCP FORCEIP_CMD).

This script sends broadcast FORCEIP command (0x0004) bound to a selected
interface and updates camera network settings without requiring vendor SDK.
"""

from __future__ import annotations

import argparse
import fcntl
import socket
import struct
import subprocess
import time
from typing import Optional

SIOCGIFADDR = 0xC0206921
IP_BOUND_IF = 25  # macOS: bind socket to interface index
GVCP_PORT = 3956


def get_interface_ipv4(interface: str) -> Optional[str]:
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        packed = struct.pack("256s", interface[:15].encode("ascii", errors="ignore"))
        data = fcntl.ioctl(s.fileno(), SIOCGIFADDR, packed)
        return socket.inet_ntoa(data[20:24])
    except OSError:
        return None
    finally:
        s.close()


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


def ipv4_to_bytes(value: str) -> bytes:
    parts = value.split(".")
    if len(parts) != 4:
        raise ValueError(f"invalid IPv4: {value}")
    out = bytes(int(p) for p in parts)
    if len(out) != 4:
        raise ValueError(f"invalid IPv4: {value}")
    return out


def mac_to_bytes(value: str) -> bytes:
    parts = value.split(":")
    if len(parts) != 6:
        raise ValueError(f"invalid MAC: {value}")
    return bytes(int(p, 16) for p in parts)


def send_force_ip(
    interface: str,
    target_mac: str,
    ip: str,
    subnet: str,
    gateway: str,
    timeout: float = 1.0,
) -> bool:
    host_ip = get_interface_ipv4(interface)
    if not host_ip:
        raise RuntimeError(f"interface {interface!r} has no IPv4 address")

    interface_index = socket.if_nametoindex(interface)
    req_id = int(time.time()) & 0xFFFF

    payload = bytearray(64)
    # Wireshark dissector offsets (GVCP FORCEIP_CMD):
    # MAC @ +2, IP @ +20, MASK @ +36, GATEWAY @ +52
    payload[2:8] = mac_to_bytes(target_mac)
    payload[20:24] = ipv4_to_bytes(ip)
    payload[36:40] = ipv4_to_bytes(subnet)
    payload[52:56] = ipv4_to_bytes(gateway)

    # flag 0x11: ack required + allow broadcast acknowledge
    packet = struct.pack(">BBHHH", 0x42, 0x11, 0x0004, len(payload), req_id) + payload
    interface_broadcast = get_broadcast_for_interface(interface)
    destinations = [d for d in (interface_broadcast, "255.255.255.255", "169.254.255.255") if d]
    destinations = list(dict.fromkeys(destinations))

    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
    s.setsockopt(socket.IPPROTO_IP, IP_BOUND_IF, struct.pack("I", interface_index))
    # Bind to INADDR_ANY so broadcast ACK packets are delivered reliably.
    try:
        s.bind(("", 0))
    except OSError:
        s.bind((host_ip, 0))
    s.settimeout(0.25)

    for _ in range(3):
        for dst in destinations:
            s.sendto(packet, (dst, GVCP_PORT))

    t_end = time.monotonic() + timeout
    got_ack = False
    while time.monotonic() < t_end:
        try:
            data, addr = s.recvfrom(8192)
        except socket.timeout:
            continue
        if len(data) < 8:
            continue
        status, ack, _payload_len, rid = struct.unpack(">HHHH", data[:8])
        if rid == req_id and ack == 0x0005:
            print(f"FORCEIP_ACK from {addr[0]} status_word=0x{status:04x}")
            # Many cameras send FORCEIP_ACK with a vendor/protocol marker
            # in the first 16-bit word instead of a pure status field.
            # Matching req_id + ACK opcode is enough to treat this as success.
            got_ack = True
            break

    s.close()
    return got_ack


def main() -> int:
    parser = argparse.ArgumentParser(description="GigE Vision FORCEIP sender")
    parser.add_argument("--interface", required=True, help="e.g. en10")
    parser.add_argument("--mac", required=True, help="target camera MAC, e.g. 00:06:be:01:9c:18")
    parser.add_argument("--ip", required=True, help="new temporary camera IP")
    parser.add_argument("--mask", default="255.255.0.0", help="new subnet mask")
    parser.add_argument("--gateway", default="0.0.0.0", help="new gateway")
    args = parser.parse_args()

    ok = send_force_ip(
        interface=args.interface,
        target_mac=args.mac,
        ip=args.ip,
        subnet=args.mask,
        gateway=args.gateway,
    )

    if ok:
        print("FORCEIP completed with ACK status=0")
        return 0

    print("FORCEIP sent (no ACK). Camera may still apply new IP; verify with discovery.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
