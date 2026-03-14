#!/usr/bin/env bash
set -euo pipefail

IFACE="${1:-en10}"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DISCOVERY_TOOL="${ROOT_DIR}/tools/baumer_gvcp_explorer.py"

echo "=== Baumer macOS network diagnose ==="
echo "time: $(date '+%Y-%m-%d %H:%M:%S %Z')"
echo "camera interface: ${IFACE}"
echo ""

echo "[1/5] Interface status"
ifconfig "${IFACE}" || true
echo ""
ifconfig en0 || true
echo ""

echo "[2/5] Route table (focus: default + link-local)"
if command -v rg >/dev/null 2>&1; then
  netstat -rn -f inet | rg 'default|169\.254|en0|en10|en14' || true
else
  netstat -rn -f inet | grep -E 'default|169\.254|en0|en10|en14' || true
fi
echo ""

echo "[3/5] ARP entries on ${IFACE}"
arp -an -i "${IFACE}" || true
echo ""

echo "[4/5] GigE Vision discovery on ${IFACE}"
DISCOVERY_JSON="$(python3 "${DISCOVERY_TOOL}" --interface "${IFACE}" --duration 4 --json || true)"
if [[ -z "${DISCOVERY_JSON}" ]]; then
  echo "No discovery payload."
else
  echo "${DISCOVERY_JSON}"
fi
echo ""

echo "[5/5] Host-route checks"
export DISCOVERY_JSON
python3 - <<'PY'
import json
import os
import subprocess

raw = os.environ.get("DISCOVERY_JSON", "").strip()
if not raw:
    print("No discovered devices to check routes for.")
    raise SystemExit(0)

try:
    devices = json.loads(raw)
except json.JSONDecodeError:
    print("Could not parse discovery JSON.")
    raise SystemExit(0)

if not devices:
    print("No discovered devices.")
    raise SystemExit(0)

for dev in devices:
    ip = dev.get("source_ip", "")
    if not ip:
        continue
    print(f"\nroute -n get {ip}")
    try:
        out = subprocess.check_output(
            ["route", "-n", "get", ip],
            text=True,
            stderr=subprocess.STDOUT,
        )
        print(out.rstrip())
        if "interface: en0" in out:
            print("WARNING: host route points to Wi-Fi (en0), camera traffic may fail.")
    except subprocess.CalledProcessError as exc:
        print(exc.output.rstrip())
PY

echo ""
echo "Done."
