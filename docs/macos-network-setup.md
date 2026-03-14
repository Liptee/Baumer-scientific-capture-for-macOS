# macOS: Network Setup for Baumer GigE Camera

Цель: одновременно иметь интернет (обычно `Wi-Fi`) и стабильный доступ к камере
через выделенный Ethernet-интерфейс.

## 1) Определить правильный интерфейс камеры

```bash
networksetup -listallhardwareports
```

Найдите ваш USB-Ethernet адаптер и запомните:
- имя сервиса в System Settings (например `AX88179A`)
- device (например `en10`)

Далее используйте placeholders:
- `<CAMERA_SERVICE>`
- `<CAMERA_IFACE>`

## 2) Проверить link и назначить IPv4 на хосте

```bash
ifconfig <CAMERA_IFACE>
```

Нужен `status: active`. Если `inactive`, проверьте кабель/питание/адаптер.

Если IP камеры неизвестен, задайте временный IP хосту:

```bash
sudo ifconfig <CAMERA_IFACE> inet 192.168.88.10 netmask 255.255.255.0 up
```

## 3) Проверить discovery (GVCP)

```bash
/opt/homebrew/bin/python3.14 tools/baumer_gvcp_explorer.py \
  --interface <CAMERA_IFACE> \
  --duration 4
```

Если камера отвечает, зафиксируйте:
- `current IP`
- `MAC`
- `model/serial`

## 4) Если камера в другой подсети: FORCEIP

```bash
/opt/homebrew/bin/python3.14 tools/baumer_force_ip.py \
  --interface <CAMERA_IFACE> \
  --mac <CAMERA_MAC> \
  --ip <TARGET_CAMERA_IP> \
  --mask <TARGET_MASK> \
  --gateway 0.0.0.0
```

Снова выполните `baumer_gvcp_explorer.py` и убедитесь, что IP камеры изменился.

## 5) Рекомендуемая policy в macOS

1. Поставить `Wi-Fi` выше camera-сервиса в service order:

```bash
networksetup -listnetworkserviceorder
```

Через GUI: `System Settings -> Network -> ... -> Set Service Order`.

2. Для camera service не указывать default router (локальный линк).

3. По возможности отключить VPN на время первичной диагностики камеры.

## 6) Если маршрут на камеру уходит в Wi-Fi (частая проблема)

Проверить:

```bash
route -n get <CAMERA_IP>
```

Если интерфейс не `<CAMERA_IFACE>`, исправить:

```bash
sudo route -n delete -host <CAMERA_IP> 2>/dev/null || true
sudo route -n add -host <CAMERA_IP> -interface <CAMERA_IFACE>
sudo arp -d <CAMERA_IP> 2>/dev/null || true
```

## 7) Полная диагностика

```bash
bash tools/baumer_network_diagnose.sh <CAMERA_IFACE>
```

Выводит:
- состояние интерфейсов
- routes
- ARP
- discovery
- проверку host-route на IP камеры

## 8) Проверка захвата после настройки сети

```bash
/opt/homebrew/bin/python3.14 tools/baumer_capture_one.py \
  --camera <CAMERA_IP> \
  --interface <CAMERA_IFACE> \
  --scientific-session \
  --profile scene_capture \
  --frames-count 1 \
  --session-dir capture
```

## Troubleshooting (кратко)

1. `No GVCP discovery replies`:
- неверный интерфейс
- интерфейс inactive/down
- firewall/VPN
- камера в другой подсети

2. `Can't connect to device at address ...`:
- неверный `camera IP`
- host-route смотрит не на camera interface
- stale ARP entry

3. Периодические обрывы/низкий FPS:
- слишком большой `GevSCPD`
- перегружен линк 1GbE
- плохой кабель/адаптер
