# Baumer Camera (macOS, Python, Aravis)

Легковесный проект для работы с GigE-камерой Baumer на macOS:

- live preview GUI (`Tkinter`)
- scientific RAW capture sessions (`.npy + .json`)
- инструменты обнаружения и восстановления IP камеры

## Что в репозитории

- `tools/baumer_live_gui.py` - основное GUI-приложение
- `tools/baumer_capture_one.py` - CLI захват кадров/сессий
- `tools/baumer_gvcp_explorer.py` - discovery камер по GVCP
- `tools/baumer_force_ip.py` - временная смена IP камеры (FORCEIP)
- `tools/baumer_network_diagnose.sh` - диагностика сети/маршрутов
- `tools/capture_profiles.py` - capture profiles
- `tools/camera_control.py` - scientific camera configuration
- `tools/raw_decode.py` - raw decode layer
- `tools/capture_session.py` - session writer
- `docs/macos-network-setup.md` - подробный сетевой гайд

## Требования

- macOS
- Homebrew
- Python `3.14` (рекомендуется `/opt/homebrew/bin/python3.14`)
- Aravis + `gi.repository` (PyGObject)

## Развертывание

### 1) Клонирование

```bash
git clone <YOUR_REPO_URL>
cd BaumerCamera
```

### 2) Системные зависимости (Homebrew)

```bash
brew update
brew install aravis pygobject3 gobject-introspection pkg-config
```

### 3) Python-зависимости

```bash
/opt/homebrew/bin/python3.14 -m pip install --break-system-packages -r requirements.txt
```

### 4) Быстрая проверка

```bash
/opt/homebrew/bin/python3.14 tools/baumer_live_gui.py --help
/opt/homebrew/bin/python3.14 tools/baumer_capture_one.py --help
```

## Настройки сети
Перед первым запуском обязательно выполните сетевую настройку.

### 1) Узнать имя Ethernet-интерфейса камеры

```bash
networksetup -listallhardwareports
```
Найдите ваш USB-Ethernet адаптер (например `AX88179A`) и его `Device` (например `en10`).

### 2) Настроить IP хоста в той же подсети камеры

Если IP камеры неизвестен, временно задайте на интерфейсе любую private-подсеть, например:

```bash
sudo ifconfig <CAMERA_IFACE> inet 192.168.88.10 netmask 255.255.255.0 up
```

Если у вас уже известная подсеть камеры - используйте её.

### 3) Проверить discovery

```bash
/opt/homebrew/bin/python3.14 tools/baumer_gvcp_explorer.py \
  --interface <CAMERA_IFACE> \
  --duration 4
```

Если discovery не видит камеру, проверьте:
- кабель/питание камеры и линк (`ifconfig <CAMERA_IFACE>`, `status: active`)
- что интерфейс не inactive в macOS Network Settings
- что VPN/корпоративный firewall не мешает локальному трафику
- что маршрут не уходит в `en0` (Wi-Fi)

### 4) Если камера в wrong subnet - применить FORCEIP

```bash
/opt/homebrew/bin/python3.14 tools/baumer_force_ip.py \
  --interface <CAMERA_IFACE> \
  --mac <CAMERA_MAC> \
  --ip <TARGET_CAMERA_IP> \
  --mask <TARGET_MASK> \
  --gateway 0.0.0.0
```

После этого снова запустите `baumer_gvcp_explorer.py`.

### 5) Если маршрут на IP камеры идет не через camera interface

```bash
route -n get <CAMERA_IP>
sudo route -n delete -host <CAMERA_IP> 2>/dev/null || true
sudo route -n add -host <CAMERA_IP> -interface <CAMERA_IFACE>
sudo arp -d <CAMERA_IP> 2>/dev/null || true
```

### 6) Полная диагностика в один шаг

```bash
bash tools/baumer_network_diagnose.sh <CAMERA_IFACE>
```

### 7) Только после этого запускать приложение

#### GUI (основной режим)

```bash
/opt/homebrew/bin/python3.14 tools/baumer_live_gui.py
```

GUI больше не подставляет «чужие» дефолтные значения IP/interface.  
Сначала укажите `Interface` и `Camera IP` в окне, либо нажмите `Auto Find/Fix`.

### CLI: scientific session (1 кадр)

```bash
/opt/homebrew/bin/python3.14 tools/baumer_capture_one.py \
  --camera <CAMERA_IP> \
  --interface <CAMERA_IFACE> \
  --scientific-session \
  --profile scene_capture \
  --frames-count 1 \
  --session-dir capture
```

### CLI: burst (например dark frames)

```bash
/opt/homebrew/bin/python3.14 tools/baumer_capture_one.py \
  --camera <CAMERA_IP> \
  --interface <CAMERA_IFACE> \
  --scientific-session \
  --profile dark_frame \
  --frames-count 16 \
  --session-dir capture
```

## Формат scientific output

```text
capture/session_YYYY-MM-DD_HH-MM-SS_xxxxxx/
  session.json
  frames/
    frame_000001.npy
    frame_000001.json
    frame_000001_preview.png   # опционально
  logs/
    warnings.log               # если были warning
```

## Частые проблемы и как чинить

1. `No GVCP discovery replies`:
- неверный интерфейс
- интерфейс inactive/down
- камера в другой подсети
- проблемы кабеля/питания/линка

2. `Can't connect to device at address ...`:
- discovery видит камеру, но host-route указывает не на camera interface
- IP камеры поменялся после reboot/forceip

3. `access-denied` при set exposure/gain:
- контроль камеры у другого клиента (другая программа/ПК)
- reconnect в GUI обычно снимает проблему

4. Низкий FPS preview:
- большой `GevSCPD`
- ограничения сети 1GbE при большом payload
- слишком высокая частота preview render

Подробный сетевой гайд: [docs/macos-network-setup.md](docs/macos-network-setup.md)

## Примечания

- Приоритет проекта: корректный RAW scientific capture, а не «красивый» preview.
- Packed 10/12-bit форматы пока intentionally не декодируются в scientific pipeline.
