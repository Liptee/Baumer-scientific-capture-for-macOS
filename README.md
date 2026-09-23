# Baumer Camera (macOS, Python, Aravis)

Легковесный проект для работы с GigE-камерой Baumer на macOS:

- live preview GUI (`Tkinter`)
- Hydra calibration/capture GUI with face segmentation/classification workflow
- scientific RAW capture sessions (`.npy + .json`)
- инструменты обнаружения и восстановления IP камеры

## Что в репозитории

- `tools/baumer_hydra_gui.py` - Hydra GUI: calibration, capture, analysis, face segmentation/classification
- `tools/baumer_live_gui.py` - легковесное live preview GUI
- `tools/baumer_capture_one.py` - CLI захват кадров/сессий
- `tools/baumer_gvcp_explorer.py` - discovery камер по GVCP
- `tools/baumer_force_ip.py` - временная смена IP камеры (FORCEIP)
- `tools/baumer_network_diagnose.sh` - диагностика сети/маршрутов
- `tools/capture_profiles.py` - capture profiles
- `tools/camera_control.py` - scientific camera configuration
- `tools/raw_decode.py` - raw decode layer
- `tools/capture_session.py` - session writer
- `tools/hsi_recon_worker.py` - worker для HSI-реконструкции в analysis/classification pipeline
- `tools/train_face_spectrum_classifier.py` - обучение простого спектрального face classifier
- `weights/` - lightweight config/metadata для реконструкции и классификации
- `docs/macos-network-setup.md` - подробный сетевой гайд

## Требования

- macOS
- Homebrew
- Python `3.14`
- Aravis + `gi.repository` (PyGObject)
- Tkinter для Python 3.14
- локальные checkpoint-файлы реконструкции в `weights/` (не хранятся в git)

## Развертывание

### 1) Клонирование

```bash
git clone <YOUR_REPO_URL>
cd HydraSoft
```

### 2) Системные зависимости (Homebrew)

```bash
brew update
brew install python@3.14 python-tk@3.14 aravis pygobject3 gobject-introspection pkg-config
```

### 3) Python-зависимости

Рекомендуемый вариант - локальное виртуальное окружение в корне проекта:

```bash
/opt/homebrew/opt/python@3.14/bin/python3.14 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Если `gi.repository` не виден из `.venv`, разрешите окружению видеть Homebrew site-packages:

```bash
python - <<'PY'
from pathlib import Path
p = Path(".venv/pyvenv.cfg")
s = p.read_text()
s = s.replace("include-system-site-packages = false", "include-system-site-packages = true")
p.write_text(s)
PY
```

### 4) Быстрая проверка

```bash
source .venv/bin/activate
python -c "import tkinter; import gi; gi.require_version('Aravis', '0.8'); from gi.repository import Aravis; print('Tk/Aravis OK')"
python -m py_compile tools/baumer_hydra_gui.py tools/baumer_capture_one.py
python tools/baumer_capture_one.py --help
```

### 5) Локальные веса

Большие checkpoint-файлы не коммитятся. Для HSI-реконструкции положите их в `weights/`:

```text
weights/weights.ckpt
weights/weights_base.ckpt          # optional
weights/weights_for_shpak.ckpt     # optional
```

Лёгкие файлы `weights/config.yaml`, `weights/wavelengths.txt` и
`weights/face_cls_model.json` можно хранить в git.

### 6) Запуск Hydra GUI

```bash
cd /path/to/HydraSoft
source .venv/bin/activate
GST_PLUGIN_PATH=/opt/homebrew/opt/aravis/lib/gstreamer-1.0 python tools/baumer_hydra_gui.py
```

Если окно Tk падает при запуске из sandboxed terminal, запустите ту же команду из обычного macOS Terminal.

### 7) Запуск lightweight live GUI

```bash
cd /path/to/HydraSoft
source .venv/bin/activate
GST_PLUGIN_PATH=/opt/homebrew/opt/aravis/lib/gstreamer-1.0 python tools/baumer_live_gui.py
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
source .venv/bin/activate
python tools/baumer_gvcp_explorer.py \
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
source .venv/bin/activate
python tools/baumer_force_ip.py \
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
source .venv/bin/activate
GST_PLUGIN_PATH=/opt/homebrew/opt/aravis/lib/gstreamer-1.0 python tools/baumer_hydra_gui.py
```

GUI больше не подставляет «чужие» дефолтные значения IP/interface.  
Сначала укажите `Interface` и `Camera IP` в окне, либо нажмите `Auto Find/Fix`.

#### Analysis / Classification в Hydra GUI

Для работы кнопок `Анализ` и `Classification` должны быть выполнены условия:

- есть live frame с камеры;
- завершена calibration/crop настройка;
- задана white point;
- выбрана `Segmentation lens` от 1 до 16;
- face segmentation model загружена;
- HSI reconstruction precheck прошёл успешно;
- для `Classification` дополнительно загружен `weights/face_cls_model.json`.

Кнопка `Демо-режим` становится доступна только тогда, когда готов штатный
`Classification`. При этом внутри самого демо-режима сегментация и классификация
лица не запускаются: система только реконструирует HSI по свежему кадру, сразу
показывает спектр случайной точки, затем выбирает ещё три точки с интервалом
5 секунд. Через 5 секунд после четвёртой точки запускается новая HSI-реконструкция
по свежему кадру, и цикл повторяется. Ручной выбор точки на HSI-изображении
немедленно останавливает демо-режим и оставляет результат в обычном интерактивном
режиме. Демо-режим также можно остановить повторным нажатием кнопки.

### CLI: scientific session (1 кадр)

```bash
source .venv/bin/activate
python tools/baumer_capture_one.py \
  --camera <CAMERA_IP> \
  --interface <CAMERA_IFACE> \
  --scientific-session \
  --profile scene_capture \
  --frames-count 1 \
  --session-dir capture
```

### CLI: burst (например dark frames)

```bash
source .venv/bin/activate
python tools/baumer_capture_one.py \
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
