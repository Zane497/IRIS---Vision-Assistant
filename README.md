# IRIS — Intelligent Recognition and Information System
### Hybrid AI Vision Assistant for the Visually Impaired

![Platform](https://img.shields.io/badge/Platform-Raspberry%20Pi%20Zero%202W-red)
![Python](https://img.shields.io/badge/Python-3.x-blue)
![AI](https://img.shields.io/badge/AI-Gemini%202.5%20Flash%20%2B%20TFLite-green)
![Cost](https://img.shields.io/badge/Hardware%20Cost-Under%20₹2000-orange)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

> A wearable, button-operated AI vision assistant that describes scenes, reads text, and detects objects — built for under ₹2,000 with zero ongoing cost.

---

##  Overview

IRIS is a compact assistive device designed for visually impaired users. Built on the Raspberry Pi Zero 2W, it uses a hybrid two-tier AI pipeline:

- **Online** → Google Gemini 2.5 Flash API for rich natural language scene descriptions and OCR
- **Offline** → TensorFlow Lite MobileNetV2 SSD for continuous real-time object detection

All output is delivered through a USB soundcard and earpiece using the Piper TTS engine. No screen. No keyboard. Just a single button.

---

##  Problem Statement

Visually impaired individuals face significant challenges interacting with their surroundings. Existing solutions require continuous internet, expensive hardware, or are too large for portable use. IRIS addresses this with a credit-card-sized device that:

- Describes scenes in natural spoken language
- Reads visible text aloud (signs, labels, menus, medicine bottles)
- Detects objects continuously even without internet
- Operates with a single physical button — no screen or keyboard required

---

##  System Architecture

```
Button Press
     │
     ├── Short Press ──► Gemini OCR (reads visible text)
     │
     └── Long Press  ──► Gemini Scene Description
                              │
                    (running continuously in background)
                              │
                         TFLite MobileNetV2 SSD
                         Object Detection Loop
                              │
                    ┌─────────▼─────────┐
                    │    Piper TTS      │
                    │  USB Soundcard    │
                    │     Earpiece      │
                    └───────────────────┘
```

**Online Path:**
```
Button → picamera2 (640×480) → base64 → Gemini 2.5 Flash API → Text → Piper TTS → Audio
```

**Offline Path:**
```
Continuous Loop → picamera2 → TFLite MobileNetV2 SSD → Labels → Piper TTS → Audio
```

---

##  Hardware

| Component  | Details                               | Cost |
| Main Board | Raspberry Pi Zero 2W                  | ₹2,000 |
| Camera     | Pi Camera Module v1.3 (5MP)           | ₹1,200 |
| Audio      | USB Soundcard + 3.5mm earpiece        | ₹150 |
| Input      | Tactile push button (GPIO 17 + GND)   | ₹10 |
| Power      | USB power bank                        | — |
| OS         | Raspberry Pi OS Lite 64-bit (Bookworm)| Free |

**Total: ~₹3,360 

---

##  Software Stack

| Component | Technology |
|---|---|
| Language | Python 3 |
| Camera | picamera2 |
| Online AI | Google Gemini 2.5 Flash API (free tier) |
| Offline AI | TFLite MobileNetV2 SSD COCO (INT8 quantised) |
| TTS | Piper TTS (pre-compiled aarch64 binary) |
| Audio | aplay via ALSA (USB soundcard) |
| GPIO | RPi.GPIO |
| Auto-start | systemd service |

---

##  Installation

### 1. Flash OS
Flash **Raspberry Pi OS Lite (64-bit, Bookworm)** using Raspberry Pi Imager. Pre-configure Wi-Fi and SSH before flashing.

### 2. System Dependencies
```bash
sudo apt update && sudo apt full-upgrade -y
sudo apt install -y python3-picamera2 python3-pip alsa-utils python3-rpi.gpio git wget unzip
```

### 3. USB Soundcard Setup
```bash
# Find card number
aplay -l

# Set as default (replace 0 with your card number)
echo "defaults.pcm.card 0
defaults.ctl.card 0" | sudo tee /etc/asound.conf
```

### 4. Piper TTS (Binary — do NOT use pip)
```bash
cd ~
wget https://github.com/rhasspy/piper/releases/download/2023.11.14-2/piper_linux_aarch64.tar.gz
tar -xzf piper_linux_aarch64.tar.gz

mkdir -p ~/piper/voices && cd ~/piper/voices
wget https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/amy/low/en_US-amy-low.onnx
wget https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/amy/low/en_US-amy-low.onnx.json

# Test
echo 'IRIS is online.' | ~/piper/piper --model ~/piper/voices/en_US-amy-low.onnx --output_raw | aplay -r 22050 -f S16_LE -t raw -
```

> ⚠️ **Do not use `pip install piper-tts`** — it causes an illegal instruction error on the Pi Zero 2W. Use the pre-compiled binary above.

### 5. Python Packages & TFLite Model
```bash
pip3 install tflite-runtime requests RPi.GPIO pillow --break-system-packages

cd ~
wget https://storage.googleapis.com/download.tensorflow.org/models/tflite/coco_ssd_mobilenet_v1_1.0_quant_2018_06_29.zip
unzip coco_ssd_mobilenet_v1_1.0_quant_2018_06_29.zip
```

### 6. Gemini API Key
Get a free API key (no credit card) from [aistudio.google.com/app/apikey](https://aistudio.google.com/app/apikey).
Set it in `IRIS.py`:
```python
GEMINI_KEY = "your-api-key-here"
```

### 7. Auto-start on Boot
```bash
sudo nano /etc/systemd/system/IRIS.service
```
```ini
[Unit]
Description=IRIS Vision Assistant
After=network.target

[Service]
ExecStart=/usr/bin/python3 /home/zane49/IRIS.py
WorkingDirectory=/home/zane49
User=zane49
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
```
```bash
sudo systemctl daemon-reload
sudo systemctl enable IRIS
sudo systemctl start IRIS
```

---

##  Usage

| Button | Action |
|---|---|
| Short press | OCR — reads all visible text aloud |
| Long press (2s+) | Full scene description via Gemini |
| Background (always) | TFLite continuously detects and announces objects |

---

##  Performance

| Metric                 | Online Mode          | Offline Mode |
| Time to first audio    | ~3s (TFLite)         | ~3s |
| Full description       | ~5s (Gemini)         | Object labels only |
| OCR                    | ✅ (Gemini)          | ❌ |
| Works without internet | ✅ (TFLite fallback)| ✅ |
| Daily limit            | 250 req/day (free)   | Unlimited |

---

##  Key Engineering Notes

- **Piper pip illegal instruction** — The `piper-tts` pip package crashes with SIGILL on the Zero 2W due to incompatible CPU instructions. Solution: use the pre-compiled aarch64 binary directly via subprocess.
- **TFLite load time** — Loading the interpreter per inference takes 8-10s. Loading once at startup keeps inference at 2-4s.
- **Speech queue** — All TTS output goes through a single thread queue to prevent overlapping audio from concurrent TFLite and Gemini threads.
- **Gemini active flag** — TFLite detection pauses while Gemini is processing to prevent camera resource conflicts.

---

##  Future Scope

- **Coral USB Accelerator** — 10x faster local inference via Google Edge TPU
- **Ollama Home PC Mode** — Route to local vision LLM over home Wi-Fi for unlimited free descriptions
- **Regional Language TTS** — Hindi/Marathi Piper voice models for non-English users
- **Currency Detection** — Dedicated prompt for Indian banknote identification
- **Haptic Feedback** — Vibration motor for tactile button confirmation
- **Battery Gauge** — INA219 on I2C to announce remaining charge
- **GPS Integration** — Location-aware scene descriptions

---

##  File Structure

```
~/
├── IRIS.py                          # Main application
├── detect.tflite                    # MobileNetV2 SSD COCO model
├── labelmap.txt                     # 80 COCO class labels
└── piper/
    ├── piper                        # Piper TTS binary
    └── voices/
        ├── en_US-amy-low.onnx       # Voice model
        └── en_US-amy-low.onnx.json  # Voice config
```

---

##  Author

**Zane Miguel Angelo De Souza**


---

##  License

MIT License — free to use, modify, and distribute with attribution.
```
