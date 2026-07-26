import socket
import io
import time
import threading
import subprocess
import base64
import json
import requests
import numpy as np
import RPi.GPIO as GPIO
import tflite_runtime.interpreter as tflite
from picamera2 import Picamera2
from PIL import Image

# ─── CONFIG ───────────────────────────────────────────────
GEMINI_KEY   = "YOUR API KEY HERE"
BUTTON_PIN   = 17
LONG_PRESS_S = 1.5
COOLDOWN     = 2
TFLITE_CONF  = 0.55
BLACKLIST    = {
    'skateboard', 'snowboard', 'skis', 'frisbee', 'kite',
    'surfboard', 'tennis racket', 'baseball'}
MAX_OBJECTS  = 4

PIPER_BIN    = "/home/zane49/piper/piper"
PIPER_MODEL  = "/home/zane49/piper/voices/en_US-amy-low.onnx"
TFLITE_MODEL = "/home/zane49/detect.tflite"
LABELMAP     = "/home/zane49/labelmap.txt"

GEMINI_URL   = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={GEMINI_KEY}"

# ─── LOAD TFLITE ──────────────────────────────────────────
interpreter = tflite.Interpreter(model_path=TFLITE_MODEL)
interpreter.allocate_tensors()
inp_details = interpreter.get_input_details()
out_details = interpreter.get_output_details()

with open(LABELMAP) as f:
    labels = [l.strip() for l in f.readlines()]

# ─── LOAD CAMERA ──────────────────────────────────────────
cam = Picamera2()
cam.configure(cam.create_still_configuration(main={"size": (640, 480)}))
cam.start()
time.sleep(3)

# ─── GPIO ─────────────────────────────────────────────────
GPIO.setmode(GPIO.BCM)
GPIO.setup(BUTTON_PIN, GPIO.IN, pull_up_down=GPIO.PUD_UP)

last_press = 0
gemini_active = False
gemini_lock = threading.Lock()
speech_queue = __import__('queue').Queue()

# ─── FUNCTIONS ────────────────────────────────────────────
def is_online():
    try:
        socket.setdefaulttimeout(2)
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.connect(("8.8.8.8", 53))
        s.close()
        return True
    except:
        return False

def capture_image():
    buf = io.BytesIO()
    cam.capture_file(buf, format='jpeg')
    buf.seek(0)
    return buf.read()

def speak(text, priority=False):
    print(f"[SPEAK] {text}")
    if priority:
        # Clear queue and speak immediately
        while not speech_queue.empty():
            try:
                speech_queue.get_nowait()
                speech_queue.task_done()
            except:
                break
    speech_queue.put(text)

def speech_worker():
    print("[SPEECH WORKER] Started")
    # Start Piper once and keep it running
    piper_proc = subprocess.Popen(
        [PIPER_BIN, '--model', PIPER_MODEL, '--output_raw', '--length_scale', '1.3'],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL
    )
    aplay_proc = subprocess.Popen(
        ['aplay', '-r', '22050', '-f', 'S16_LE', '-t', 'raw', '-B', '500000'],
        stdin=piper_proc.stdout, stderr=subprocess.DEVNULL
    )
    print("[SPEECH WORKER] Piper loaded and ready")

    while True:
        text = speech_queue.get()
        print(f"[SPEECH WORKER] Speaking: {text}")
        try:
            piper_proc.stdin.write((text + '\n').encode())
            piper_proc.stdin.flush()
            time.sleep(len(text) * 0.06 + 1.0)  # Estimate speech duration
        except Exception as e:
            print(f"[SPEAK ERROR] {e}")
        finally:
            speech_queue.task_done()

def call_gemini(image_bytes, prompt):
    b64 = base64.b64encode(image_bytes).decode()
    body = {
        "contents": [{"parts": [
            {"inline_data": {"mime_type": "image/jpeg", "data": b64}},
            {"text": prompt}
        ]}],
        "generationConfig": {"maxOutputTokens": 500}
    }
    r = requests.post(GEMINI_URL, json=body, timeout=15)
    return r.json()["candidates"][0]["content"]["parts"][0]["text"]

def run_tflite(image_bytes):
    input_size = inp_details[0]['shape'][1]
    img = Image.open(io.BytesIO(image_bytes)).resize((input_size, input_size))
    arr = np.expand_dims(np.array(img, dtype=np.uint8), axis=0)
    interpreter.set_tensor(inp_details[0]['index'], arr)
    interpreter.invoke()

    scores  = interpreter.get_tensor(out_details[2]['index'])[0]
    classes = interpreter.get_tensor(out_details[1]['index'])[0].astype(int)


    detected = list(dict.fromkeys(
        labels[c] for i, c in enumerate(classes)
        if scores[i] > TFLITE_CONF and 0 <= c < len(labels)
    ))[:MAX_OBJECTS]
    return detected


# ─── CONTINUOUS TFLITE LOOP ───────────────────────────────
def continuous_detection():
    global gemini_active
    last_spoken = []
    while True:
        try:
            if gemini_active:
                last_spoken = []
                time.sleep(4)
                continue
            img = capture_image()
            objects = run_tflite(img)
            unique = sorted(set(o for o in objects if o not in BLACKLIST))
            if unique and unique != last_spoken:
                # Clear any queued TFLite speech before adding new
                while not speech_queue.empty():
                    try:
                        speech_queue.get_nowait()
                        speech_queue.task_done()
                    except:
                        break
                speak("I can see " + ", ".join(unique), priority=False)
                last_spoken = unique
            time.sleep(1.5)
        except Exception as e:
            print(f"[TFLITE ERROR] {e}")
            time.sleep(2)
def handle_ocr():
    global last_press, gemini_active
    if not gemini_lock.acquire(blocking=False):
        print("[GEMINI] Already running, ignoring press")
        return
    try:
        if time.time() - last_press < COOLDOWN:
            return
        last_press = time.time()
        gemini_active = True
        time.sleep(0.5)
        while not speech_queue.empty():
            try:
                speech_queue.get_nowait()
                speech_queue.task_done()
            except:
                break
        img = capture_image()
        if is_online():
            speak("Reading text.")
            try:
                text = call_gemini(img, "Read aloud all visible text in this image exactly as written. If no text is visible, say: No readable text found.")
                speak(text)
            except Exception as e:
                print(f"[GEMINI ERROR] {e}")
                speak("Could not read text. Please try again.")
        else:
            speak("No internet. Text reading unavailable.")
    finally:
        gemini_active = False
        gemini_lock.release()

def on_button(channel):
    press_time = time.time()
    print("[BUTTON] Pressed")
    while GPIO.input(BUTTON_PIN) == GPIO.LOW:
        time.sleep(0.05)
    duration = time.time() - press_time
    print(f"[BUTTON] Released after {duration:.2f}s")
    if duration >= LONG_PRESS_S:
        print("[BUTTON] Long press - Scene description")
        threading.Thread(target=handle_long_press).start()
    else:
        print("[BUTTON] Short press - OCR")
        threading.Thread(target=handle_ocr).start()
def handle_long_press():
    global last_press, gemini_active
    if not gemini_lock.acquire(blocking=False):
        print("[GEMINI] Already running, ignoring press")
        return
    try:
        if time.time() - last_press < COOLDOWN:
            return
        last_press = time.time()
        gemini_active = True
        time.sleep(0.5)
        # Clear any pending TFLite speech
        while not speech_queue.empty():
            try:
                speech_queue.get_nowait()
                speech_queue.task_done()
            except:
                break

        img = capture_image()
        if is_online():
            speak("Getting full description.", priority=True)
            try:
                desc = call_gemini(img, "In 1 sentence, describe who or what is directly in front of the camera.")
                speak(desc)
            except Exception as e:
                print(f"[GEMINI ERROR] {e}")
                speak("Could not reach Gemini. Please try again.")
        else:
            speak("No internet. Cannot get full description.")
    finally:
        gemini_active = False
        gemini_lock.release()


# ─── MAIN ─────────────────────────────────────────────────
speech_thread = threading.Thread(target=speech_worker, daemon=True)
speech_thread.start()

speak("IRIS ready", priority=True)
print("[IRIS] Running.")

detection_thread = threading.Thread(target=continuous_detection, daemon=True)
detection_thread.start()

last_state = GPIO.HIGH
try:
    while True:
        current_state = GPIO.input(BUTTON_PIN)
        if current_state == GPIO.LOW and last_state == GPIO.HIGH:
            threading.Thread(target=on_button, args=(BUTTON_PIN,)).start()
        last_state = current_state
        time.sleep(0.05)
except KeyboardInterrupt:
    print("[IRIS] Shutting down.")
    GPIO.cleanup()
