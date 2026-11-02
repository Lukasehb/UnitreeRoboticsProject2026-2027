#!/usr/bin/env python3
"""
Headless gezichtsherkenning + "auto-opslaan bij onbekende persoon" (OpenCV YuNet + SFace)

AANGEPAST:
- Het script gebruikt enkel de lokale camera via --cam.
- Externe camera-URL's zijn verwijderd.
- Wanneer detection_control.json aangeeft dat detection_enabled false is, wordt de camera vrijgegeven met cap.release().
- Terwijl detectie uit staat, blijft het script draaien en wordt detection_control.json periodiek gecontroleerd.
- Wanneer detection_enabled opnieuw true wordt, wordt de lokale camera opnieuw geopend.

TTS (Piper):
- Standaard voice: nl_BE-nathalie-medium.onnx (+ .json)
"""

import os
import time
import argparse
import urllib.request
import numpy as np
import cv2
import multiprocessing as mp
import signal
import subprocess
import queue as pyqueue
import sys
import json
import re
from datetime import datetime

YUNET_URL = "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx"
SFACE_URL = "https://github.com/opencv/opencv_zoo/raw/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SAFE_PERSON_RE = re.compile(r"[^A-Za-z0-9_-]+")

DEFAULT_DETECTION_CONTROL_PATH = os.environ.get(
    "FACEASSIST_DETECTION_CONTROL",
    os.path.join(BASE_DIR, "detection_control.json"),
)


# -----------------------------
# Helpers
# -----------------------------

def download_if_missing(url: str, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if os.path.exists(path):
        return

    print(f"[INFO] Downloaden: {os.path.basename(path)} ...", flush=True)
    urllib.request.urlretrieve(url, path)
    print(f"[OK] Opgeslagen naar {path}", flush=True)


def load_known(known_dir: str):
    known = {}

    if not os.path.isdir(known_dir):
        return known

    for fn in os.listdir(known_dir):
        if fn.lower().endswith(".npz"):
            name = os.path.splitext(fn)[0]
            data = np.load(os.path.join(known_dir, fn))
            feats = data["features"].astype(np.float32)
            known[name] = feats

    return known


def largest_face(faces: np.ndarray):
    if faces is None or len(faces) == 0:
        return None

    areas = faces[:, 2] * faces[:, 3]
    return faces[int(np.argmax(areas))]


def normalize_match_feature(feat):
    if feat is None:
        return None

    arr = np.asarray(feat, dtype=np.float32)
    if arr.size == 0:
        return None

    return arr.reshape(1, -1)


def best_match(recognizer, feat, known: dict):
    feat_match = normalize_match_feature(feat)
    if feat_match is None:
        return None, -1.0, -1.0

    scores = []

    for name, feats in known.items():
        best = -1.0

        for f in feats:
            f_match = normalize_match_feature(f)

            if f_match is None or f_match.shape != feat_match.shape:
                continue

            s = float(
                recognizer.match(
                    feat_match,
                    f_match,
                    cv2.FaceRecognizerSF_FR_COSINE,
                )
            )

            if s > best:
                best = s

        if best > -1.0:
            scores.append((name, best))

    if not scores:
        return None, -1.0, -1.0

    scores.sort(key=lambda x: x[1], reverse=True)

    best_name, best_score = scores[0]
    second_score = scores[1][1] if len(scores) > 1 else -1.0

    return best_name, best_score, second_score


def face_direction_nl(x: int, w_face: int, frame_w: int) -> str:
    cx = x + (w_face // 2)

    if cx < frame_w / 3:
        return "is links van je"
    elif cx > 2 * frame_w / 3:
        return "is rechts van je"

    return "staat voor je"


def normalize_qr_text(text: str) -> str:
    return " ".join(str(text or "").split())


def limit_tts_text(text: str, max_chars: int) -> str:
    if max_chars <= 0 or len(text) <= max_chars:
        return text

    return text[:max_chars].rstrip() + "..."


def decode_qr_codes(qr_detector, frame):
    decoded = []

    if hasattr(qr_detector, "detectAndDecodeMulti"):
        try:
            ok, decoded_info, _, _ = qr_detector.detectAndDecodeMulti(frame)
            if ok:
                decoded.extend(normalize_qr_text(item) for item in decoded_info)
        except Exception:
            pass

    if not any(decoded):
        try:
            text, _, _ = qr_detector.detectAndDecode(frame)
            decoded.append(normalize_qr_text(text))
        except Exception:
            pass

    unique = []
    seen = set()

    for item in decoded:
        if item and item not in seen:
            unique.append(item)
            seen.add(item)

    return unique


def open_camera_linux(cam_index: int, width: int, height: int, fps: int):
    dev = f"/dev/video{cam_index}"

    gst_pipeline = (
        f"v4l2src device={dev} ! "
        f"image/jpeg,width={width},height={height},framerate={fps}/1 ! "
        f"jpegdec ! videoconvert ! appsink drop=true sync=false max-buffers=1"
    )

    cap = cv2.VideoCapture(gst_pipeline, cv2.CAP_GSTREAMER)
    if cap.isOpened():
        print("[INFO] Camera geopend via GStreamer.", flush=True)
        return cap

    cap = cv2.VideoCapture(cam_index, cv2.CAP_V4L2)
    if cap.isOpened():
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        cap.set(cv2.CAP_PROP_FPS, fps)
        print("[INFO] Camera geopend via V4L2 (OpenCV).", flush=True)
        return cap

    cap = cv2.VideoCapture(cam_index)
    if cap.isOpened():
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        cap.set(cv2.CAP_PROP_FPS, fps)
        print("[INFO] Camera geopend via standaard backend (OpenCV).", flush=True)
        return cap

    return cap


def str2bool(s: str) -> bool:
    return str(s).strip().lower() in ("1", "true", "yes", "y", "on", "ja", "j")


def sanitize_name(name: str) -> str:
    name = name.strip().replace("/", "_").replace("\\", "_")
    name = name.replace("..", ".")
    return name


def sanitize_person_name(name: str) -> str:
    cleaned = SAFE_PERSON_RE.sub("_", normalize_qr_text(name))
    cleaned = cleaned.strip("_")
    return cleaned or "qr_person"


def unique_path(path: str) -> str:
    if not os.path.exists(path):
        return path

    base, ext = os.path.splitext(path)
    n = 1

    while True:
        candidate = f"{base}_{n}{ext}"
        if not os.path.exists(candidate):
            return candidate
        n += 1


def ask_input(prompt: str) -> str:
    sys.stdout.write(prompt)
    sys.stdout.flush()
    return input()


# -----------------------------
# Foto snapshot
# -----------------------------

def save_person_snapshot(frame, name: str, out_dir: str = "snapshots") -> str:
    os.makedirs(out_dir, exist_ok=True)

    safe_name = sanitize_name(name) if name else "Onbekend"
    ts = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    path = os.path.join(out_dir, f"{safe_name}_{ts}.jpg")

    cv2.imwrite(path, frame)

    return path


# -----------------------------
# Unknown foto opslag
# -----------------------------

def save_unknown_photo(frame, face_row, out_dir: str, idx: int) -> str:
    os.makedirs(out_dir, exist_ok=True)

    x, y, fw, fh = face_row[:4].astype(int)
    h, w = frame.shape[:2]

    pad_w = int(fw * 0.15)
    pad_h = int(fh * 0.15)

    x1 = max(0, x - pad_w)
    y1 = max(0, y - pad_h)
    x2 = min(w, x + fw + pad_w)
    y2 = min(h, y + fh + pad_h)

    crop = frame[y1:y2, x1:x2]

    if crop is None or crop.size == 0:
        raise RuntimeError("Lege face-crop")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    path = os.path.join(out_dir, f"{ts}_{idx:04d}.jpg")

    cv2.imwrite(path, crop)

    return path


def save_known_qr_photo(frame, face_row, known_dir: str, person: str, idx: int) -> str:
    person_dir = os.path.join(known_dir, person)
    os.makedirs(person_dir, exist_ok=True)

    x, y, fw, fh = face_row[:4].astype(int)
    h, w = frame.shape[:2]

    pad_w = int(fw * 0.15)
    pad_h = int(fh * 0.15)

    x1 = max(0, x - pad_w)
    y1 = max(0, y - pad_h)
    x2 = min(w, x + fw + pad_w)
    y2 = min(h, y + fh + pad_h)

    crop = frame[y1:y2, x1:x2]

    if crop is None or crop.size == 0:
        raise RuntimeError("Lege QR face-crop")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    path = unique_path(os.path.join(person_dir, f"{ts}_qr_{idx:04d}.jpg"))

    cv2.imwrite(path, crop)

    return path


def load_npz_features(npz_path: str):
    if not os.path.isfile(npz_path):
        return None

    data = np.load(npz_path, allow_pickle=True)

    keys = list(data.keys())
    key = "features" if "features" in data else (keys[0] if keys else None)

    if key is None:
        return None

    arr = np.asarray(data[key], dtype=np.float32)

    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    elif arr.ndim > 2:
        arr = arr.reshape(arr.shape[0], -1)

    if arr.ndim != 2 or arr.shape[0] == 0:
        return None

    return arr


def normalize_face_feature(feat):
    if feat is None:
        return None

    arr = np.asarray(feat, dtype=np.float32)

    if arr.size == 0:
        return None

    return arr.reshape(-1)


def append_known_features(known_dir: str, person: str, new_features: list) -> int:
    if not new_features:
        return 0

    normalized = []

    for feat in new_features:
        feat_1d = normalize_face_feature(feat)
        if feat_1d is not None:
            normalized.append(feat_1d)

    if not normalized:
        return 0

    os.makedirs(known_dir, exist_ok=True)

    npz_path = os.path.join(known_dir, f"{person}.npz")
    new_stack = np.stack(normalized, axis=0).astype(np.float32)

    old = load_npz_features(npz_path)

    if old is not None:
        merged = np.concatenate([old, new_stack], axis=0)
    else:
        merged = new_stack

    np.savez_compressed(npz_path, features=merged)

    return int(new_stack.shape[0])


def play_qr_click(duration_ms: int = 70, frequency: int = 1200, volume: float = 0.35) -> None:
    try:
        sample_rate = 16000
        sample_count = max(1, int(sample_rate * max(10, duration_ms) / 1000.0))

        t = np.arange(sample_count, dtype=np.float32)
        audio = np.sin((2.0 * np.pi * float(frequency) * t) / sample_rate)

        ramp_len = min(sample_count // 2, int(sample_rate * 0.005))

        if ramp_len > 0:
            ramp = np.linspace(0.0, 1.0, ramp_len, dtype=np.float32)
            audio[:ramp_len] *= ramp
            audio[-ramp_len:] *= ramp[::-1]

        pcm = np.clip(
            audio * 32767.0 * max(0.0, min(1.0, float(volume))),
            -32768,
            32767,
        )

        raw_audio = pcm.astype(np.int16).tobytes()

        result = subprocess.run(
            ["aplay", "-q", "-r", str(sample_rate), "-f", "S16_LE", "-t", "raw", "-"],
            input=raw_audio,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=2,
            check=False,
        )

        if result.returncode != 0:
            print("\a", end="", flush=True)

    except Exception:
        try:
            print("\a", end="", flush=True)
        except Exception:
            pass


# -----------------------------
# Piper TTS
# -----------------------------

def read_piper_sample_rate(model_path: str, default_rate: int = 22050) -> int:
    json_path = model_path + ".json"

    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        for key in ("sample_rate", "audio.sample_rate", "audio_sample_rate"):
            if key in data and isinstance(data[key], int):
                return int(data[key])

        if isinstance(data.get("audio"), dict) and isinstance(data["audio"].get("sample_rate"), int):
            return int(data["audio"]["sample_rate"])

    except Exception:
        pass

    return default_rate


def piper_say(
    text: str,
    model_path: str,
    sample_rate: int,
    length_scale: float = 1.0,
    volume: int = 100,
):
    p1 = subprocess.Popen(
        [
            "/home/jetson/piper/piper",
            "--model",
            model_path,
            "--output_raw",
            "--length_scale",
            str(length_scale),
        ],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
    )

    try:
        raw_audio, _ = p1.communicate(input=(text + "\n").encode("utf-8"), timeout=60)
    except subprocess.TimeoutExpired:
        p1.kill()
        return

    if not raw_audio:
        return

    vol = max(0, min(100, int(volume)))

    if vol < 100:
        pcm = np.frombuffer(raw_audio, dtype=np.int16).astype(np.float32)
        pcm *= vol / 100.0
        np.clip(pcm, -32768, 32767, out=pcm)
        raw_audio = pcm.astype(np.int16).tobytes()

    p2 = subprocess.Popen(
        ["aplay", "-r", str(sample_rate), "-f", "S16_LE", "-t", "raw", "-"],
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    try:
        p2.communicate(input=raw_audio, timeout=60)
    except subprocess.TimeoutExpired:
        p2.kill()


def tts_worker_loop(tts_queue: mp.Queue, stop_event: mp.Event, args, done_queue: mp.Queue = None):
    signal.signal(signal.SIGINT, signal.SIG_IGN)

    try:
        subprocess.run(["piper", "--help"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
    except FileNotFoundError:
        print("[WAARSCHUWING] 'piper' niet gevonden in PATH.", flush=True)
        return

    model_path = os.path.expanduser(args.piper_model)

    if not os.path.exists(model_path):
        print(f"[WAARSCHUWING] Piper model niet gevonden: {model_path}", flush=True)
        return

    sample_rate = args.piper_rate

    if args.piper_rate_auto:
        sample_rate = read_piper_sample_rate(model_path, default_rate=args.piper_rate)

    while not stop_event.is_set():
        try:
            msg = tts_queue.get(timeout=0.1)
        except pyqueue.Empty:
            continue

        if msg is None:
            break

        done_token = None

        if isinstance(msg, dict):
            text = str(msg.get("text", "")).strip()
            done_token = msg.get("done_token")
        else:
            text = str(msg).strip()

        if not text:
            if done_token and done_queue is not None:
                try:
                    done_queue.put_nowait(done_token)
                except pyqueue.Full:
                    pass
            continue

        try:
            piper_say(
                text,
                model_path=model_path,
                sample_rate=sample_rate,
                length_scale=args.piper_length_scale,
                volume=args.voice_volume,
            )
        except Exception:
            pass
        finally:
            if done_token and done_queue is not None:
                try:
                    done_queue.put_nowait(done_token)
                except pyqueue.Full:
                    pass


def tts_enqueue(tts_queue, text: str, done_token=None) -> bool:
    if tts_queue is None:
        return False

    msg = {"text": text, "done_token": done_token} if done_token else text

    try:
        tts_queue.put_nowait(msg)
        return True
    except pyqueue.Full:
        return False


class DetectionControl:
    def __init__(self, path: str, poll_interval: float = 10.0, default_enabled: bool = True):
        self.path = os.path.abspath(path) if path else ""
        self.poll_interval = max(0.1, float(poll_interval))
        self.default_enabled = bool(default_enabled)
        self._last_checked = 0.0
        self._enabled = bool(default_enabled)

    def enabled(self) -> bool:
        now = time.time()

        if (now - self._last_checked) < self.poll_interval:
            return self._enabled

        self._last_checked = now

        if not self.path or not os.path.isfile(self.path):
            self._enabled = self.default_enabled
            return self._enabled

        try:
            with open(self.path, "r", encoding="utf-8") as f:
                data = json.load(f)

            if isinstance(data, dict):
                self._enabled = bool(
                    data.get(
                        "detection_enabled",
                        data.get("enabled", self.default_enabled),
                    )
                )
            else:
                self._enabled = self.default_enabled

        except Exception as exc:
            print(f"[WAARSCHUWING] Detectie-control lezen mislukt: {exc}", flush=True)
            self._enabled = self.default_enabled

        return self._enabled


def drain_done_queue(done_queue) -> set:
    tokens = set()

    if done_queue is None:
        return tokens

    while True:
        try:
            tokens.add(done_queue.get_nowait())
        except pyqueue.Empty:
            break

    return tokens


# -----------------------------
# Snapshot opslag
# -----------------------------

def save_snapshot(frame, out_dir: str, tag: str) -> str:
    os.makedirs(out_dir, exist_ok=True)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_tag = sanitize_name(tag) if tag else "onbekend"
    path = os.path.join(out_dir, f"{ts}_{safe_tag}.jpg")

    cv2.imwrite(path, frame)

    return path


# -----------------------------
# Main
# -----------------------------

def main():
    ap = argparse.ArgumentParser()

    # Camera + detectie
    ap.add_argument("--cam", type=int, default=0)
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--fps", type=int, default=15)
    ap.add_argument("--infer_every", type=int, default=2)

    ap.add_argument("--min_face", type=int, default=50)
    ap.add_argument("--score_th", type=float, default=0.9)
    ap.add_argument("--nms_th", type=float, default=0.3)
    ap.add_argument("--topk", type=int, default=5000)

    # Herkenning
    ap.add_argument("--threshold", type=float, default=0.60)
    ap.add_argument("--margin", type=float, default=0.06)

    # Entry / leave
    ap.add_argument("--lost_timeout", type=float, default=1.0)
    ap.add_argument("--enter_confirm_frames", type=int, default=3)
    ap.add_argument("--reannounce_after", type=float, default=6.0)

    # Onbekend gedrag
    ap.add_argument(
        "--unknown_seconds",
        type=float,
        default=5.0,
        help="(oud) Als een onbekend gezicht zo lang zichtbaar blijft, vragen om op te slaan.",
    )
    ap.add_argument(
        "--unknown_confirm_frames",
        type=int,
        default=5,
        help="Aantal opeenvolgende 'onbekend' frames voor we starten.",
    )
    ap.add_argument(
        "--cooldown_after_unknown",
        type=float,
        default=300.0,
        help="Na een onbekend-afhandeling even wachten.",
    )

    ap.add_argument(
        "--unknown_capture_interval",
        type=float,
        default=0.5,
        help="(oud) Tijdens onbekend: neem een feature-snapshot om de N seconden.",
    )
    ap.add_argument(
        "--unknown_max_snaps",
        type=int,
        default=60,
        help="(oud) Max aantal feature-snapshots dat we bijhouden.",
    )

    # Opslag
    ap.add_argument(
        "--known",
        type=str,
        default=os.path.join(BASE_DIR, "known"),
        help="Map met .npz identiteiten",
    )
    ap.add_argument(
        "--min_save_samples",
        type=int,
        default=20,
        help="(oud) Niet opslaan als er te weinig snapshots zijn",
    )

    # Foto's
    ap.add_argument(
        "--unknown_photos",
        type=str,
        default="unknown_photos",
        help="(oud) Map om optioneel een JPG te bewaren",
    )
    ap.add_argument(
        "--save_unknown_snapshot",
        action="store_true",
        help="(oud) Bewaar ook 1 JPG wanneer je een nieuwe persoon opslaat.",
    )

    # QR-code scanner
    ap.add_argument("--no_qr", action="store_true", help="QR-code scanner uitschakelen.")
    ap.add_argument("--qr_every", type=int, default=5, help="Scan elke N frames op QR-codes.")
    ap.add_argument(
        "--qr_cooldown",
        type=float,
        default=8.0,
        help="Aantal seconden voordat dezelfde QR-code opnieuw wordt uitgesproken.",
    )
    ap.add_argument(
        "--qr_max_chars",
        type=int,
        default=500,
        help="Maximaal aantal QR-tekens voor TTS. Gebruik 0 voor onbeperkt.",
    )
    ap.add_argument(
        "--qr_prefix",
        type=str,
        default=(
            "Dank je wel om je te registreren. "
            "Ga enkele seconden in de deuropening staan met je gezicht naar de camera, "
            "je zal worden geregistreerd onder de naam: "
        ),
        help="Tekst die voor de QR-inhoud wordt uitgesproken.",
    )
    ap.add_argument(
        "--qr_countdown",
        type=int,
        default=10,
        help="Aantal seconden aftellen na het uitspreken van de QR-inhoud.",
    )
    ap.add_argument(
        "--qr_photo_count",
        type=int,
        default=5,
        help="Aantal foto's om te bewaren in known/<QR-naam>/ na de countdown.",
    )
    ap.add_argument(
        "--qr_capture_interval",
        type=float,
        default=0.5,
        help="Tijd tussen QR-registratiefoto's in seconden.",
    )
    ap.add_argument(
        "--no_qr_clicks",
        action="store_true",
        help="Geen korte klik/beep afspelen bij start capture en per QR-foto.",
    )

    # Piper TTS
    ap.add_argument("--no_tts", action="store_true")
    ap.add_argument("--speak", type=str, default="True")
    ap.add_argument(
        "--piper_model",
        type=str,
        default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "voices", "nl_BE-nathalie-medium.onnx"),
    )
    ap.add_argument("--piper_rate", type=int, default=22050)
    ap.add_argument("--piper_rate_auto", action="store_true")
    ap.add_argument("--piper_length_scale", type=float, default=1.0)
    ap.add_argument("--voice_volume", type=int, default=100)
    ap.add_argument("--tts_queue_size", type=int, default=20)

    # Detectie-control
    ap.add_argument(
        "--control_file",
        type=str,
        default=DEFAULT_DETECTION_CONTROL_PATH,
        help="JSON-bestand waarmee detectie aan/uit gezet wordt zonder het proces te stoppen.",
    )
    ap.add_argument(
        "--control_poll_interval",
        type=float,
        default=10.0,
        help="Aantal seconden tussen checks van het detectie-controlbestand.",
    )

    args = ap.parse_args()

    args.voice_volume = max(0, min(100, int(args.voice_volume)))
    args.qr_every = max(1, int(args.qr_every))
    args.qr_max_chars = max(0, int(args.qr_max_chars))
    args.qr_countdown = max(0, int(args.qr_countdown))
    args.qr_photo_count = max(1, int(args.qr_photo_count))
    args.qr_capture_interval = max(0.0, float(args.qr_capture_interval))
    args.control_poll_interval = max(0.1, float(args.control_poll_interval))

    yunet_path = os.path.join("models", "face_detection_yunet_2023mar.onnx")
    sface_path = os.path.join("models", "face_recognition_sface_2021dec.onnx")

    download_if_missing(YUNET_URL, yunet_path)
    download_if_missing(SFACE_URL, sface_path)

    os.makedirs(args.known, exist_ok=True)

    # TTS
    stop_event = mp.Event()
    tts_queue = None
    tts_done_queue = None
    tts_proc = None

    speak_enabled = (not args.no_tts) and str2bool(args.speak)

    if speak_enabled:
        tts_queue = mp.Queue(maxsize=args.tts_queue_size)
        tts_done_queue = mp.Queue(maxsize=args.tts_queue_size)

        tts_proc = mp.Process(
            target=tts_worker_loop,
            args=(tts_queue, stop_event, args, tts_done_queue),
            daemon=True,
        )
        tts_proc.start()

        if args.no_qr:
            tts_enqueue(tts_queue, "Gezichtsherkenning is gestart.")
        else:
            tts_enqueue(tts_queue, "Gezichtsherkenning en QR scanner zijn gestart.")

    cap = None

    # Camera
    cap = open_camera_linux(args.cam, args.width, args.height, args.fps)

    if cap is None or not cap.isOpened():
        print("[FOUT] Kan camera niet openen.", flush=True)
        stop_event.set()

        if tts_queue is not None:
            try:
                tts_queue.put_nowait(None)
            except Exception:
                pass

        return

    ok, frame = cap.read()

    if not ok or frame is None:
        print("[FOUT] Kan eerste frame niet lezen.", flush=True)

        if cap is not None:
            cap.release()
            cap = None

        return

    h, w = frame.shape[:2]

    detector = cv2.FaceDetectorYN.create(
        yunet_path,
        "",
        (w, h),
        args.score_th,
        args.nms_th,
        args.topk,
    )
    recognizer = cv2.FaceRecognizerSF.create(sface_path, "")

    qr_detector = None
    qr_enabled = not args.no_qr

    if qr_enabled:
        try:
            qr_detector = cv2.QRCodeDetector()
            print("[INFO] QR-scanner actief.", flush=True)
        except Exception as e:
            qr_enabled = False
            print(f"[WAARSCHUWING] QR-scanner kon niet starten: {e}", flush=True)

    known = load_known(args.known)

    if known:
        print("[INFO] Bekend:", ", ".join(sorted(known.keys())), flush=True)

        if speak_enabled:
            tts_enqueue(tts_queue, f"{len(known)} personen geladen.")
    else:
        print(f"[WAARSCHUWING] Geen bekende identiteiten in '{args.known}'.", flush=True)

        if speak_enabled:
            tts_enqueue(tts_queue, "Ik ken nog niemand.")

    # Entry/leave state
    present = False
    present_name = None
    last_seen = 0.0

    consec_needed = args.enter_confirm_frames
    consec_count = 0
    candidate_name = None
    last_announced_at = {}

    # Unknown state
    unknown_consec = 0
    unknown_started_at = None
    last_unknown_handled_at = 0.0

    unknown_dir = "unknown"
    unknown_photo_count = 0
    unknown_photo_interval = 60
    unknown_last_photo_at = 0.0

    last_person_photo_at = {}
    person_photo_cooldown = 300.0

    # QR-code state
    last_qr_announced_at = {}
    qr_registration = None
    qr_registration_seq = 0

    frame_id = 0

    detection_control = DetectionControl(
        args.control_file,
        args.control_poll_interval,
        default_enabled=True,
    )

    detection_paused = False

    print(f"[INFO] Detectie-controlbestand: {detection_control.path}", flush=True)
    print(f"[INFO] Detectie-control poll interval: {args.control_poll_interval:.1f}s", flush=True)
    print("[INFO] Headless actief. Ctrl+C om te stoppen.", flush=True)

    try:
        while True:
            if not detection_control.enabled():
                if not detection_paused:
                    print(
                        "[INFO] Detectie gepauzeerd via configuration. Camera wordt vrijgegeven.",
                        flush=True,
                    )

                    if cap is not None:
                        cap.release()
                        cap = None
                        print("[INFO] Camera vrijgegeven.", flush=True)

                    present = False
                    present_name = None
                    last_seen = 0.0
                    consec_count = 0
                    candidate_name = None
                    unknown_consec = 0
                    unknown_started_at = None
                    unknown_dir = "unknown"
                    unknown_photo_count = 0
                    unknown_last_photo_at = 0.0
                    qr_registration = None
                    detection_paused = True

                drain_done_queue(tts_done_queue)
                time.sleep(args.control_poll_interval)
                continue

            if detection_paused:
                print(
                    "[INFO] Detectie hervat via configuration. Camera wordt opnieuw geopend.",
                    flush=True,
                )

                cap = open_camera_linux(args.cam, args.width, args.height, args.fps)

                if cap is None or not cap.isOpened():
                    print("[FOUT] Kan camera niet opnieuw openen.", flush=True)

                    if cap is not None:
                        cap.release()
                        cap = None

                    time.sleep(args.control_poll_interval)
                    continue

                ok, frame = cap.read()

                if not ok or frame is None:
                    print("[FOUT] Kan eerste frame na hervatten niet lezen.", flush=True)

                    cap.release()
                    cap = None

                    time.sleep(args.control_poll_interval)
                    continue

                h, w = frame.shape[:2]
                detector.setInputSize((w, h))

                detection_paused = False

            if cap is None or not cap.isOpened():
                print("[WAARSCHUWING] Camera is niet beschikbaar. Nieuwe poging volgt.", flush=True)
                time.sleep(args.control_poll_interval)
                continue

            ok, frame = cap.read()

            if not ok or frame is None:
                time.sleep(0.1)
                continue

            frame_id += 1
            now = time.time()

            tts_done_tokens = drain_done_queue(tts_done_queue)

            if qr_registration is not None:
                if qr_registration["state"] == "waiting_speech":
                    if qr_registration.get("speech_token") in tts_done_tokens:
                        qr_registration["state"] = "countdown"
                        qr_registration["next_count"] = args.qr_countdown
                        qr_registration["next_count_at"] = now

                        print(
                            f"[QR] QR-tekst uitgesproken. Countdown start voor {qr_registration['person']}.",
                            flush=True,
                        )

                    elif speak_enabled and tts_proc is not None and not tts_proc.is_alive():
                        qr_registration["state"] = "countdown"
                        qr_registration["next_count"] = args.qr_countdown
                        qr_registration["next_count_at"] = now

                        print(
                            "[WAARSCHUWING] TTS-proces is gestopt; countdown start zonder spraakbevestiging.",
                            flush=True,
                        )

                if qr_registration["state"] == "waiting_countdown_done":
                    if qr_registration.get("countdown_token") in tts_done_tokens:
                        qr_registration["state"] = "capturing"
                        qr_registration["last_capture_at"] = 0.0

                        print(
                            f"[QR] Countdown klaar. Foto's nemen voor {qr_registration['person']}.",
                            flush=True,
                        )

                        if not args.no_qr_clicks:
                            play_qr_click()

                    elif speak_enabled and tts_proc is not None and not tts_proc.is_alive():
                        qr_registration["state"] = "capturing"
                        qr_registration["last_capture_at"] = 0.0

                        print("[WAARSCHUWING] TTS-proces is gestopt; foto's nemen.", flush=True)

                        if not args.no_qr_clicks:
                            play_qr_click()

                if qr_registration["state"] == "countdown" and now >= qr_registration["next_count_at"]:
                    count = qr_registration["next_count"]

                    print(f"[QR] Countdown: {count}", flush=True)

                    if count == 0:
                        token = f"qr-countdown:{qr_registration['id']}"
                        qr_registration["countdown_token"] = token

                        if speak_enabled and tts_enqueue(tts_queue, str(count), done_token=token):
                            qr_registration["state"] = "waiting_countdown_done"
                        else:
                            qr_registration["state"] = "capturing"
                            qr_registration["last_capture_at"] = 0.0

                            print(
                                f"[QR] Countdown klaar. Foto's nemen voor {qr_registration['person']}.",
                                flush=True,
                            )

                            if not args.no_qr_clicks:
                                play_qr_click()
                    else:
                        if speak_enabled:
                            tts_enqueue(tts_queue, str(count))

                        qr_registration["next_count"] = count - 1
                        qr_registration["next_count_at"] = now + 1.0

            if qr_enabled and qr_registration is None and qr_detector is not None and frame_id % args.qr_every == 0:
                for qr_text in decode_qr_codes(qr_detector, frame):
                    last_qr = last_qr_announced_at.get(qr_text, 0.0)

                    if now - last_qr < args.qr_cooldown:
                        continue

                    last_qr_announced_at[qr_text] = now

                    qr_name = sanitize_person_name(qr_text)

                    print(f"[QR] {qr_text}", flush=True)
                    print(f"[QR] Registratie-map: {os.path.join(args.known, qr_name)}", flush=True)

                    qr_registration_seq += 1
                    speech_token = f"qr-speech:{qr_registration_seq}"

                    qr_registration = {
                        "id": qr_registration_seq,
                        "raw_text": qr_text,
                        "person": qr_name,
                        "state": "waiting_speech",
                        "speech_token": speech_token,
                        "countdown_token": None,
                        "next_count": args.qr_countdown,
                        "next_count_at": now,
                        "captured": 0,
                        "features_added": 0,
                        "last_capture_at": 0.0,
                        "last_status_at": 0.0,
                    }

                    if speak_enabled:
                        tts_text = limit_tts_text(qr_text, args.qr_max_chars)

                        spoken = tts_enqueue(
                            tts_queue,
                            f"{args.qr_prefix} {tts_text}".strip(),
                            done_token=speech_token,
                        )

                        if not spoken:
                            qr_registration["state"] = "countdown"
                            print(
                                "[WAARSCHUWING] TTS-wachtrij vol; countdown start zonder QR-spraakbevestiging.",
                                flush=True,
                            )
                    else:
                        qr_registration["state"] = "countdown"
                        print("[QR] TTS staat uit. Countdown start.", flush=True)

                    break

            if qr_registration is not None and qr_registration["state"] != "capturing":
                continue

            if frame_id % args.infer_every != 0:
                continue

            detector.setInputSize((w, h))

            _, faces = detector.detect(frame)
            face = largest_face(faces)

            if face is None:
                if qr_registration is not None and qr_registration["state"] == "capturing":
                    if now - qr_registration.get("last_status_at", 0.0) >= 1.0:
                        print(
                            f"[QR] Wacht op gezicht voor {qr_registration['person']}...",
                            flush=True,
                        )
                        qr_registration["last_status_at"] = now

                    continue

                if present and now - last_seen >= args.lost_timeout:
                    print(f"[INFO] {present_name} is uit beeld.", flush=True)

                    present = False
                    present_name = None
                    consec_count = 0
                    candidate_name = None

                unknown_consec = 0
                unknown_started_at = None
                unknown_dir = "unknown"
                unknown_photo_count = 0
                unknown_last_photo_at = 0.0

                continue

            x, y, fw, fh = face[:4].astype(int)

            if fw < args.min_face:
                if qr_registration is not None and qr_registration["state"] == "capturing":
                    if now - qr_registration.get("last_status_at", 0.0) >= 1.0:
                        print(
                            f"[QR] Gezicht te klein voor {qr_registration['person']} ({fw}px).",
                            flush=True,
                        )
                        qr_registration["last_status_at"] = now

                    continue

                if present and now - last_seen >= args.lost_timeout:
                    print(f"[INFO] {present_name} is uit beeld.", flush=True)

                    present = False
                    present_name = None

                unknown_consec = 0
                unknown_started_at = None
                unknown_dir = "unknown"
                unknown_photo_count = 0
                unknown_last_photo_at = 0.0

                consec_count = 0
                candidate_name = None

                continue

            richting = face_direction_nl(x, fw, w)

            if qr_registration is not None and qr_registration["state"] == "capturing":
                if args.qr_capture_interval <= 0 or now - qr_registration["last_capture_at"] >= args.qr_capture_interval:
                    idx = qr_registration["captured"] + 1

                    try:
                        p = save_known_qr_photo(frame, face, args.known, qr_registration["person"], idx)

                        added_now = 0

                        try:
                            aligned_qr = recognizer.alignCrop(frame, face)
                            feat_qr = recognizer.feature(aligned_qr).astype(np.float32)

                            added_now = append_known_features(
                                args.known,
                                qr_registration["person"],
                                [feat_qr],
                            )

                            qr_registration["features_added"] += added_now

                            if added_now > 0:
                                known = load_known(args.known)

                        except Exception as e:
                            print(f"[WAARSCHUWING] QR feature extractie mislukt: {e}", flush=True)

                        qr_registration["captured"] = idx
                        qr_registration["last_capture_at"] = now

                        if not args.no_qr_clicks:
                            play_qr_click()

                        print(
                            f"[OK] QR foto {idx}/{args.qr_photo_count}: {p} "
                            f"({added_now} feature toegevoegd)",
                            flush=True,
                        )

                    except Exception as e:
                        qr_registration["last_capture_at"] = now
                        print(f"[WAARSCHUWING] QR foto opslaan mislukt: {e}", flush=True)

                    if qr_registration["captured"] >= args.qr_photo_count:
                        known = load_known(args.known)

                        print(
                            f"[INFO] QR-registratie klaar voor {qr_registration['person']}: "
                            f"{qr_registration['captured']} foto('s), "
                            f"{qr_registration['features_added']} feature(s).",
                            flush=True,
                        )

                        if speak_enabled:
                            tts_enqueue(tts_queue, f"Registratie klaar voor {qr_registration['person']}")

                        last_qr_announced_at[qr_registration["raw_text"]] = time.time()

                        qr_registration = None

                continue

            aligned = recognizer.alignCrop(frame, face)
            feat = recognizer.feature(aligned).astype(np.float32)

            if known:
                best_name, best_score, second_score = best_match(recognizer, feat, known)
            else:
                best_name, best_score, second_score = None, -1.0, -1.0

            confident = (
                best_name is not None
                and best_score >= args.threshold
                and (best_score - second_score) >= args.margin
            )

            # -------------------------
            # ONBEKEND
            # -------------------------
            if not confident:
                if present and now - last_seen >= args.lost_timeout:
                    print(f"[INFO] {present_name} is uit beeld.", flush=True)

                    present = False
                    present_name = None
                    consec_count = 0
                    candidate_name = None

                if now - last_unknown_handled_at < args.cooldown_after_unknown:
                    unknown_consec = 0
                    unknown_started_at = None
                    unknown_dir = "unknown"
                    unknown_photo_count = 0
                    unknown_last_photo_at = 0.0

                    continue

                unknown_consec += 1

                if unknown_consec < args.unknown_confirm_frames:
                    continue

                if unknown_started_at is None:
                    unknown_started_at = now
                    unknown_photo_count = 0
                    unknown_last_photo_at = 0.0

                    os.makedirs(unknown_dir, exist_ok=True)

                    print(f"[INFO] Onbekende persoon gedetecteerd -> map: {unknown_dir}", flush=True)

                # Deze foto-opslag staat bewust nog uit, zoals in jouw originele code.
                # if unknown_dir is not None:
                #     if now - unknown_last_photo_at >= unknown_photo_interval:
                #         unknown_photo_count += 1
                #         p = save_unknown_photo(frame, face, unknown_dir, unknown_photo_count)
                #         unknown_last_photo_at = now
                #         print(f"[OK] Unknown foto {unknown_photo_count}/20: {p}", flush=True)

                if unknown_photo_count >= 20:
                    print(f"[INFO] Unknown sessie klaar (20 foto's) -> {unknown_dir}", flush=True)

                    last_unknown_handled_at = time.time()

                    unknown_consec = 0
                    unknown_started_at = None
                    unknown_dir = "unknown"
                    unknown_photo_count = 0
                    unknown_last_photo_at = 0.0

                continue

            # -------------------------
            # BEKEND
            # -------------------------
            last_seen = now

            unknown_consec = 0
            unknown_started_at = None
            unknown_dir = "unknown"
            unknown_photo_count = 0
            unknown_last_photo_at = 0.0

            if present and best_name == present_name:
                continue

            if candidate_name == best_name:
                consec_count += 1
            else:
                candidate_name = best_name
                consec_count = 1

            if consec_count < consec_needed:
                continue

            last_spoke = last_announced_at.get(candidate_name, 0.0)

            if now - last_spoke < args.reannounce_after:
                present = True
                present_name = candidate_name
                consec_count = 0
                candidate_name = None

                continue

            present = True
            present_name = candidate_name

            last_announced_at[present_name] = now

            print(
                f"[INFO] BINNEN: {present_name} {richting} "
                f"(score={best_score:.2f}, tweede={second_score:.2f})",
                flush=True,
            )

            if best_score > args.threshold:
                last_t = last_person_photo_at.get(present_name, 0.0)

                if now - last_t >= person_photo_cooldown:
                    # p = save_person_snapshot(frame, present_name, out_dir="snapshots")
                    # last_person_photo_at[present_name] = now
                    # print("[OK] Snapshot opgeslagen:", p, flush=True)

                    if speak_enabled:
                        tts_enqueue(tts_queue, f"Hallo {present_name}")

                consec_count = 0
                candidate_name = None

    except KeyboardInterrupt:
        print("\n[INFO] Stoppen...", flush=True)

    finally:
        if cap is not None:
            cap.release()

        stop_event.set()

        if tts_queue is not None:
            try:
                tts_queue.put_nowait(None)
            except Exception:
                pass

        if tts_proc is not None:
            tts_proc.join(timeout=1.0)

            if tts_proc.is_alive():
                tts_proc.terminate()
                tts_proc.join()


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()