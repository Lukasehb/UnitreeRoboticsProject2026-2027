from flask import Flask, render_template, send_from_directory, jsonify, redirect, url_for, Response, request
import os
import re
import shutil
import uuid
import base64
import asyncio
from datetime import datetime, timedelta
import subprocess
import threading
from collections import deque
import cv2
import time
import numpy as np
import json
import websockets
import urllib.request
import urllib.error


app = Flask(__name__)

BASE_DIR = os.path.dirname(__file__)
SNAPSHOT_DIR = os.path.join(BASE_DIR, "snapshots")
KNOWN_DIR = os.path.join(BASE_DIR, "known")
UNKNOWN_DIR = os.path.join(BASE_DIR, "unknown")
MODELS_DIR = os.path.join(BASE_DIR, "models")
YUNET_PATH = os.path.join(MODELS_DIR, "face_detection_yunet_2023mar.onnx")
SFACE_PATH = os.path.join(MODELS_DIR, "face_recognition_sface_2021dec.onnx")
FACE_EXTRACT_TMP_DIR = os.path.join(BASE_DIR, "_tmp_face_extract")
SIGNALING_SERVER_URL = os.environ.get("SIGNALING_SERVER_URL", "ws://192.168.0.64:9000")
SEG_MODEL_PATH = os.path.join(BASE_DIR, "models", "unrealsim.pt")
SEG_DETECTION_CONFIDENCE = 0.3
DET_MODEL_PATH = os.environ.get("DET_MODEL_PATH", "yolo11n.pt")
DET_CONFIDENCE = float(os.environ.get("DET_CONFIDENCE", "0.25"))
SEG_SCAN_HEIGHTS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
MOBILE_VIEW_DROIDCAM_URL = os.environ.get("MOBILE_VIEW_DROIDCAM_URL", "http://192.168.0.55:4747/video")
MOBILE_VIEW_WIDTH = int(os.environ.get("MOBILE_VIEW_WIDTH", "640"))
MOBILE_VIEW_HEIGHT = int(os.environ.get("MOBILE_VIEW_HEIGHT", "480"))
VOICE_VOLUME = int(os.environ.get("VOICE_VOLUME", "100"))
PIPER_MODEL_PATH = os.environ.get("PIPER_MODEL_PATH", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "voices", "nl_BE-nathalie-medium.onnx"))
PIPER_RATE = int(os.environ.get("PIPER_RATE", "22050"))
PIPER_LENGTH_SCALE = float(os.environ.get("PIPER_LENGTH_SCALE", "1.0"))
SETTINGS_PATH = os.path.join(BASE_DIR, "settings.json")
OLLAMA_CHAT_URL = os.environ.get("OLLAMA_CHAT_URL", "http://10.2.160.41:11434/api/chat")
OLLAMA_CHAT_MODEL = os.environ.get("OLLAMA_CHAT_MODEL", "gemma3")
_seg_model = None
_seg_models = {}


def _default_app_settings():
    return {
        "segmentation_server": SIGNALING_SERVER_URL,
        "droidcam_url": MOBILE_VIEW_DROIDCAM_URL,
        "segmentation_model": SEG_MODEL_PATH,
        "voice_volume": VOICE_VOLUME,
        "ollama_chat_url": OLLAMA_CHAT_URL,
        "ollama_chat_model": OLLAMA_CHAT_MODEL,
    }


def _coerce_voice_volume(value, default_value=100):
    try:
        v = int(value)
    except Exception:
        v = int(default_value)
    return max(0, min(100, v))


def load_app_settings():
    defaults = _default_app_settings()
    if not os.path.isfile(SETTINGS_PATH):
        return defaults
    try:
        with open(SETTINGS_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            return defaults
        merged = dict(defaults)
        merged.update({
            "segmentation_server": str(data.get("segmentation_server", defaults["segmentation_server"])).strip() or defaults["segmentation_server"],
            "droidcam_url": str(data.get("droidcam_url", defaults["droidcam_url"])).strip() or defaults["droidcam_url"],
            "segmentation_model": str(data.get("segmentation_model", defaults["segmentation_model"])).strip() or defaults["segmentation_model"],
            "voice_volume": _coerce_voice_volume(data.get("voice_volume", defaults["voice_volume"]), defaults["voice_volume"]),
            "ollama_chat_url": str(data.get("ollama_chat_url", defaults["ollama_chat_url"])).strip() or defaults["ollama_chat_url"],
            "ollama_chat_model": str(data.get("ollama_chat_model", defaults["ollama_chat_model"])).strip() or defaults["ollama_chat_model"],
        })
        return merged
    except Exception:
        return defaults


def save_app_settings(settings: dict):
    payload = {
        "segmentation_server": str(settings.get("segmentation_server", "")).strip(),
        "droidcam_url": str(settings.get("droidcam_url", "")).strip(),
        "segmentation_model": str(settings.get("segmentation_model", "")).strip(),
        "voice_volume": _coerce_voice_volume(settings.get("voice_volume", 100), 100),
        "ollama_chat_url": str(settings.get("ollama_chat_url", "")).strip(),
        "ollama_chat_model": str(settings.get("ollama_chat_model", "")).strip(),
    }
    with open(SETTINGS_PATH, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def apply_runtime_settings(settings: dict):
    global SIGNALING_SERVER_URL, MOBILE_VIEW_DROIDCAM_URL, SEG_MODEL_PATH, VOICE_VOLUME, OLLAMA_CHAT_URL, OLLAMA_CHAT_MODEL, _seg_model, _seg_models
    old_model = SEG_MODEL_PATH
    SIGNALING_SERVER_URL = str(settings.get("segmentation_server", SIGNALING_SERVER_URL)).strip() or SIGNALING_SERVER_URL
    MOBILE_VIEW_DROIDCAM_URL = str(settings.get("droidcam_url", MOBILE_VIEW_DROIDCAM_URL)).strip() or MOBILE_VIEW_DROIDCAM_URL
    SEG_MODEL_PATH = str(settings.get("segmentation_model", SEG_MODEL_PATH)).strip() or SEG_MODEL_PATH
    VOICE_VOLUME = _coerce_voice_volume(settings.get("voice_volume", VOICE_VOLUME), VOICE_VOLUME)
    OLLAMA_CHAT_URL = str(settings.get("ollama_chat_url", OLLAMA_CHAT_URL)).strip() or OLLAMA_CHAT_URL
    OLLAMA_CHAT_MODEL = str(settings.get("ollama_chat_model", OLLAMA_CHAT_MODEL)).strip() or OLLAMA_CHAT_MODEL
    if SEG_MODEL_PATH != old_model:
        _seg_model = None
        _seg_models.pop(old_model, None)


def current_app_settings():
    return {
        "segmentation_server": SIGNALING_SERVER_URL,
        "droidcam_url": MOBILE_VIEW_DROIDCAM_URL,
        "segmentation_model": SEG_MODEL_PATH,
        "voice_volume": VOICE_VOLUME,
        "ollama_chat_url": OLLAMA_CHAT_URL,
        "ollama_chat_model": OLLAMA_CHAT_MODEL,
    }


apply_runtime_settings(load_app_settings())


_menu_tts_lock = threading.Lock()
_menu_tts_last_at = 0.0


def _piper_say_text(text: str):
    msg = (text or "").strip()
    if not msg:
        return
    if not os.path.isfile(os.path.expanduser(PIPER_MODEL_PATH)):
        return

    try:
        p1 = subprocess.Popen(
            ["piper", "--model", os.path.expanduser(PIPER_MODEL_PATH), "--output_raw", "--length_scale", str(PIPER_LENGTH_SCALE)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )
    except Exception:
        return

    try:
        raw_audio, _ = p1.communicate(input=(msg + "\n").encode("utf-8"), timeout=40)
    except Exception:
        try:
            p1.kill()
        except Exception:
            pass
        return

    if not raw_audio:
        return

    vol = _coerce_voice_volume(VOICE_VOLUME, 100)
    if vol < 100:
        try:
            pcm = np.frombuffer(raw_audio, dtype=np.int16).astype(np.float32)
            pcm *= (vol / 100.0)
            np.clip(pcm, -32768, 32767, out=pcm)
            raw_audio = pcm.astype(np.int16).tobytes()
        except Exception:
            pass

    try:
        p2 = subprocess.Popen(
            ["aplay", "-r", str(PIPER_RATE), "-f", "S16_LE", "-t", "raw", "-"],
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        p2.communicate(input=raw_audio, timeout=40)
    except Exception:
        pass


def _speak_menu_async(text: str):
    t = threading.Thread(target=_piper_say_text, args=(text,), daemon=True)
    t.start()


def _normalize_chat_messages(raw_messages):
    normalized = []
    for item in raw_messages or []:
        if not isinstance(item, dict):
            continue
        role = str(item.get("role", "")).strip().lower()
        content = str(item.get("content", "")).strip()
        if role not in ("user", "assistant", "system") or not content:
            continue
        normalized.append({
            "role": role,
            "content": content[:4000],
        })
    return normalized[-40:]


def _ollama_chat(messages):
    payload = {
        "model": OLLAMA_CHAT_MODEL,
        "messages": messages,
        "stream": False,
    }
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        OLLAMA_CHAT_URL,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        body = resp.read()
    parsed = json.loads(body.decode("utf-8"))
    message = parsed.get("message") or {}
    return str(message.get("content", "")).strip()


UNKNOWN_TS_RE_8 = re.compile(r"^(?P<d>\d{8})_(?P<t>\d{6})$")
UNKNOWN_TS_RE_6 = re.compile(r"^(?P<d>\d{6})_(?P<t>\d{6})$")
UNKNOWN_ROOT_IMAGE_TS_RE = re.compile(r"^(?P<d>\d{8})_(?P<t>\d{6})_[^/\\]+\.(jpg|jpeg|png|bmp|webp)$", re.IGNORECASE)


def parse_unknown_session_name(folder_name: str):
    for regex, fmt, out_fmt in (
        (UNKNOWN_TS_RE_8, "%Y%m%d_%H%M%S", "%d/%m/%Y %H:%M:%S"),
        (UNKNOWN_TS_RE_6, "%y%m%d_%H%M%S", "%d/%m/%Y %H:%M:%S"),
    ):
        m = regex.match(folder_name)
        if not m:
            continue
        raw = f"{m.group('d')}_{m.group('t')}"
        try:
            dt = datetime.strptime(raw, fmt)
            return dt, dt.strftime(out_fmt)
        except Exception:
            return None, folder_name
    return None, folder_name


def iter_image_files(folder: str):
    exts = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
    files = []
    for fn in os.listdir(folder):
        p = os.path.join(folder, fn)
        if os.path.isfile(p) and os.path.splitext(fn)[1].lower() in exts:
            files.append(fn)
    files.sort()
    return files


def is_allowed_image_filename(filename: str) -> bool:
    ext = os.path.splitext((filename or "").lower())[1]
    return ext in (".jpg", ".jpeg", ".png", ".bmp", ".webp")


SAFE_NAME_RE = re.compile(r"[^A-Za-z0-9_-]+")


def sanitize_person_name(name: str) -> str:
    cleaned = SAFE_NAME_RE.sub("_", (name or "").strip())
    cleaned = cleaned.strip("_")
    return cleaned


def unique_path(path: str) -> str:
    if not os.path.exists(path):
        return path
    base, ext = os.path.splitext(path)
    n = 1
    while True:
        cand = f"{base}_{n}{ext}"
        if not os.path.exists(cand):
            return cand
        n += 1


def load_npz_features(npz_path: str):
    data = np.load(npz_path, allow_pickle=True)
    keys = list(data.keys())
    key = "features" if "features" in data else (keys[0] if keys else None)
    if key is None:
        return None
    arr = np.asarray(data[key], dtype=np.float32)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if arr.ndim != 2 or arr.shape[0] == 0:
        return None
    return arr


def append_features_for_person(person: str, new_features: list):
    if not new_features:
        return 0

    os.makedirs(KNOWN_DIR, exist_ok=True)
    npz_path = os.path.join(KNOWN_DIR, f"{person}.npz")
    new_stack = np.stack(new_features, axis=0).astype(np.float32)

    if os.path.isfile(npz_path):
        old = load_npz_features(npz_path)
        if old is not None:
            merged = np.concatenate([old, new_stack], axis=0)
        else:
            merged = new_stack
    else:
        merged = new_stack

    np.savez_compressed(npz_path, features=merged)
    return int(new_stack.shape[0])


def resolve_face_extract_session_dir(session: str):
    session = (session or "").strip()
    if not session:
        return None
    d = os.path.abspath(os.path.join(FACE_EXTRACT_TMP_DIR, session))
    root = os.path.abspath(FACE_EXTRACT_TMP_DIR)
    if os.path.commonpath([d, root]) != root:
        return None
    return d


def load_face_extract_manifest(session_dir: str):
    path = os.path.join(session_dir, "manifest.json")
    if not os.path.isfile(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_face_extract_manifest(session_dir: str, manifest: dict):
    path = os.path.join(session_dir, "manifest.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)



def load_known_embeddings():
    known_dir = KNOWN_DIR
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


def list_unknown_sessions():
    os.makedirs(UNKNOWN_DIR, exist_ok=True)
    sessions = []
    for folder in os.listdir(UNKNOWN_DIR):
        full = os.path.join(UNKNOWN_DIR, folder)
        if not os.path.isdir(full):
            continue
        files = iter_image_files(full)
        print(f"Session '{folder}': found {len(files)} image files.")
        dt, dt_str = parse_unknown_session_name(folder)
        sessions.append({
            "folder": folder,
            "count": len(files),
            "preview": files[0] if files else None,
            "dt": dt,
            "dt_str": dt_str,
        })
    sessions.sort(
        key=lambda x: (x["dt"] is not None, x["dt"] or datetime.min, x["folder"]),
        reverse=True,
    )
    return sessions


def list_unknown_root_images():
    os.makedirs(UNKNOWN_DIR, exist_ok=True)
    files = []
    for fn in os.listdir(UNKNOWN_DIR):
        p = os.path.join(UNKNOWN_DIR, fn)
        if os.path.isfile(p) and is_allowed_image_filename(fn):
            files.append({
                "filename": fn,
                "mtime": os.path.getmtime(p),
            })
    files.sort(key=lambda x: x["mtime"], reverse=True)
    return [x["filename"] for x in files]


def _parse_unknown_root_image_datetime(filename: str):
    m = UNKNOWN_ROOT_IMAGE_TS_RE.match((filename or "").strip())
    if not m:
        return None
    raw = f"{m.group('d')}_{m.group('t')}"
    try:
        return datetime.strptime(raw, "%Y%m%d_%H%M%S")
    except Exception:
        return None


def cleanup_old_unknown_root_images(max_age_days: int = 7):
    os.makedirs(UNKNOWN_DIR, exist_ok=True)
    cutoff = datetime.now() - timedelta(days=max_age_days)
    removed = 0

    for fn in os.listdir(UNKNOWN_DIR):
        p = os.path.join(UNKNOWN_DIR, fn)
        if not os.path.isfile(p) or not is_allowed_image_filename(fn):
            continue

        shot_dt = _parse_unknown_root_image_datetime(fn)
        if shot_dt is None or shot_dt >= cutoff:
            continue

        try:
            os.remove(p)
            removed += 1
        except Exception:
            pass

    return removed


def resolve_unknown_session_dir(session: str):
    session = (session or "").strip()
    if not session:
        return None
    session_dir = os.path.abspath(os.path.join(UNKNOWN_DIR, session))
    unknown_root = os.path.abspath(UNKNOWN_DIR)
    if os.path.commonpath([session_dir, unknown_root]) != unknown_root:
        return None
    return session_dir


def move_unknown_session_images_to_known(session_dir: str, person: str, session: str):
    """
    Verplaatst snapshots van unknown/<session>/ naar known/<person>/.
    Bestandsnamen krijgen een session-prefix om collisions te vermijden.
    """
    person_dir = os.path.join(KNOWN_DIR, person)
    os.makedirs(person_dir, exist_ok=True)

    files = iter_image_files(session_dir)
    moved = 0
    for idx, fn in enumerate(files, start=1):
        src = os.path.join(session_dir, fn)
        ext = os.path.splitext(fn)[1].lower() or ".jpg"
        base = f"{session}_{idx:04d}"
        dst = os.path.join(person_dir, f"{base}{ext}")

        # Unieke naam forceren als bestand al bestaat.
        n = 1
        while os.path.exists(dst):
            dst = os.path.join(person_dir, f"{base}_{n}{ext}")
            n += 1

        shutil.move(src, dst)
        moved += 1

    return person_dir, moved

# ---- Herkenning procesbeheer ----
RECOGNITION_SCRIPT = os.path.join(BASE_DIR, "nl_launchv2.py")  # QR-registratie zit in de v2 launcher.
_recognition_proc = None
_recognition_source = "local"
_log_lines = deque(maxlen=400)
_log_lock = threading.Lock()

def _append_log(line: str):
    with _log_lock:
        _log_lines.append(line.rstrip("\n"))

def _reader_thread(proc: subprocess.Popen):
    try:
        for line in proc.stdout:
            if not line:
                break
            _append_log(line)
    except Exception as e:
        _append_log(f"[LOG-ERROR] {e}")
    finally:
        _append_log("[INFO] Herkenning gestopt.")

def recognition_running() -> bool:
    global _recognition_proc
    return _recognition_proc is not None and _recognition_proc.poll() is None

def get_recognition_source() -> str:
    return _recognition_source


def start_recognition(source: str = "local"):
    global _recognition_proc, _recognition_source
    if recognition_running():
        return

    source = (source or "").strip().lower()
    if source not in ("local", "droidcam"):
        source = "local"
    _recognition_source = source

    os.makedirs(SNAPSHOT_DIR, exist_ok=True)
    os.makedirs(KNOWN_DIR, exist_ok=True)

    # Belangrijk:
    # - nl_launchv2.py is headless en verwerkt ook QR-registraties.
    #   Deze web-integratie start hem vooral voor "bekenden herkennen" + snapshots opslaan.
    cmd = [
        "python3", RECOGNITION_SCRIPT,
        "--known", KNOWN_DIR,
        "--voice_volume", str(VOICE_VOLUME),
        # optioneel: zet speak uit als je geen audio wil vanuit deze service:
        # "--no_tts",
        # optioneel: camera settings
        # "--cam", "0",
        # "--width", "640", "--height", "480", "--fps", "15",
    ]

    _append_log("[INFO] Start herkenning…")
    if source == "droidcam":
        cmd.extend(["--cam_url", _normalize_droidcam_url(MOBILE_VIEW_DROIDCAM_URL)])
    _append_log(f"[INFO] Bron: {'DroidCam' if source == 'droidcam' else 'Lokale camera'}")
    _append_log("[INFO] CMD: " + " ".join(cmd))

    _recognition_proc = subprocess.Popen(
        cmd,
        cwd=BASE_DIR,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1
    )

    t = threading.Thread(target=_reader_thread, args=(_recognition_proc,), daemon=True)
    t.start()

def stop_recognition():
    global _recognition_proc
    if not recognition_running():
        _recognition_proc = None
        return
    _append_log("[INFO] Stop signaal gestuurd…")
    try:
        _recognition_proc.terminate()
    except Exception:
        pass

# ---- Helpers: bekende personen uit known/*.npz ----
def list_known_people():
    os.makedirs(KNOWN_DIR, exist_ok=True)
    people = []
    for fn in os.listdir(KNOWN_DIR):
        if fn.lower().endswith(".npz"):
            people.append(os.path.splitext(fn)[0])
    people.sort(key=lambda s: s.lower())
    return people


def resolve_known_person_dir(person: str):
    person = (person or "").strip()
    if not person:
        return None, None
    person_dir = os.path.abspath(os.path.join(KNOWN_DIR, person))
    known_root = os.path.abspath(KNOWN_DIR)
    if os.path.commonpath([person_dir, known_root]) != known_root:
        return None, None
    npz_path = os.path.join(KNOWN_DIR, f"{person}.npz")
    return person_dir, npz_path


def list_known_people_with_photos():
    items = []
    for person in list_known_people():
        person_dir = os.path.join(KNOWN_DIR, person)
        files = iter_image_files(person_dir) if os.path.isdir(person_dir) else []
        items.append({
            "name": person,
            "count": len(files),
            "preview": files[0] if files else None,
        })
    return items


def list_known_photo_dirs():
    os.makedirs(KNOWN_DIR, exist_ok=True)
    dirs = []
    for name in os.listdir(KNOWN_DIR):
        full = os.path.join(KNOWN_DIR, name)
        if os.path.isdir(full):
            dirs.append(name)
    dirs.sort(key=lambda s: s.lower())
    return dirs


_camera = None
_camera_lock = threading.Lock()
_mobile_cam = None
_mobile_cam_lock = threading.Lock()
_stream_enabled = False
_stream_lock = threading.Lock()
_stream_source = "local"
_droidcam_last_frame = None
_droidcam_last_frame_at = 0.0
_droidcam_frame_lock = threading.Lock()
_face_lock = threading.Lock()
_face_detector = None
_face_recognizer = None
_known_cache = {}
_known_cache_at = 0.0
_seg_thread = None
_seg_stop_event = threading.Event()
_seg_lock = threading.Lock()
_seg_running = False
_seg_connected = False
_seg_source = "websocket"
_seg_last_error = ""
_seg_last_heading = 90.0
_seg_last_frame_id = None
_seg_last_jpeg = None
_seg_last_update = 0.0
_det_models = {}
_det_model_lock = threading.Lock()
_seg_model_lock = threading.Lock()

def get_camera(cam_index=0):
    global _camera
    with _camera_lock:
        if _camera is None or not _camera.isOpened():
            _camera = cv2.VideoCapture(cam_index)
            _camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            _camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        return _camera

def release_camera():
    global _camera
    with _camera_lock:
        if _camera is not None:
            try:
                _camera.release()
            except Exception:
                pass
        _camera = None


def _normalize_droidcam_url(url: str):
    u = (url or "").strip()
    if not u:
        return ""
    if u.endswith("/video") or u.endswith("/mjpegfeed"):
        return u
    if u.endswith("/"):
        return u + "video"
    return u + "/video"


def get_mobile_cam():
    global _mobile_cam
    stream_url = _normalize_droidcam_url(MOBILE_VIEW_DROIDCAM_URL)
    with _mobile_cam_lock:
        if _mobile_cam is None or not _mobile_cam.isOpened():
            _mobile_cam = cv2.VideoCapture(stream_url)
            _mobile_cam.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            _mobile_cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            _mobile_cam.set(cv2.CAP_PROP_FPS, 15)
        return _mobile_cam


def release_mobile_cam():
    global _mobile_cam
    with _mobile_cam_lock:
        if _mobile_cam is not None:
            try:
                _mobile_cam.release()
            except Exception:
                pass
        _mobile_cam = None

def is_stream_enabled():
    with _stream_lock:
        return _stream_enabled

def set_stream_enabled(val: bool):
    global _stream_enabled
    with _stream_lock:
        _stream_enabled = val


def get_stream_source():
    with _stream_lock:
        return _stream_source


def set_stream_source(source: str):
    global _stream_source
    s = (source or "").strip().lower()
    if s not in ("local", "droidcam"):
        s = "local"
    with _stream_lock:
        _stream_source = s


def update_droidcam_last_frame(frame: np.ndarray):
    global _droidcam_last_frame, _droidcam_last_frame_at
    if frame is None:
        return
    with _droidcam_frame_lock:
        _droidcam_last_frame = frame.copy()
        _droidcam_last_frame_at = time.time()


def get_droidcam_last_frame(max_age_sec: float = 2.0):
    with _droidcam_frame_lock:
        if _droidcam_last_frame is None:
            return None
        if (time.time() - _droidcam_last_frame_at) > max_age_sec:
            return None
        return _droidcam_last_frame.copy()


def decode_signal_message_to_frame(msg):
    def _decode_data_url(s: str):
        if not s.startswith("data:image"):
            return None
        parts = s.split(",", 1)
        if len(parts) != 2:
            return None
        try:
            return base64.b64decode(parts[1])
        except Exception:
            return None

    def _decode_base64_str(s: str):
        s = s.strip()
        if not s:
            return None
        try:
            return base64.b64decode(s, validate=False)
        except Exception:
            return None

    def _bytes_to_frame(jpeg_bytes: bytes):
        np_arr = np.frombuffer(jpeg_bytes, dtype=np.uint8)
        return cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    def _extract_b64_from_payload(payload):
        if isinstance(payload, dict):
            # meest voorkomende keys
            for key in ("data", "image", "frame", "jpeg", "img", "blob"):
                val = payload.get(key)
                if isinstance(val, str) and val.strip():
                    return val
            # nested payloads
            for key in ("payload", "frame", "message"):
                val = payload.get(key)
                if isinstance(val, dict):
                    nested = _extract_b64_from_payload(val)
                    if nested:
                        return nested
        return None

    try:
        if isinstance(msg, (bytes, bytearray)):
            raw = bytes(msg)

            # Soms komt JSON als bytes binnen.
            if raw[:1] in (b"{", b"["):
                try:
                    txt = raw.decode("utf-8", errors="ignore")
                    payload = json.loads(txt)
                    b64 = _extract_b64_from_payload(payload)
                    if b64:
                        jpeg_bytes = _decode_data_url(b64) or _decode_base64_str(b64)
                        if jpeg_bytes:
                            frame = _bytes_to_frame(jpeg_bytes)
                            if frame is not None:
                                return frame
                except Exception:
                    pass

            # Eerst proberen als raw jpeg bytes.
            frame = _bytes_to_frame(raw)
            if frame is not None:
                return frame

            # Fallback: bytes bevatten base64-tekst.
            try:
                txt = raw.decode("utf-8", errors="ignore")
                jpeg_bytes = _decode_data_url(txt) or _decode_base64_str(txt)
                if jpeg_bytes:
                    return _bytes_to_frame(jpeg_bytes)
            except Exception:
                return None
            return None
        elif isinstance(msg, str):
            text = msg.strip()

            # Data URL rechtstreeks.
            jpeg_bytes = _decode_data_url(text)
            if jpeg_bytes:
                frame = _bytes_to_frame(jpeg_bytes)
                if frame is not None:
                    return frame

            # JSON payload.
            try:
                payload = json.loads(text)
                b64 = _extract_b64_from_payload(payload)
                if b64:
                    jpeg_bytes = _decode_data_url(b64) or _decode_base64_str(b64)
                    if jpeg_bytes:
                        frame = _bytes_to_frame(jpeg_bytes)
                        if frame is not None:
                            return frame
            except json.JSONDecodeError:
                pass

            # Plain base64 string.
            jpeg_bytes = _decode_base64_str(text)
            if jpeg_bytes:
                frame = _bytes_to_frame(jpeg_bytes)
                if frame is not None:
                    return frame
            return None
        else:
            return None
    except Exception:
        return None


def list_segmentation_model_options():
    options = []
    seen = set()

    default_model = (SEG_MODEL_PATH or "").strip()
    if default_model:
        default_value = default_model if os.path.isabs(default_model) else os.path.basename(default_model)
        options.append({
            "value": default_value,
            "label": os.path.basename(default_model),
        })
        seen.add(default_value)

    if os.path.isdir(MODELS_DIR):
        for fn in sorted(os.listdir(MODELS_DIR), key=str.lower):
            if not fn.lower().endswith(".pt"):
                continue
            if fn in seen:
                continue
            options.append({
                "value": fn,
                "label": fn,
            })
            seen.add(fn)

    return options


def _coerce_segmentation_model(value, default_value=SEG_MODEL_PATH):
    options = {item["value"] for item in list_segmentation_model_options()}
    selected = (value or "").strip()
    fallback = (default_value or "").strip()
    if selected in options:
        return selected
    if fallback in options:
        return fallback
    fallback_name = os.path.basename(fallback)
    if fallback_name in options:
        return fallback_name
    return fallback_name or "unrealsim.pt"


def _resolve_segmentation_model_source(model_source):
    selected = _coerce_segmentation_model(model_source, SEG_MODEL_PATH)
    if os.path.isabs(selected):
        return selected

    models_candidate = os.path.join(MODELS_DIR, selected)
    if os.path.isfile(models_candidate):
        return models_candidate

    base_candidate = os.path.join(BASE_DIR, selected)
    if os.path.isfile(base_candidate):
        return base_candidate

    return selected


def get_segmentation_model(model_source=None):
    global _seg_model
    try:
        from ultralytics import YOLO
    except Exception as e:
        raise RuntimeError(f"Ultralytics import faalde: {e}")

    model_source = _resolve_segmentation_model_source(model_source)
    if os.path.isabs(model_source) and not os.path.isfile(model_source):
        raise RuntimeError(f"Segmentation model niet gevonden: {model_source}")

    with _seg_model_lock:
        if model_source == SEG_MODEL_PATH and _seg_model is not None:
            return _seg_model

        model = _seg_models.get(model_source)
        if model is None:
            try:
                model = YOLO(model_source, verbose=False)
            except Exception as e:
                raise RuntimeError(f"Segmentation model laden mislukt: {e}") from e
            _seg_models[model_source] = model

        if model_source == SEG_MODEL_PATH:
            _seg_model = model
        return model


def _coerce_det_confidence(value, default_value=DET_CONFIDENCE):
    try:
        v = float(value)
    except Exception:
        v = float(default_value)
    return max(0.01, min(0.95, v))


def list_detection_model_options():
    options = []
    seen = set()

    default_model = (DET_MODEL_PATH or "").strip() or "yolo11n.pt"
    if default_model and default_model not in seen:
        options.append({
            "value": default_model,
            "label": os.path.basename(default_model),
        })
        seen.add(default_model)

    if os.path.isdir(MODELS_DIR):
        for fn in sorted(os.listdir(MODELS_DIR), key=str.lower):
            if not fn.lower().endswith(".pt"):
                continue
            if fn in seen:
                continue
            options.append({
                "value": fn,
                "label": fn,
            })
            seen.add(fn)

    return options


def _coerce_detection_model(value, default_value=DET_MODEL_PATH):
    options = {item["value"] for item in list_detection_model_options()}
    selected = (value or "").strip()
    fallback = (default_value or "").strip() or "yolo11n.pt"
    if selected in options:
        return selected
    return fallback


def _resolve_detection_model_source(model_source):
    selected = _coerce_detection_model(model_source, DET_MODEL_PATH)
    if os.path.isabs(selected):
        return selected

    models_candidate = os.path.join(MODELS_DIR, selected)
    if os.path.isfile(models_candidate):
        return models_candidate

    base_candidate = os.path.join(BASE_DIR, selected)
    if os.path.isfile(base_candidate):
        return base_candidate

    return selected


def get_detection_model(model_source=None):
    try:
        from ultralytics import YOLO
    except Exception as e:
        raise RuntimeError(f"Ultralytics import faalde: {e}")

    model_source = _resolve_detection_model_source(model_source)
    if os.path.isabs(model_source) and not os.path.isfile(model_source):
        raise RuntimeError(f"YOLO model niet gevonden: {model_source}")

    with _det_model_lock:
        model = _det_models.get(model_source)
        print(f"Requesting detection model '{model_source}'. Cache hit: {'Yes' if model is not None else 'No'}")

        if model is None:
            try:
                model = YOLO(model_source, verbose=False)
            except Exception as e:
                raise RuntimeError(f"YOLO model laden mislukt: {e}") from e
            _det_models[model_source] = model
    return model


def detect_objects_uploaded_image(img: np.ndarray, confidence: float = DET_CONFIDENCE, model_source=None):
    model = get_detection_model(model_source=model_source)
    conf = _coerce_det_confidence(confidence, DET_CONFIDENCE)
    t0 = time.perf_counter()
    results = model(img, conf=conf, verbose=False)
    inference_ms = (time.perf_counter() - t0) * 1000.0

    annotated = img.copy()
    detections = []
    idx = 0

    for r in results:
        if r.boxes is None or len(r.boxes) == 0:
            continue
        try:
            annotated = r.plot()
        except Exception:
            pass

        names = r.names if isinstance(r.names, dict) else {}
        for b in r.boxes:
            idx += 1
            cls_idx = int(float(b.cls[0].item())) if b.cls is not None else -1
            conf_score = float(b.conf[0].item()) if b.conf is not None else 0.0
            x1, y1, x2, y2 = [int(v) for v in b.xyxy[0].tolist()]
            label = names.get(cls_idx, str(cls_idx))
            detections.append({
                "idx": idx,
                "label": label,
                "score": f"{conf_score:.4f}",
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
            })

    return annotated, detections, conf, inference_ms


def process_segment_frame(
    frame,
    model_source=None,
    confidence=SEG_DETECTION_CONFIDENCE,
    draw_polygons=False,
    allow_detection_boxes=False,
):
    model = get_segmentation_model(model_source=model_source)
    h, w = frame.shape[:2]
    overlay = frame.copy()
    infer_t0 = time.perf_counter()
    results = model(frame, conf=confidence, verbose=False)
    infer_ms = (time.perf_counter() - infer_t0) * 1000.0

    midpoints = []
    has_masks = False
    for r in results:
        if r.masks is None or len(r.masks.data) == 0:
            continue
        has_masks = True

        mask_polygons = list(r.masks.xy) if getattr(r.masks, "xy", None) is not None else []

        for mask_idx, mask_tensor in enumerate(r.masks.data):
            mask = mask_tensor.cpu().numpy()
            mask = (mask * 255).astype(np.uint8)
            mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)

            green = np.full_like(frame, (0, 255, 0))
            blended = cv2.addWeighted(frame, 0.3, green, 0.7, 0)
            overlay[mask > 0] = blended[mask > 0]

            if draw_polygons and mask_idx < len(mask_polygons):
                polygon = np.array(mask_polygons[mask_idx], dtype=np.int32)
                if polygon.ndim == 2 and len(polygon) >= 3:
                    cv2.polylines(overlay, [polygon.reshape((-1, 1, 2))], True, (0, 255, 255), 3)

            for rr in SEG_SCAN_HEIGHTS:
                y = int(h * rr)
                if y >= h:
                    continue
                scan_row = mask[y, :]
                idx = np.where(scan_row > 0)[0]
                if len(idx) > 0:
                    mx = int(np.mean(idx))
                    midpoints.append((mx, y))
                    cv2.circle(overlay, (mx, y), 5, (255, 0, 0), -1)

    for rr in SEG_SCAN_HEIGHTS:
        y = int(h * rr)
        if y < h:
            cv2.line(overlay, (0, y), (w, y), (150, 150, 150), 1)

    if allow_detection_boxes and not has_masks:
        for r in results:
            if r.boxes is None or len(r.boxes) == 0:
                continue
            names = r.names if isinstance(r.names, dict) else {}
            for b in r.boxes:
                cls_idx = int(float(b.cls[0].item())) if b.cls is not None else -1
                conf_score = float(b.conf[0].item()) if b.conf is not None else 0.0
                x1, y1, x2, y2 = [int(v) for v in b.xyxy[0].tolist()]
                label = names.get(cls_idx, str(cls_idx))
                cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 200, 255), 2)
                cv2.putText(
                    overlay,
                    f"{label} {conf_score:.2f}",
                    (max(0, x1), max(24, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 200, 255),
                    2,
                )

    direction_angle = 90.0
    start_point = (w // 2, h)
    if midpoints:
        avg_x = int(np.mean([p[0] for p in midpoints]))
        target_point = (avg_x, min([p[1] for p in midpoints]))
        dx = avg_x - start_point[0]
        dy = start_point[1] - target_point[1]
        direction_angle = float(np.degrees(np.arctan2(dy, dx)))
        cv2.arrowedLine(overlay, start_point, target_point, (0, 0, 255), 5, tipLength=0.2)
    elif has_masks:
        cv2.arrowedLine(overlay, start_point, (w // 2, int(h * 0.6)), (0, 0, 255), 5, tipLength=0.2)

    cv2.putText(
        overlay,
        f"Heading: {direction_angle:.2f}",
        (20, 36),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (255, 255, 255),
        2,
    )
    cv2.putText(
        overlay,
        f"Inference: {infer_ms:.1f} ms",
        (20, 72),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
    )
    if allow_detection_boxes and not has_masks:
        cv2.putText(
            overlay,
            "Mode: detection",
            (20, 108),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
        )
    return overlay, direction_angle


def set_seg_state(**kwargs):
    global _seg_running, _seg_connected, _seg_source, _seg_last_error, _seg_last_heading, _seg_last_frame_id, _seg_last_jpeg, _seg_last_update
    with _seg_lock:
        if "running" in kwargs:
            _seg_running = kwargs["running"]
        if "connected" in kwargs:
            _seg_connected = kwargs["connected"]
        if "source" in kwargs:
            s = (kwargs["source"] or "").strip().lower()
            _seg_source = s if s in ("websocket", "local", "droidcam") else "websocket"
        if "last_error" in kwargs:
            _seg_last_error = kwargs["last_error"]
        if "last_heading" in kwargs:
            _seg_last_heading = kwargs["last_heading"]
        if "last_frame_id" in kwargs:
            _seg_last_frame_id = kwargs["last_frame_id"]
        if "last_jpeg" in kwargs:
            _seg_last_jpeg = kwargs["last_jpeg"]
        if "last_update" in kwargs:
            _seg_last_update = kwargs["last_update"]


def get_seg_state():
    with _seg_lock:
        return {
            "running": _seg_running,
            "connected": _seg_connected,
            "source": _seg_source,
            "last_error": _seg_last_error,
            "heading": _seg_last_heading,
            "frame_id": _seg_last_frame_id,
            "has_frame": _seg_last_jpeg is not None,
            "last_update": _seg_last_update,
        }


async def segmentation_ws_loop(stop_event: threading.Event):
    set_seg_state(running=True, connected=False, source="websocket", last_error="")
    pending_frame_id = None

    while not stop_event.is_set():
        #print ("Starting segmentation WebSocket loop")
        try:
            async with websockets.connect(SIGNALING_SERVER_URL, max_size=None) as ws:
                set_seg_state(connected=True, last_error="")
                while not stop_event.is_set():
                    try:
                        msg = await asyncio.wait_for(ws.recv(), timeout=1.0)
                    except asyncio.TimeoutError:
                        continue

                    frame_id = None
                    if isinstance(msg, str):
                        try:
                            payload = json.loads(msg)
                            if payload.get("type") == "frame_meta":
                                pending_frame_id = payload.get("frame_id")
                                continue
                        except json.JSONDecodeError:
                            pass

                    frame = decode_signal_message_to_frame(msg)
                    #print (f"Received frame {msg}")
                    if frame is None:
                        continue

                    if isinstance(msg, (bytes, bytearray)):
                        frame_id = pending_frame_id
                        pending_frame_id = None
                    elif isinstance(msg, str):
                        try:
                            payload = json.loads(msg)
                            frame_id = payload.get("frame_id", pending_frame_id)
                        except Exception:
                            frame_id = pending_frame_id
                        pending_frame_id = None

                    overlay, heading = process_segment_frame(frame)
                    ok, enc = cv2.imencode(".jpg", overlay)
                    if not ok:
                        continue

                    set_seg_state(
                        last_jpeg=enc.tobytes(),
                        last_heading=float(heading),
                        last_frame_id=frame_id,
                        last_update=time.time(),
                    )

                    try:
                        await ws.send(json.dumps({"heading": round(float(heading), 2), "frame_id": frame_id}))
                    except Exception:
                        pass
        except Exception as e:
            set_seg_state(connected=False, last_error=str(e))
            await asyncio.sleep(1.0)

    set_seg_state(running=False, connected=False)


def segmentation_local_loop(stop_event: threading.Event):
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 15)
    set_seg_state(running=True, connected=cap.isOpened(), source="local", last_error="")
    try:
        if not cap.isOpened():
            set_seg_state(last_error="Lokale camera niet beschikbaar.")
            return
        frame_id = 0
        while not stop_event.is_set():
            ok, frame = cap.read()
            if not ok or frame is None:
                set_seg_state(connected=False, last_error="Kon geen frame lezen van lokale camera.")
                time.sleep(0.1)
                continue
            frame_id += 1
            set_seg_state(connected=True, last_error="")
            overlay, heading = process_segment_frame(frame)
            ok, enc = cv2.imencode(".jpg", overlay)
            if not ok:
                continue
            set_seg_state(
                last_jpeg=enc.tobytes(),
                last_heading=float(heading),
                last_frame_id=frame_id,
                last_update=time.time(),
            )
    finally:
        try:
            cap.release()
        except Exception:
            pass
        set_seg_state(running=False, connected=False)


def segmentation_droidcam_loop(stop_event: threading.Event):
    stream_url = _normalize_droidcam_url(MOBILE_VIEW_DROIDCAM_URL)
    set_seg_state(running=True, connected=False, source="droidcam", last_error="")
    frame_id = 0
    while not stop_event.is_set():
        try:
            for jpeg in _iter_http_mjpeg_frames(stream_url):
                if stop_event.is_set():
                    break
                arr = np.frombuffer(jpeg, dtype=np.uint8)
                frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                if frame is None:
                    continue
                frame = cv2.resize(frame, (MOBILE_VIEW_WIDTH, MOBILE_VIEW_HEIGHT), interpolation=cv2.INTER_AREA)
                frame_id += 1
                set_seg_state(connected=True, last_error="")
                overlay, heading = process_segment_frame(frame)
                ok, enc = cv2.imencode(".jpg", overlay)
                if not ok:
                    continue
                set_seg_state(
                    last_jpeg=enc.tobytes(),
                    last_heading=float(heading),
                    last_frame_id=frame_id,
                    last_update=time.time(),
                )
        except Exception as e:
            set_seg_state(connected=False, last_error=str(e))
            time.sleep(0.8)
    set_seg_state(running=False, connected=False)


def segmentation_thread_target(stop_event: threading.Event, source: str):
    src = (source or "").strip().lower()
    if src == "local":
        segmentation_local_loop(stop_event)
    elif src == "droidcam":
        segmentation_droidcam_loop(stop_event)
    else:
        asyncio.run(segmentation_ws_loop(stop_event))


def start_segmentation_stream(source: str = "websocket"):
    global _seg_thread
    st = get_seg_state()
    if st["running"]:
        return
    src = (source or "").strip().lower()
    if src not in ("websocket", "local", "droidcam"):
        src = "websocket"
    _seg_stop_event.clear()
    set_seg_state(source=src, last_error="")
    _seg_thread = threading.Thread(target=segmentation_thread_target, args=(_seg_stop_event, src), daemon=True)
    _seg_thread.start()


def stop_segmentation_stream():
    global _seg_thread
    _seg_stop_event.set()
    if _seg_thread is not None and _seg_thread.is_alive():
        _seg_thread.join(timeout=2.0)
    _seg_thread = None


def gen_segmentation_frames():
    while True:
        st = get_seg_state()
        if not st["running"]:
            return

        with _seg_lock:
            frame_bytes = _seg_last_jpeg
        if frame_bytes is None:
            time.sleep(0.05)
            continue

        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n"
        )

def gen_frames(source="local"):
    # Wacht tot stream aan staat (of stop meteen)
    while not is_stream_enabled():
        time.sleep(0.1)

    src = (source or "local").strip().lower()
    if src == "droidcam":
        stream_url = _normalize_droidcam_url(MOBILE_VIEW_DROIDCAM_URL)
        while True:
            if not is_stream_enabled():
                return
            try:
                for jpeg in _iter_http_mjpeg_frames(stream_url):
                    if not is_stream_enabled():
                        return
                    arr = np.frombuffer(jpeg, dtype=np.uint8)
                    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                    if frame is None:
                        continue
                    frame = cv2.resize(frame, (MOBILE_VIEW_WIDTH, MOBILE_VIEW_HEIGHT), interpolation=cv2.INTER_AREA)
                    update_droidcam_last_frame(frame)
                    out = _annotate_mobile_face_frame(frame)
                    ok, buffer = cv2.imencode(".jpg", out)
                    if not ok:
                        continue
                    frame_bytes = buffer.tobytes()
                    yield (b"--frame\r\n"
                           b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n")
            except Exception:
                time.sleep(0.5)
        return

    cam = get_camera(0)

    while True:
        # Als iemand op "stop" drukt: stop generator + release camera
        if not is_stream_enabled():
            release_camera()
            return  # <-- beëindigt de HTTP stream netjes

        ok, frame = cam.read()
        if not ok or frame is None:
            time.sleep(0.05)
            continue

        out = frame.copy()
        try:
            known = get_known_embeddings_cached(refresh_sec=5.0)
            with _face_lock:
                detector, recognizer = _get_face_models()
                h, w = out.shape[:2]
                detector.setInputSize((w, h))
                _, faces = detector.detect(out)
                if faces is not None and len(faces) > 0:
                    for i, face in enumerate(faces, start=1):
                        x, y, fw, fh = face[:4].astype(int)
                        label = "Onbekend"
                        color = (0, 0, 255)
                        score_txt = ""

                        try:
                            aligned = recognizer.alignCrop(out, face)
                            feat = recognizer.feature(aligned).astype(np.float32)
                            best_name, best_score, second_score = _best_match(recognizer, feat, known) if known else (None, -1.0, -1.0)
                            confident = (
                                best_name is not None
                                and (best_score >= 0.70)
                                and ((best_score - second_score) >= 0.06)
                            )
                            if confident:
                                label = best_name
                                color = (0, 255, 0)
                            if best_score >= 0:
                                score_txt = f" {best_score:.2f}"
                        except Exception:
                            pass

                        cv2.rectangle(out, (x, y), (x + fw, y + fh), color, 2)
                        cv2.putText(
                            out,
                            f"{label}{score_txt}",
                            (max(0, x), max(20, y - 8)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            color,
                            2,
                        )
        except Exception:
            pass

        ok, buffer = cv2.imencode(".jpg", out)
        if not ok:
            continue

        frame_bytes = buffer.tobytes()

        yield (b"--frame\r\n"
               b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n")


def _iter_http_mjpeg_frames(url: str):
    # Browser kan de stream tonen; deze parser leest dezelfde JPEG-bytes rechtstreeks van HTTP.
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=10) as resp:
        buf = b""
        while True:
            chunk = resp.read(4096)
            if not chunk:
                break
            buf += chunk
            while True:
                soi = buf.find(b"\xff\xd8")
                if soi < 0:
                    if len(buf) > 2_000_000:
                        buf = buf[-64_000:]
                    break
                eoi = buf.find(b"\xff\xd9", soi + 2)
                if eoi < 0:
                    if soi > 0:
                        buf = buf[soi:]
                    break
                jpeg = buf[soi:eoi + 2]
                buf = buf[eoi + 2:]
                yield jpeg


def _decode_jpeg_bytes(jpeg: bytes):
    if not jpeg:
        return None
    arr = np.frombuffer(jpeg, dtype=np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)


def _fetch_one_droidcam_frame():
    stream_url = _normalize_droidcam_url(MOBILE_VIEW_DROIDCAM_URL)
    base = stream_url
    if base.endswith("/video"):
        base = base[:-6]
    elif base.endswith("/mjpegfeed"):
        base = base[:-10]

    # DroidCam ondersteunt meestal /shot.jpg voor een single-frame snapshot.
    for snap in ("shot.jpg", "photo.jpg", "image.jpg"):
        try:
            req = urllib.request.Request(f"{base}/{snap}", headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=2) as resp:
                jpg = resp.read()
            frame = _decode_jpeg_bytes(jpg)
            if frame is not None:
                return frame
        except Exception:
            pass

    # Fallback: probeer 1 frame uit de MJPEG-stream.
    try:
        for jpg in _iter_http_mjpeg_frames(stream_url):
            frame = _decode_jpeg_bytes(jpg)
            if frame is not None:
                return frame
            break
    except Exception:
        pass
    return None


def _annotate_mobile_face_frame(frame):
    out = frame.copy()
    try:
        known = get_known_embeddings_cached(refresh_sec=5.0)
        with _face_lock:
            detector, recognizer = _get_face_models()
            h, w = out.shape[:2]
            detector.setInputSize((w, h))
            _, faces = detector.detect(out)
            if faces is not None and len(faces) > 0:
                for face in faces:
                    x, y, fw, fh = face[:4].astype(int)
                    label = "Onbekend"
                    color = (0, 0, 255)
                    score_txt = ""

                    try:
                        aligned = recognizer.alignCrop(out, face)
                        feat = recognizer.feature(aligned).astype(np.float32)
                        best_name, best_score, second_score = _best_match(recognizer, feat, known) if known else (None, -1.0, -1.0)
                        confident = (
                            best_name is not None
                            and (best_score >= 0.70)
                            and ((best_score - second_score) >= 0.06)
                        )
                        if confident:
                            label = best_name
                            color = (0, 255, 0)
                        if best_score >= 0:
                            score_txt = f" {best_score:.2f}"
                    except Exception:
                        pass

                    cv2.rectangle(out, (x, y), (x + fw, y + fh), color, 2)
                    cv2.putText(
                        out,
                        f"{label}{score_txt}",
                        (max(0, x), max(20, y - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        color,
                        2,
                    )
    except Exception:
        pass
    return out


def gen_mobile_face_frames():
    stream_url = _normalize_droidcam_url(MOBILE_VIEW_DROIDCAM_URL)
    while True:
        try:
            for jpeg in _iter_http_mjpeg_frames(stream_url):
                arr = np.frombuffer(jpeg, dtype=np.uint8)
                frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                if frame is None:
                    continue
                frame = cv2.resize(frame, (MOBILE_VIEW_WIDTH, MOBILE_VIEW_HEIGHT), interpolation=cv2.INTER_AREA)
                update_droidcam_last_frame(frame)
                out = _annotate_mobile_face_frame(frame)
                ok, buffer = cv2.imencode(".jpg", out)
                if not ok:
                    continue
                frame_bytes = buffer.tobytes()
                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n"
                )
        except Exception as e:
            print(f"[mobile-face] streamfout: {e}")
            time.sleep(1.0)


def capture_known_person_from_frame(person: str, frame: np.ndarray, filename_suffix: str = "camera"):
    person = sanitize_person_name(person)
    if not person:
        raise RuntimeError("Naam is ongeldig of leeg.")

    if frame is None:
        raise RuntimeError("Geen frame beschikbaar.")

    with _face_lock:
        detector, recognizer = _get_face_models()
        h, w = frame.shape[:2]
        detector.setInputSize((w, h))
        _, faces = detector.detect(frame)
        face = _largest_face(faces)
        if face is None:
            raise RuntimeError("Geen gezicht gedetecteerd.")

        x, y, fw, fh = face[:4].astype(int)
        pad_w = int(fw * 0.15)
        pad_h = int(fh * 0.15)
        x1 = max(0, x - pad_w)
        y1 = max(0, y - pad_h)
        x2 = min(w, x + fw + pad_w)
        y2 = min(h, y + fh + pad_h)

        crop = frame[y1:y2, x1:x2]
        if crop is None or crop.size == 0:
            raise RuntimeError("Kon geen geldige crop maken.")

        feat = None
        try:
            aligned = recognizer.alignCrop(frame, face)
            feat = recognizer.feature(aligned).astype(np.float32)
        except Exception:
            feat = None

    person_dir = os.path.join(KNOWN_DIR, person)
    os.makedirs(person_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = unique_path(os.path.join(person_dir, f"{ts}_{filename_suffix}.jpg"))
    cv2.imwrite(out_path, crop)

    added = 0
    if feat is not None and feat.ndim == 1 and feat.shape[0] > 0:
        added = append_features_for_person(person, [feat])

    return out_path, added


def capture_known_person_from_camera(person: str, cam_index: int = 0):
    cam = get_camera(cam_index)
    ok, frame = cam.read()
    if not ok or frame is None:
        raise RuntimeError("Kan geen frame lezen van camera.")
    return capture_known_person_from_frame(person, frame, filename_suffix="camera")


def _largest_face(faces: np.ndarray):
    if faces is None or len(faces) == 0:
        return None
    areas = faces[:, 2] * faces[:, 3]
    return faces[int(np.argmax(areas))]


def _get_face_models():
    global _face_detector, _face_recognizer
    if _face_detector is not None and _face_recognizer is not None:
        return _face_detector, _face_recognizer

    if not os.path.exists(YUNET_PATH):
        raise RuntimeError(f"Model ontbreekt: {YUNET_PATH}")
    if not os.path.exists(SFACE_PATH):
        raise RuntimeError(f"Model ontbreekt: {SFACE_PATH}")

    _face_detector = cv2.FaceDetectorYN.create(
        YUNET_PATH, "", (320, 320), 0.9, 0.3, 5000
    )
    _face_recognizer = cv2.FaceRecognizerSF.create(SFACE_PATH, "")
    return _face_detector, _face_recognizer


def get_known_embeddings_cached(refresh_sec: float = 5.0):
    global _known_cache, _known_cache_at
    now = time.time()
    if _known_cache and (now - _known_cache_at) < refresh_sec:
        return _known_cache
    _known_cache = load_known_embeddings()
    _known_cache_at = now
    return _known_cache


def _best_match(recognizer, feat: np.ndarray, known: dict):
    feat_match = _normalize_match_feature(feat)
    if feat_match is None:
        return None, -1.0, -1.0

    scores = []
    for name, feats in known.items():
        best = -1.0
        for f in feats:
            f_match = _normalize_match_feature(f)
            if f_match is None or f_match.shape != feat_match.shape:
                continue
            s = float(recognizer.match(feat_match, f_match, cv2.FaceRecognizerSF_FR_COSINE))
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


def _normalize_match_feature(feat):
    if feat is None:
        return None
    arr = np.asarray(feat, dtype=np.float32)
    if arr.size == 0:
        return None
    return arr.reshape(1, -1)


def _annotate_face(img, face_row, label: str, color):
    x, y, w, h = face_row[:4].astype(int)
    cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
    cv2.putText(
        img,
        label,
        (max(0, x), max(20, y - 10)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        color,
        2,
    )


def annotate_uploaded_image(img: np.ndarray, threshold: float = 0.50, margin: float = 0.06, min_face: int = 80):
    known = load_known_embeddings()
    print(f"Annotating image with {len(known)} known people loaded.")
    if not known:
        raise RuntimeError("Geen bekende personen in known/. Voeg eerst .npz profielen toe.")

    with _face_lock:
        detector, recognizer = _get_face_models()
        h, w = img.shape[:2]
        detector.setInputSize((w, h))
        _, faces = detector.detect(img)

        annotated = img.copy()
        results = []
        if faces is None or len(faces) == 0:
            return annotated, results

        for idx, face in enumerate(faces):
            x, y, fw, fh = face[:4].astype(int)

            if fw < min_face:
                label = f"SMALL ({fw}px)"
                _annotate_face(annotated, face, label, (0, 255, 255))
                results.append({
                    "idx": idx + 1,
                    "label": "TOO_SMALL",
                    "score": "",
                    "x": x, "y": y, "w": fw, "h": fh,
                })
                continue

            try:
                aligned = recognizer.alignCrop(img, face)
                feat = recognizer.feature(aligned).astype(np.float32)
            except Exception:
                _annotate_face(annotated, face, "FEATURE_FAIL", (0, 0, 255))
                results.append({
                    "idx": idx + 1,
                    "label": "FEATURE_FAIL",
                    "score": "",
                    "x": x, "y": y, "w": fw, "h": fh,
                })
                continue

            best_name, best_score, second_score = _best_match(recognizer, feat, known)
            confident = (best_score >= threshold) and ((best_score - second_score) >= margin)

            if confident and best_name:
                label = f"{best_name} ({best_score:.2f})"
                _annotate_face(annotated, face, label, (0, 255, 0))
                results.append({
                    "idx": idx + 1,
                    "label": best_name,
                    "score": f"{best_score:.4f}",
                    "x": x, "y": y, "w": fw, "h": fh,
                })
            else:
                label = f"UNKNOWN ({best_score:.2f})"
                _annotate_face(annotated, face, label, (0, 0, 255))
                results.append({
                    "idx": idx + 1,
                    "label": "UNKNOWN",
                    "score": f"{best_score:.4f}",
                    "x": x, "y": y, "w": fw, "h": fh,
                })

    return annotated, results


def extract_features_from_unknown_folder(folder_path: str, min_face: int = 80):
    files = iter_image_files(folder_path)
    feats = []
    skipped = 0

    with _face_lock:
        detector, recognizer = _get_face_models()
        for fn in files:
            img_path = os.path.join(folder_path, fn)
            img = cv2.imread(img_path)
            if img is None:
                skipped += 1
                continue

            h, w = img.shape[:2]
            detector.setInputSize((w, h))
            _, faces = detector.detect(img)
            face = _largest_face(faces)
            if face is None:
                skipped += 1
                continue

            fw = int(face[2])
            if fw < min_face:
                skipped += 1
                continue

            try:
                aligned = recognizer.alignCrop(img, face)
                feat = recognizer.feature(aligned).astype(np.float32)
            except Exception:
                skipped += 1
                continue

            feats.append(feat)

    return feats, len(files), skipped


def extract_feature_from_image_path(img_path: str, min_face: int = 50):
    img = cv2.imread(img_path)
    if img is None:
        return None, "READ_FAIL"

    with _face_lock:
        detector, recognizer = _get_face_models()
        h, w = img.shape[:2]
        detector.setInputSize((w, h))
        _, faces = detector.detect(img)
        face = _largest_face(faces)
        if face is None:
            return None, "NO_FACE"
        fw = int(face[2])
        if fw < min_face:
            return None, "FACE_TOO_SMALL"

        try:
            aligned = recognizer.alignCrop(img, face)
            feat = recognizer.feature(aligned).astype(np.float32)
            return feat, "OK"
        except Exception:
            return None, "FEATURE_FAIL"





# ---- Routes ----
@app.route("/")
def root_redirect():
    return redirect(url_for("unknown_page"))

@app.route("/personen")
def personen():
    source = (request.args.get("source") or get_recognition_source()).strip().lower()
    if source not in ("local", "droidcam"):
        source = get_recognition_source()
    return render_template(
        "personen.html",
        running=recognition_running(),
        known_people=list_known_people(),
        source=source,
    )

@app.route("/personen/start", methods=["POST"])
def personen_start():
    source = (request.form.get("source") or "local").strip().lower()
    if source not in ("local", "droidcam"):
        source = "local"
    start_recognition(source=source)
    return redirect(url_for("personen", source=source))

@app.route("/personen/stop", methods=["POST"])
def personen_stop():
    stop_recognition()
    return redirect(url_for("personen"))


@app.route("/nl-launch")
def nl_launch_page():
    source = (request.args.get("source") or get_recognition_source()).strip().lower()
    if source not in ("local", "droidcam"):
        source = get_recognition_source()
    return render_template(
        "nl_launch.html",
        running=recognition_running(),
        source=source,
        recognition_script=RECOGNITION_SCRIPT,
        msg=request.args.get("msg", ""),
        level=request.args.get("level", "info"),
    )


@app.route("/nl-launch/start", methods=["POST"])
def nl_launch_start():
    source = (request.form.get("source") or get_recognition_source()).strip().lower()
    if source not in ("local", "droidcam"):
        source = "local"

    if recognition_running():
        return redirect(url_for(
            "nl_launch_page",
            level="info",
            msg="nl_launchv2.py draait al.",
            source=source,
        ))

    try:
        start_recognition(source=source)
    except Exception as e:
        return redirect(url_for(
            "nl_launch_page",
            level="error",
            msg=f"nl_launchv2.py starten mislukt: {e}",
            source=source,
        ))

    return redirect(url_for(
        "nl_launch_page",
        level="ok",
        msg="nl_launchv2.py gestart.",
        source=source,
    ))


@app.route("/nl-launch/stop", methods=["POST"])
def nl_launch_stop():
    if not recognition_running():
        stop_recognition()
        return redirect(url_for(
            "nl_launch_page",
            level="info",
            msg="nl_launchv2.py draait niet.",
        ))

    stop_recognition()
    return redirect(url_for(
        "nl_launch_page",
        level="ok",
        msg="Stop signaal naar nl_launchv2.py gestuurd.",
    ))


@app.route("/api/personen/status")
def api_personen_status():
    return jsonify({
        "running": recognition_running(),
        "known_count": len(list_known_people()),
        "source": get_recognition_source(),
    })

@app.route("/api/personen/log")
def api_personen_log():
    with _log_lock:
        return jsonify({"lines": list(_log_lines)})


@app.route("/api/tts/menu", methods=["POST"])
def api_tts_menu():
    payload = request.get_json(silent=True) or {}
    text = str(payload.get("text", "")).strip()
    if not text:
        return jsonify({"ok": False, "error": "Geen tekst opgegeven."}), 400

    # Vermijd spam bij snelle refreshes.
    global _menu_tts_last_at
    now = time.time()
    with _menu_tts_lock:
        if (now - _menu_tts_last_at) < 1.5:
            return jsonify({"ok": True, "queued": False})
        _menu_tts_last_at = now

    text = text[:280]
    _speak_menu_async(text)
    return jsonify({"ok": True, "queued": True})


@app.route("/ollama-chat")
def ollama_chat_page():
    return render_template(
        "ollama_chat.html",
        url=OLLAMA_CHAT_URL,
        model=OLLAMA_CHAT_MODEL,
    )


@app.route("/api/ollama-chat", methods=["POST"])
def api_ollama_chat():
    payload = request.get_json(silent=True) or {}
    messages = _normalize_chat_messages(payload.get("messages"))
    if not messages:
        return jsonify({"ok": False, "error": "Geen geldige berichten ontvangen."}), 400

    try:
        reply = _ollama_chat(messages)
    except urllib.error.HTTPError as e:
        try:
            detail = e.read().decode("utf-8", errors="replace").strip()
        except Exception:
            detail = str(e)
        return jsonify({"ok": False, "error": f"Ollama HTTP-fout: {detail or e.reason}"}), 502
    except urllib.error.URLError as e:
        return jsonify({"ok": False, "error": f"Ollama niet bereikbaar: {e.reason}"}), 502
    except Exception as e:
        return jsonify({"ok": False, "error": f"Ollama request mislukt: {e}"}), 500

    return jsonify({"ok": True, "message": reply})


@app.route("/settings", methods=["GET", "POST"])
def settings_page():
    msg = request.args.get("msg", "")
    level = request.args.get("level", "info")

    if request.method == "POST":
        settings = current_app_settings()
        settings["segmentation_server"] = (request.form.get("segmentation_server") or "").strip() or settings["segmentation_server"]
        settings["droidcam_url"] = (request.form.get("droidcam_url") or "").strip() or settings["droidcam_url"]
        settings["segmentation_model"] = (request.form.get("segmentation_model") or "").strip() or settings["segmentation_model"]
        settings["voice_volume"] = _coerce_voice_volume(request.form.get("voice_volume"), settings["voice_volume"])
        settings["ollama_chat_url"] = (request.form.get("ollama_chat_url") or "").strip() or settings["ollama_chat_url"]
        settings["ollama_chat_model"] = (request.form.get("ollama_chat_model") or "").strip() or settings["ollama_chat_model"]

        try:
            save_app_settings(settings)
            apply_runtime_settings(settings)
            msg = "Instellingen opgeslagen."
            level = "ok"
        except Exception as e:
            msg = f"Instellingen opslaan mislukt: {e}"
            level = "error"

        return redirect(url_for("settings_page", msg=msg, level=level))

    return render_template(
        "settings.html",
        msg=msg,
        level=level,
        settings=current_app_settings(),
        nl_launch_running=recognition_running(),
        nl_launch_source=get_recognition_source(),
        recognition_script=RECOGNITION_SCRIPT,
    )


def _run_system_action_later(cmds):
    def _worker():
        time.sleep(1.0)
        for cmd in cmds:
            try:
                res = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
                if res.returncode == 0:
                    break
            except Exception:
                continue
    t = threading.Thread(target=_worker, daemon=True)
    t.start()


@app.route("/settings/restart", methods=["POST"])
def settings_restart():
    # Strak: enkel reboot via systemd.
    _run_system_action_later([
        ["sudo", "-n", "systemctl", "reboot"],
        ["systemctl", "reboot"],
    ])
    return redirect(url_for("settings_page", level="ok", msg="Herstart aangevraagd."))


@app.route("/settings/shutdown", methods=["POST"])
def settings_shutdown():
    # Strak: enkel poweroff via systemd.
    _run_system_action_later([
        ["sudo", "-n", "systemctl", "poweroff"],
        ["systemctl", "poweroff"],
    ])
    return redirect(url_for("settings_page", level="ok", msg="Shutdown aangevraagd."))


@app.route("/settings/nl-launch/start", methods=["POST"])
def settings_nl_launch_start():
    source = (request.form.get("source") or get_recognition_source()).strip().lower()
    if source not in ("local", "droidcam"):
        source = "local"

    if recognition_running():
        return redirect(url_for(
            "settings_page",
            level="info",
            msg="nl_launchv2.py draait al.",
        ))

    try:
        start_recognition(source=source)
    except Exception as e:
        return redirect(url_for(
            "settings_page",
            level="error",
            msg=f"nl_launchv2.py starten mislukt: {e}",
        ))

    return redirect(url_for(
        "settings_page",
        level="ok",
        msg="nl_launchv2.py gestart.",
    ))


@app.route("/settings/nl-launch/stop", methods=["POST"])
def settings_nl_launch_stop():
    if not recognition_running():
        stop_recognition()
        return redirect(url_for(
            "settings_page",
            level="info",
            msg="nl_launchv2.py draait niet.",
        ))

    stop_recognition()
    return redirect(url_for(
        "settings_page",
        level="ok",
        msg="Stop signaal naar nl_launchv2.py gestuurd.",
    ))


@app.route("/camera")
def camera_page():
    source = (request.args.get("source") or get_stream_source()).strip().lower()
    if source not in ("local", "droidcam"):
        source = get_stream_source()
    return render_template(
        "camera.html",
        stream_on=is_stream_enabled(),
        stream_source=source,
        msg=request.args.get("msg", ""),
        level=request.args.get("level", "info"),
        capture_name=request.args.get("name", ""),
    )

@app.route("/segmentation")
def segmentation_page():
    st = get_seg_state()
    source = (request.args.get("source") or st.get("source") or "websocket").strip().lower()
    if source not in ("websocket", "local", "droidcam"):
        source = st.get("source") or "websocket"
    return render_template(
        "segmentation.html",
        running=st["running"],
        connected=st["connected"],
        heading=st["heading"],
        source=source,
        active_source=st.get("source", "websocket"),
        msg=request.args.get("msg", ""),
        level=request.args.get("level", "info"),
        signaling_server=SIGNALING_SERVER_URL,
        model_path=SEG_MODEL_PATH,
    )

@app.route("/segmentation/start", methods=["POST"])
def segmentation_start():
    try:
        source = (request.form.get("source") or "websocket").strip().lower()
        if source not in ("websocket", "local", "droidcam"):
            source = "websocket"
        start_segmentation_stream(source=source)
        return redirect(url_for("segmentation_page", source=source, level="ok", msg="Segmentatie gestart."))
    except Exception as e:
        return redirect(url_for("segmentation_page", level="error", msg=f"Start mislukt: {e}"))

@app.route("/segmentation/stop", methods=["POST"])
def segmentation_stop():
    stop_segmentation_stream()
    return redirect(url_for("segmentation_page", level="ok", msg="Segmentatie gestopt."))

@app.route("/segmentation_feed")
def segmentation_feed():
    st = get_seg_state()
    if not st["running"]:
        return ("", 204)
    return Response(gen_segmentation_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/api/segmentation/status")
def api_segmentation_status():
    st = get_seg_state()
    return jsonify(st)


def _encode_mjpeg_frame(frame: np.ndarray):
    ok, buffer = cv2.imencode(".jpg", frame)
    if not ok:
        return None
    return (
        b"--frame\r\n"
        b"Content-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n"
    )


def _make_segmentation_test_error_frame(message: str, detail: str = ""):
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.putText(frame, "Segmentation Test", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
    cv2.putText(frame, message[:48], (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 180, 255), 2)
    if detail:
        words = detail.split()
        lines = []
        line = []
        for word in words:
            if len(" ".join(line + [word])) > 52:
                lines.append(" ".join(line))
                line = [word]
            else:
                line.append(word)
        if line:
            lines.append(" ".join(line))
        for idx, text in enumerate(lines[:6], start=0):
            y = 150 + (idx * 36)
            cv2.putText(frame, text, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (220, 220, 220), 2)
    return frame


def _iter_segmentation_test_frames(source: str, model_source: str):
    selected_source = (source or "local").strip().lower()
    if selected_source not in ("local", "droidcam"):
        selected_source = "local"

    try:
        get_segmentation_model(model_source=model_source)
    except Exception as e:
        frame_bytes = _encode_mjpeg_frame(_make_segmentation_test_error_frame("Model load failed", str(e)))
        if frame_bytes is not None:
            while True:
                yield frame_bytes
                time.sleep(1.0)
        return

    if selected_source == "droidcam":
        stream_url = _normalize_droidcam_url(MOBILE_VIEW_DROIDCAM_URL)
        while True:
            try:
                for jpeg in _iter_http_mjpeg_frames(stream_url):
                    arr = np.frombuffer(jpeg, dtype=np.uint8)
                    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                    if frame is None:
                        continue
                    frame = cv2.resize(frame, (MOBILE_VIEW_WIDTH, MOBILE_VIEW_HEIGHT), interpolation=cv2.INTER_AREA)
                    overlay, _ = process_segment_frame(
                        frame,
                        model_source=model_source,
                        draw_polygons=True,
                        allow_detection_boxes=True,
                    )
                    frame_bytes = _encode_mjpeg_frame(overlay)
                    if frame_bytes is not None:
                        yield frame_bytes
            except Exception as e:
                frame_bytes = _encode_mjpeg_frame(_make_segmentation_test_error_frame("DroidCam error", str(e)))
                if frame_bytes is not None:
                    yield frame_bytes
                time.sleep(0.8)
        return

    cam = get_camera(0)
    while True:
        ok, frame = cam.read()
        if not ok or frame is None:
            error_frame = _make_segmentation_test_error_frame("Camera error", "Geen frame ontvangen van lokale camera.")
            frame_bytes = _encode_mjpeg_frame(error_frame)
            if frame_bytes is not None:
                yield frame_bytes
            time.sleep(0.2)
            continue

        try:
            overlay, _ = process_segment_frame(
                frame,
                model_source=model_source,
                draw_polygons=True,
                allow_detection_boxes=True,
            )
        except Exception as e:
            overlay = _make_segmentation_test_error_frame("Inference error", str(e))

        frame_bytes = _encode_mjpeg_frame(overlay)
        if frame_bytes is not None:
            yield frame_bytes


@app.route("/segmentation-test")
def segmentation_test_page():
    model_options = list_segmentation_model_options()
    selected_model = _coerce_segmentation_model(request.args.get("model_path"), SEG_MODEL_PATH)
    source = (request.args.get("source") or "local").strip().lower()
    if source not in ("local", "droidcam"):
        source = "local"
    active = request.args.get("run", "1") != "0"
    selected_model_path = _resolve_segmentation_model_source(selected_model)

    return render_template(
        "segmentation_test.html",
        source=source,
        active=active,
        model_options=model_options,
        model_path=selected_model,
        model_full_path=selected_model_path,
        droidcam_url=MOBILE_VIEW_DROIDCAM_URL,
        msg=request.args.get("msg", ""),
        level=request.args.get("level", "info"),
    )


@app.route("/segmentation-test-feed")
def segmentation_test_feed():
    source = (request.args.get("source") or "local").strip().lower()
    model_source = _coerce_segmentation_model(request.args.get("model_path"), SEG_MODEL_PATH)
    return Response(
        _iter_segmentation_test_frames(source=source, model_source=model_source),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )

@app.route("/object-detection", methods=["GET", "POST"])
def object_detection_page():
    msg = ""
    level = "info"
    image_b64 = ""
    detections = []
    filename = ""
    inference_ms = None
    model_options = list_detection_model_options()
    selected_model = _coerce_detection_model(request.form.get("model_path"), DET_MODEL_PATH)
    confidence = _coerce_det_confidence(request.form.get("confidence", DET_CONFIDENCE), DET_CONFIDENCE)

    if request.method == "POST":
        up = request.files.get("photo")
        if up is None or not up.filename:
            msg = "Selecteer een foto."
            level = "error"
        elif not is_allowed_image_filename(up.filename):
            msg = "Bestandstype niet ondersteund. Gebruik jpg/jpeg/png/bmp/webp."
            level = "error"
        else:
            filename = os.path.basename(up.filename)
            data = up.read()
            arr = np.frombuffer(data, dtype=np.uint8)
            img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if img is None:
                msg = "Kon de afbeelding niet lezen."
                level = "error"
            else:
                try:
                    annotated, detections, confidence, inference_ms = detect_objects_uploaded_image(
                        img,
                        confidence=confidence,
                        model_source=selected_model,
                    )
                    ok, enc = cv2.imencode(".jpg", annotated)
                    if not ok:
                        raise RuntimeError("Annotatie kon niet als JPG worden opgeslagen.")
                    image_b64 = base64.b64encode(enc.tobytes()).decode("ascii")
                    msg = f"Klaar: {len(detections)} object(en) gedetecteerd."
                    level = "ok"
                except Exception as e:
                    msg = f"Objectdetectie mislukt: {e}"
                    level = "error"

    return render_template(
        "object_detection.html",
        msg=msg,
        level=level,
        image_b64=image_b64,
        detections=detections,
        filename=filename,
        confidence=confidence,
        inference_ms=inference_ms,
        model_path=selected_model,
        model_options=model_options,
    )

@app.route("/annotate-photo", methods=["GET", "POST"])
def annotate_photo_page():
    msg = ""
    level = "info"
    image_b64 = ""
    results = []
    filename = ""

    if request.method == "POST":
        up = request.files.get("photo")
        if up is None or not up.filename:
            msg = "Selecteer een foto."
            level = "error"
        elif not is_allowed_image_filename(up.filename):
            msg = "Bestandstype niet ondersteund. Gebruik jpg/jpeg/png/bmp/webp."
            level = "error"
        else:
            filename = os.path.basename(up.filename)
            data = up.read()
            arr = np.frombuffer(data, dtype=np.uint8)
            img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if img is None:
                msg = "Kon de afbeelding niet lezen."
                level = "error"
            else:
                try:
                    annotated, results = annotate_uploaded_image(img)
                    ok, enc = cv2.imencode(".jpg", annotated)
                    if not ok:
                        raise RuntimeError("Annotatie kon niet als JPG worden opgeslagen.")
                    image_b64 = base64.b64encode(enc.tobytes()).decode("ascii")
                    msg = f"Klaar: {len(results)} gezicht(en) verwerkt."
                    level = "ok"
                except Exception as e:
                    msg = f"Annotatie mislukt: {e}"
                    level = "error"

    return render_template(
        "annotate_photo.html",
        msg=msg,
        level=level,
        image_b64=image_b64,
        results=results,
        filename=filename,
    )


@app.route("/mobile-face")
def mobile_face_page():
    stream_url = _normalize_droidcam_url(MOBILE_VIEW_DROIDCAM_URL)
    return render_template("mobile_face.html", stream_url=stream_url)


@app.route("/mobile_view")
def mobile_face_page_alias():
    return redirect(url_for("mobile_face_page"))


@app.route("/mobile-face/feed")
def mobile_face_feed():
    return Response(gen_mobile_face_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/mobile-face/reconnect", methods=["POST"])
def mobile_face_reconnect():
    release_mobile_cam()
    return redirect(url_for("mobile_face_page"))


@app.route("/api/mobile-face-detect", methods=["POST"])
def api_mobile_face_detect():
    payload = request.get_json(silent=True) or {}
    b64 = (payload.get("image") or "").strip()
    if not b64:
        return jsonify({"ok": False, "error": "Geen image payload."}), 400

    # data URL toegestaan: data:image/jpeg;base64,...
    if b64.startswith("data:image"):
        parts = b64.split(",", 1)
        if len(parts) != 2:
            return jsonify({"ok": False, "error": "Ongeldige data URL."}), 400
        b64 = parts[1]

    try:
        jpeg = base64.b64decode(b64)
        arr = np.frombuffer(jpeg, dtype=np.uint8)
        frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    except Exception:
        frame = None

    if frame is None:
        return jsonify({"ok": False, "error": "Kon frame niet decoderen."}), 400

    boxes = []
    try:
        with _face_lock:
            detector, _ = _get_face_models()
            h, w = frame.shape[:2]
            detector.setInputSize((w, h))
            _, faces = detector.detect(frame)
            if faces is not None and len(faces) > 0:
                for f in faces:
                    x, y, fw, fh = f[:4].astype(int)
                    boxes.append({"x": int(x), "y": int(y), "w": int(fw), "h": int(fh)})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500

    return jsonify({
        "ok": True,
        "width": int(frame.shape[1]),
        "height": int(frame.shape[0]),
        "boxes": boxes,
    })


@app.route("/faces-extract", methods=["GET", "POST"])
def faces_extract_page():
    msg = request.args.get("msg", "")
    level = request.args.get("level", "info")
    manifest = None
    session = ""

    if request.method == "POST":
        uploads = request.files.getlist("photos")
        if not uploads:
            one = request.files.get("photo")
            if one is not None:
                uploads = [one]

        valid_uploads = [u for u in uploads if u is not None and u.filename]
        if not valid_uploads:
            msg = "Selecteer minstens 1 foto."
            level = "error"
        else:
            try:
                with _face_lock:
                    detector, recognizer = _get_face_models()
                    known = load_known_embeddings()
                    session = uuid.uuid4().hex
                    session_dir = os.path.join(FACE_EXTRACT_TMP_DIR, session)
                    os.makedirs(session_dir, exist_ok=True)

                    entries = []
                    source_files = []
                    face_id = 0

                    for up in valid_uploads:
                        src_name = os.path.basename(up.filename or "")
                        if not is_allowed_image_filename(src_name):
                            continue

                        data = up.read()
                        arr = np.frombuffer(data, dtype=np.uint8)
                        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                        if img is None:
                            continue

                        source_files.append(src_name)
                        h, w = img.shape[:2]
                        detector.setInputSize((w, h))
                        _, faces = detector.detect(img)
                        if faces is None or len(faces) == 0:
                            continue

                        # Sorteer links->rechts voor voorspelbare volgorde.
                        face_rows = sorted(list(faces), key=lambda r: float(r[0]))
                        for face in face_rows:
                            x, y, fw, fh = face[:4].astype(int)
                            pad_w = int(fw * 0.15)
                            pad_h = int(fh * 0.15)
                            x1 = max(0, x - pad_w)
                            y1 = max(0, y - pad_h)
                            x2 = min(w, x + fw + pad_w)
                            y2 = min(h, y + fh + pad_h)

                            crop = img[y1:y2, x1:x2]
                            if crop is None or crop.size == 0:
                                continue

                            face_id += 1
                            img_name = f"face_{face_id:04d}.jpg"
                            img_path = os.path.join(session_dir, img_name)
                            cv2.imwrite(img_path, crop)

                            feat_name = None
                            suggested_name = ""
                            suggested_score = ""
                            try:
                                aligned = recognizer.alignCrop(img, face)
                                feat = recognizer.feature(aligned).astype(np.float32)
                                feat_name = f"face_{face_id:04d}.npy"
                                np.save(os.path.join(session_dir, feat_name), feat)
                                best_name, best_score, _ = _best_match(recognizer, feat, known) if known else (None, -1.0, -1.0)
                                if best_name:
                                    suggested_name = best_name
                                    suggested_score = f"{best_score:.4f}"
                            except Exception:
                                feat_name = None

                            entries.append({
                                "id": face_id,
                                "image": img_name,
                                "feature": feat_name,
                                "box": [int(x), int(y), int(fw), int(fh)],
                                "source_filename": src_name,
                                "suggested_name": suggested_name,
                                "suggested_score": suggested_score,
                            })

                    if not entries:
                        shutil.rmtree(session_dir, ignore_errors=True)
                        msg = "Geen bruikbare gezichten gevonden in de geselecteerde foto('s)."
                        level = "error"
                    else:
                        manifest = {
                            "session": session,
                            "source_files": source_files,
                            "entries": entries,
                        }
                        save_face_extract_manifest(session_dir, manifest)
                        msg = (
                            f"{len(entries)} gezicht(en) gevonden in "
                            f"{len(source_files)} foto('s). Vul per gezicht een naam in."
                        )
                        level = "ok"
            except Exception as e:
                msg = f"Verwerken mislukt: {e}"
                level = "error"

    return render_template(
        "faces_extract.html",
        msg=msg,
        level=level,
        manifest=manifest,
        session=session,
    )


@app.route("/faces-extract/tmp/<session>/<path:filename>")
def faces_extract_tmp_file(session, filename):
    session_dir = resolve_face_extract_session_dir(session)
    if session_dir is None or not os.path.isdir(session_dir):
        return ("", 404)
    return send_from_directory(session_dir, filename)


@app.route("/faces-extract/save", methods=["POST"])
def faces_extract_save():
    session = (request.form.get("session") or "").strip()
    session_dir = resolve_face_extract_session_dir(session)
    if session_dir is None or not os.path.isdir(session_dir):
        return redirect(url_for("faces_extract_page", level="error", msg="Sessie niet gevonden of verlopen."))

    manifest = load_face_extract_manifest(session_dir)
    if not manifest or not manifest.get("entries"):
        shutil.rmtree(session_dir, ignore_errors=True)
        return redirect(url_for("faces_extract_page", level="error", msg="Geen sessiegegevens gevonden."))

    by_person_features = {}
    saved_photos = 0
    saved_faces = 0
    skipped = 0

    try:
        for e in manifest["entries"]:
            rid = str(e.get("id"))
            raw_name = (request.form.get(f"name_{rid}") or "").strip()
            person = sanitize_person_name(raw_name)
            if not person:
                skipped += 1
                continue

            person_dir = os.path.join(KNOWN_DIR, person)
            os.makedirs(person_dir, exist_ok=True)

            src_img = os.path.join(session_dir, e.get("image", ""))
            if os.path.isfile(src_img):
                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                dst_img = os.path.join(person_dir, f"{ts}_{rid}.jpg")
                dst_img = unique_path(dst_img)
                shutil.copy2(src_img, dst_img)
                saved_photos += 1

            feat_file = e.get("feature")
            if feat_file:
                feat_path = os.path.join(session_dir, feat_file)
                if os.path.isfile(feat_path):
                    feat = np.load(feat_path).astype(np.float32)
                    if feat.ndim == 1 and feat.shape[0] > 0:
                        by_person_features.setdefault(person, []).append(feat)
                        saved_faces += 1

        for person, feats in by_person_features.items():
            append_features_for_person(person, feats)
    finally:
        shutil.rmtree(session_dir, ignore_errors=True)

    msg = f"Opslaan klaar: {saved_photos} foto('s), {saved_faces} feature(s), {skipped} overgeslagen."
    return redirect(url_for("faces_extract_page", level="ok", msg=msg))


@app.route("/unknown-process")
def unknown_process_page():
    msg = request.args.get("msg", "")
    level = request.args.get("level", "info")
    cleanup_old_unknown_root_images(max_age_days=7)
    files = list_unknown_root_images()
    items = []

    known = load_known_embeddings()
    try:
        with _face_lock:
            detector, recognizer = _get_face_models()
            for fn in files:
                p = os.path.join(UNKNOWN_DIR, fn)
                suggested_name = ""
                suggested_score = ""
                best_name = ""
                best_score = ""

                img = cv2.imread(p)
                if img is not None:
                    h, w = img.shape[:2]
                    detector.setInputSize((w, h))
                    _, faces = detector.detect(img)
                    face = _largest_face(faces)
                    if face is not None:
                        try:
                            aligned = recognizer.alignCrop(img, face)
                            feat = recognizer.feature(aligned).astype(np.float32)
                            bname, bscore, sscore = _best_match(recognizer, feat, known) if known else (None, -1.0, -1.0)
                            if bname:
                                best_name = bname
                                best_score = f"{bscore:.4f}"
                                if (bscore >= 0.70) and ((bscore - sscore) >= 0.06):
                                    suggested_name = bname
                                    suggested_score = f"{bscore:.4f}"
                        except Exception:
                            pass

                items.append({
                    "filename": fn,
                    "suggested_name": suggested_name,
                    "suggested_score": suggested_score,
                    "best_name": best_name,
                    "best_score": best_score,
                })
    except Exception as e:
        msg = f"Analyse niet beschikbaar: {e}"
        level = "error"

    return render_template("unknown_process.html", items=items, msg=msg, level=level)


@app.route("/unknown-process/file/<path:filename>")
def unknown_process_file(filename):
    return send_from_directory(UNKNOWN_DIR, filename)


@app.route("/unknown-process/save-one", methods=["POST"])
def unknown_process_save_one():
    filename = os.path.basename((request.form.get("filename") or "").strip())
    raw_name = (request.form.get("name") or "").strip()
    person = sanitize_person_name(raw_name)
    if not filename or not is_allowed_image_filename(filename):
        return redirect(url_for("unknown_process_page", level="error", msg="Ongeldige bestandsnaam."))
    if not person:
        return redirect(url_for("unknown_process_page", level="error", msg="Naam is ongeldig of leeg."))

    src = os.path.join(UNKNOWN_DIR, filename)
    if not os.path.isfile(src):
        return redirect(url_for("unknown_process_page", level="error", msg="Bestand bestaat niet meer."))

    feat, status = extract_feature_from_image_path(src, min_face=40)

    person_dir = os.path.join(KNOWN_DIR, person)
    os.makedirs(person_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    dst = unique_path(os.path.join(person_dir, f"{ts}_{filename}"))
    try:
        shutil.move(src, dst)
    except Exception as e:
        return redirect(url_for("unknown_process_page", level="error", msg=f"Bestand verplaatsen mislukt: {e}"))

    added = 0
    if feat is not None and feat.ndim == 1 and feat.shape[0] > 0:
        added = append_features_for_person(person, [feat])

    if added > 0:
        msg = f"{filename} opgeslagen bij {person}. Foto + feature toegevoegd."
    else:
        msg = f"{filename} opgeslagen bij {person}. Foto toegevoegd, geen feature ({status})."
    return redirect(url_for("unknown_process_page", level="ok", msg=msg))


@app.route("/unknown-process/delete-one", methods=["POST"])
def unknown_process_delete_one():
    filename = os.path.basename((request.form.get("filename") or "").strip())
    if not filename or not is_allowed_image_filename(filename):
        return redirect(url_for("unknown_process_page", level="error", msg="Ongeldige bestandsnaam."))

    p = os.path.join(UNKNOWN_DIR, filename)
    if not os.path.isfile(p):
        return redirect(url_for("unknown_process_page", level="error", msg="Bestand bestaat niet meer."))
    try:
        os.remove(p)
    except Exception as e:
        return redirect(url_for("unknown_process_page", level="error", msg=f"Verwijderen mislukt: {e}"))

    return redirect(url_for("unknown_process_page", level="ok", msg=f"{filename} verwijderd."))


@app.route("/unknown-process/delete-all", methods=["POST"])
def unknown_process_delete_all():
    os.makedirs(UNKNOWN_DIR, exist_ok=True)
    removed = 0
    failed = 0

    for fn in os.listdir(UNKNOWN_DIR):
        p = os.path.join(UNKNOWN_DIR, fn)
        if os.path.isfile(p) and is_allowed_image_filename(fn):
            try:
                os.remove(p)
                removed += 1
            except Exception:
                failed += 1

    if removed == 0 and failed == 0:
        return redirect(url_for("unknown_process_page", level="info", msg="Geen onverwerkte personen gevonden."))

    if failed > 0:
        return redirect(
            url_for(
                "unknown_process_page",
                level="error",
                msg=f"{removed} foto('s) verwijderd, {failed} niet kunnen verwijderen.",
            )
        )

    return redirect(url_for("unknown_process_page", level="ok", msg=f"Alle onverwerkte personen verwijderd ({removed} foto('s))."))


@app.route("/faces")
def unknown_page():
    return render_template(
        "unknown.html",
        people=list_known_people_with_photos(),
        sessions=list_unknown_sessions(),
        msg=request.args.get("msg", ""),
        level=request.args.get("level", "info"),
    )


@app.route("/unknown")
def unknown_legacy_redirect():
    return redirect(url_for("unknown_page"))


@app.route("/unknown/person/<person>")
def known_person_page(person):
    person_dir, npz_path = resolve_known_person_dir(person)
    if person_dir is None or not os.path.isfile(npz_path):
        return redirect(url_for("unknown_page", level="error", msg="Persoon bestaat niet (meer)."))

    files = iter_image_files(person_dir) if os.path.isdir(person_dir) else []
    return render_template(
        "known_person.html",
        person=person,
        files=files,
        msg=request.args.get("msg", ""),
        level=request.args.get("level", "info"),
    )


@app.route("/known/<person>/<path:filename>")
def known_person_file(person, filename):
    person_dir, npz_path = resolve_known_person_dir(person)
    if person_dir is None or not os.path.isfile(npz_path):
        return ("", 404)
    return send_from_directory(person_dir, filename)


@app.route("/known/<person>/delete-photo", methods=["POST"])
def known_delete_photo(person):
    person_dir, npz_path = resolve_known_person_dir(person)
    if person_dir is None or not os.path.isdir(person_dir):
        return redirect(url_for("unknown_page", level="error", msg="Persoon/map bestaat niet (meer)."))

    filename = os.path.basename((request.form.get("filename") or "").strip())
    if not filename:
        return redirect(url_for("known_person_page", person=person))
    if not is_allowed_image_filename(filename):
        return redirect(url_for("known_person_page", person=person))

    photo_path = os.path.join(person_dir, filename)
    if not os.path.isfile(photo_path):
        return redirect(
            url_for(
                "known_person_page",
                person=person,
            )
        )

    try:
        os.remove(photo_path)
    except Exception as e:
        return redirect(url_for("known_person_page", person=person, level="error", msg=f"Foto verwijderen mislukt: {e}"))

    # Houd .npz in sync met de resterende foto's.
    try:
        feats, total, _ = extract_features_from_unknown_folder(person_dir, min_face=80)
        if total == 0 or len(feats) == 0:
            if os.path.isfile(npz_path):
                os.remove(npz_path)
            msg = f"Foto verwijderd. Geen bruikbare foto's meer voor {person}; {person}.npz verwijderd."
        else:
            np.savez_compressed(npz_path, features=np.stack(feats, axis=0))
            msg = f"Foto verwijderd. {person}.npz opnieuw opgebouwd met {len(feats)} features."
    except Exception as e:
        msg = f"Foto verwijderd, maar .npz heropbouwen mislukt: {e}"
        return redirect(url_for("known_person_page", person=person, level="error", msg=msg))

    return redirect(url_for("known_person_page", person=person, level="ok", msg=msg))


@app.route("/known/delete", methods=["POST"])
def known_delete():
    person = (request.form.get("person") or "").strip()
    person_dir, npz_path = resolve_known_person_dir(person)

    if person_dir is None:
        return redirect(url_for("unknown_page", level="error", msg="Ongeldige persoonsnaam."))

    exists_npz = os.path.isfile(npz_path)
    exists_dir = os.path.isdir(person_dir)
    if not exists_npz and not exists_dir:
        return redirect(url_for("unknown_page", level="error", msg=f"Persoon {person} bestaat niet (meer)."))

    try:
        if exists_npz:
            os.remove(npz_path)
        if exists_dir:
            shutil.rmtree(person_dir, ignore_errors=True)
    except Exception as e:
        return redirect(url_for("unknown_page", level="error", msg=f"Verwijderen van {person} mislukt: {e}"))

    return redirect(url_for("unknown_page", level="ok", msg=f"Persoon {person} verwijderd."))


@app.route("/known/reprocess", methods=["POST"])
def known_reprocess():
    person_dirs = list_known_photo_dirs()
    if not person_dirs:
        return redirect(url_for("unknown_page", level="error", msg="Geen mappen gevonden in known/."))

    processed = 0
    skipped = 0
    total_features = 0

    for person in person_dirs:
        folder = os.path.join(KNOWN_DIR, person)
        try:
            feats, total, _ = extract_features_from_unknown_folder(folder, min_face=80)
        except Exception:
            skipped += 1
            continue

        if total == 0 or len(feats) == 0:
            skipped += 1
            continue

        out_path = os.path.join(KNOWN_DIR, f"{person}.npz")
        np.savez_compressed(out_path, features=np.stack(feats, axis=0))
        processed += 1
        total_features += len(feats)

    msg = (
        f"Gezichten verwerkt: {processed} map(pen), {skipped} overgeslagen, "
        f"{total_features} feature(s) opgeslagen."
    )
    level = "ok" if processed > 0 else "error"
    return redirect(url_for("unknown_page", level=level, msg=msg))


@app.route("/unknown/enroll", methods=["POST"])
def unknown_enroll():
    session = (request.form.get("session") or "").strip()
    raw_name = (request.form.get("name") or "").strip()
    person = sanitize_person_name(raw_name)

    if not session:
        return redirect(url_for("unknown_page", level="error", msg="Geen map geselecteerd."))
    if not person:
        return redirect(url_for("unknown_page", level="error", msg="Naam is ongeldig of leeg."))

    session_dir = resolve_unknown_session_dir(session)
    if session_dir is None:
        return redirect(url_for("unknown_page", level="error", msg="Ongeldige mapnaam."))
    if not os.path.isdir(session_dir):
        return redirect(url_for("unknown_page", level="error", msg="Map bestaat niet meer."))

    try:
        feats, total, skipped = extract_features_from_unknown_folder(session_dir, min_face=80)
    except Exception as e:
        return redirect(url_for("unknown_page", level="error", msg=f"Fout bij verwerken: {e}"))

    if len(feats) < 3:
        return redirect(
            url_for(
                "unknown_page",
                level="error",
                msg=f"Te weinig bruikbare gezichten in {session} ({len(feats)}/{total}). Minimaal 3 nodig.",
            )
        )

    os.makedirs(KNOWN_DIR, exist_ok=True)
    out_path = os.path.join(KNOWN_DIR, f"{person}.npz")
    overwritten = os.path.exists(out_path)
    np.savez_compressed(out_path, features=np.stack(feats, axis=0))

    try:
        person_dir, moved_count = move_unknown_session_images_to_known(session_dir, person, session)
        shutil.rmtree(session_dir, ignore_errors=True)
    except Exception as e:
        return redirect(
            url_for(
                "unknown_page",
                level="error",
                msg=f".npz opgeslagen, maar foto's verplaatsen/verwijderen mislukte: {e}",
            )
        )

    msg = (
        f"{person} opgeslagen met {len(feats)} features (geskipt: {skipped}, totaal: {total}). "
        f"{moved_count} foto's verplaatst naar {person_dir}. Unknown map verwijderd."
    )
    if overwritten:
        msg = f"{person} overschreven. " + msg
    return redirect(url_for("unknown_page", level="ok", msg=msg))


@app.route("/unknown/delete", methods=["POST"])
def unknown_delete():
    session = (request.form.get("session") or "").strip()
    session_dir = resolve_unknown_session_dir(session)

    if session_dir is None:
        return redirect(url_for("unknown_page", level="error", msg="Ongeldige mapnaam."))
    if not os.path.isdir(session_dir):
        return redirect(url_for("unknown_page", level="error", msg="Map bestaat niet meer."))

    try:
        shutil.rmtree(session_dir)
    except Exception as e:
        return redirect(url_for("unknown_page", level="error", msg=f"Verwijderen mislukt: {e}"))

    return redirect(url_for("unknown_page", level="ok", msg=f"Map {session} verwijderd."))

@app.get("/stayontrails")
def stayontrails():
    # Render de pagina (HTML/JS blijft client-side)
    return render_template("stayontrails.html")




@app.route("/unknown/upload-enroll", methods=["POST"])
def unknown_upload_enroll():
    raw_name = (request.form.get("name") or "").strip()
    person = sanitize_person_name(raw_name)
    if not person:
        return redirect(url_for("unknown_page", level="error", msg="Naam is ongeldig of leeg."))

    uploaded = request.files.getlist("photos")
    if not uploaded:
        return redirect(url_for("unknown_page", level="error", msg="Geen foto's geupload."))

    tmp_dir = os.path.join(BASE_DIR, "_tmp_uploads", uuid.uuid4().hex)
    os.makedirs(tmp_dir, exist_ok=True)
    saved = 0

    try:
        for f in uploaded:
            if not f or not f.filename:
                continue

            rel = f.filename.replace("\\", "/")
            rel = os.path.basename(rel)
            if not is_allowed_image_filename(rel):
                continue

            safe_name = sanitize_person_name(os.path.splitext(rel)[0]) or f"img_{saved+1:04d}"
            ext = os.path.splitext(rel)[1].lower()
            out_name = f"{saved+1:04d}_{safe_name}{ext}"
            out_path = os.path.join(tmp_dir, out_name)
            f.save(out_path)
            saved += 1

        if saved == 0:
            return redirect(url_for("unknown_page", level="error", msg="Geen geldige image-bestanden gevonden."))

        feats, total, skipped = extract_features_from_unknown_folder(tmp_dir, min_face=80)
    except Exception as e:
        return redirect(url_for("unknown_page", level="error", msg=f"Upload verwerken mislukt: {e}"))
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    if len(feats) < 3:
        return redirect(
            url_for(
                "unknown_page",
                level="error",
                msg=f"Te weinig bruikbare gezichten uit upload ({len(feats)}/{total}). Minimaal 3 nodig.",
            )
        )

    os.makedirs(KNOWN_DIR, exist_ok=True)
    out_path = os.path.join(KNOWN_DIR, f"{person}.npz")
    overwritten = os.path.exists(out_path)
    np.savez_compressed(out_path, features=np.stack(feats, axis=0))

    msg = f"{person} opgeslagen vanuit upload met {len(feats)} features (geskipt: {skipped}, totaal: {total})."
    if overwritten:
        msg = f"{person} overschreven. " + msg
    return redirect(url_for("unknown_page", level="ok", msg=msg))

@app.route("/camera/start", methods=["POST"])
def camera_start():
    source = (request.form.get("source") or "local").strip().lower()
    if source not in ("local", "droidcam"):
        source = "local"
    if source == "droidcam":
        release_camera()
    else:
        release_mobile_cam()
    set_stream_source(source)
    set_stream_enabled(True)
    return redirect(url_for("camera_page", source=source))

@app.route("/camera/stop", methods=["POST"])
def camera_stop():
    set_stream_enabled(False)
    release_camera()
    release_mobile_cam()
    return redirect(url_for("camera_page"))


@app.route("/camera/capture", methods=["POST"])
def camera_capture():
    raw_name = (request.form.get("name") or "").strip()
    source = (request.form.get("source") or get_stream_source()).strip().lower()
    if source not in ("local", "droidcam"):
        source = "local"
    person = sanitize_person_name(raw_name)
    if not person:
        return redirect(url_for("camera_page", level="error", msg="Naam is ongeldig of leeg.", name=raw_name, source=source))

    try:
        if source == "droidcam":
            frame = get_droidcam_last_frame(max_age_sec=3.0)
            if frame is None:
                f = _fetch_one_droidcam_frame()
                if f is not None:
                    frame = cv2.resize(f, (MOBILE_VIEW_WIDTH, MOBILE_VIEW_HEIGHT), interpolation=cv2.INTER_AREA)
            if frame is None:
                raise RuntimeError("Geen frame ontvangen van DroidCam.")
            out_path, added = capture_known_person_from_frame(person, frame, filename_suffix="droidcam")
        else:
            out_path, added = capture_known_person_from_camera(person, cam_index=0)
    except Exception as e:
        return redirect(url_for("camera_page", level="error", msg=f"Capture mislukt: {e}", name=person, source=source))

    msg = f"Foto opgeslagen voor {person}: {out_path}"
    if added > 0:
        msg += f" | {added} feature toegevoegd aan {person}.npz"
    return redirect(url_for("camera_page", level="ok", msg=msg, name=person, source=source))

@app.route("/video_feed")
def video_feed():
    # Als stream niet aan staat: geen feed (voorkomt eindeloos reconnecten)
    if not is_stream_enabled():
        return ("", 204)  # No Content
    return Response(gen_frames(source=get_stream_source()), mimetype="multipart/x-mixed-replace; boundary=frame")



if __name__ == "__main__":

    app.run(host="0.0.0.0", port=5000, debug=False)

    # HTTPS
    #app.run(
    #    host="0.0.0.0",
    #    port=5443,
    #    ssl_context=("../jetson-desktop+5.pem", "../jetson-desktop+5-key.pem"),
    #    debug=False
    #)
