#!/usr/bin/env python3
"""
Headless gezichtsherkenning + "auto-opslaan bij onbekende persoon" (OpenCV YuNet + SFace)

EXTRA:
- Bij detectie van een persoon (bekend of onbekend) wordt een FOTO-snapshot opgeslagen in map `snapshots/`
  met bestandsnaam:  Naam_yyyy_mm_dd_hh_mm_ss.jpg
  - Voor onbekend gebruiken we "Onbekend" als naam.
  - Om spam te vermijden: we bewaren enkel bij "BINNEN" (entry) voor bekende personen,
    en bij start van een onbekende sessie (na unknown_confirm_frames).

NIEUW (vereenvoudigd):
- Voor onbekende personen slaan we face-crops rechtstreeks op in:
    unknown/
  (dus geen submappen).

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
from datetime import datetime

YUNET_URL = "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx"
SFACE_URL = "https://github.com/opencv/opencv_zoo/raw/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx"


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


def best_match(recognizer, feat, known: dict):
    scores = []
    for name, feats in known.items():
        best = -1.0
        for f in feats:
            s = float(recognizer.match(feat, f, cv2.FaceRecognizerSF_FR_COSINE))
            if s > best:
                best = s
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


def open_camera_from_url(cam_url: str, width: int, height: int, fps: int):
    url = (cam_url or "").strip()
    cap = cv2.VideoCapture(url)
    if cap.isOpened():
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        cap.set(cv2.CAP_PROP_FPS, fps)
        print(f"[INFO] Camera geopend via URL: {url}", flush=True)
    return cap


def str2bool(s: str) -> bool:
    return str(s).strip().lower() in ("1", "true", "yes", "y", "on", "ja", "j")


def sanitize_name(name: str) -> str:
    name = name.strip().replace("/", "_").replace("\\", "_")
    name = name.replace("..", ".")
    return name


def ask_input(prompt: str) -> str:
    sys.stdout.write(prompt)
    sys.stdout.flush()
    return input()


# -----------------------------
# Foto snapshot (bekend/onbekend label)
# -----------------------------

def save_person_snapshot(frame, name: str, out_dir: str = "snapshots") -> str:
    """
    Slaat 1 JPG op in out_dir met naam: Naam_yyyy_mm_dd_hh_mm_ss.jpg
    """
    os.makedirs(out_dir, exist_ok=True)
    safe_name = sanitize_name(name) if name else "Onbekend"
    ts = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    path = os.path.join(out_dir, f"{safe_name}_{ts}.jpg")
    cv2.imwrite(path, frame)
    return path


# -----------------------------
# Unknown foto opslag (face-crop in unknown/)
# -----------------------------

def save_unknown_photo(frame, face_row, out_dir: str, idx: int) -> str:
    """
    Slaat een face-crop op in unknown/ met unieke bestandsnaam.
    """
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


def piper_say(text: str, model_path: str, sample_rate: int, length_scale: float = 1.0, volume: int = 100):
    p1 = subprocess.Popen(
        ["/home/jetson/piper/piper", "--model", model_path, "--output_raw", "--length_scale", str(length_scale)],
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
        pcm *= (vol / 100.0)
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


def tts_worker_loop(tts_queue: mp.Queue, stop_event: mp.Event, args):
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

        text = str(msg).strip()
        if not text:
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


def tts_enqueue(tts_queue, text: str):
    if tts_queue is None:
        return
    try:
        tts_queue.put_nowait(text)
    except pyqueue.Full:
        pass


# -----------------------------
# Snapshot opslag (features) - behouden, maar niet gebruikt voor deze vraag
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
    ap.add_argument("--cam_url", type=str, default="")
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--fps", type=int, default=15)
    ap.add_argument("--infer_every", type=int, default=2)

    ap.add_argument("--min_face", type=int, default=50)
    ap.add_argument("--score_th", type=float, default=0.9)
    ap.add_argument("--nms_th", type=float, default=0.3)
    ap.add_argument("--topk", type=int, default=5000)

    # Herkenning (confidence)
    ap.add_argument("--threshold", type=float, default=0.60)
    ap.add_argument("--margin", type=float, default=0.06)

    # Entry / leave
    ap.add_argument("--lost_timeout", type=float, default=1.0)
    ap.add_argument("--enter_confirm_frames", type=int, default=3)
    ap.add_argument("--reannounce_after", type=float, default=6.0)

    # Onbekend gedrag
    ap.add_argument("--unknown_seconds", type=float, default=5.0,
                    help="(oud) Als een onbekend gezicht zo lang zichtbaar blijft, vragen om op te slaan.")
    ap.add_argument("--unknown_confirm_frames", type=int, default=5,
                    help="Aantal opeenvolgende 'onbekend' frames voor we starten.")
    ap.add_argument("--cooldown_after_unknown", type=float, default=300.0,
                    help="Na een onbekend-afhandeling even wachten.")

    # Meteen snapshots (features) verzamelen (hier niet nodig, maar laten staan)
    ap.add_argument("--unknown_capture_interval", type=float, default=0.5,
                    help="(oud) Tijdens onbekend: neem een feature-snapshot om de N seconden.")
    ap.add_argument("--unknown_max_snaps", type=int, default=60,
                    help="(oud) Max aantal feature-snapshots dat we bijhouden.")

    # Opslag
    ap.add_argument("--known", type=str, default="known", help="Map met .npz identiteiten")
    ap.add_argument("--min_save_samples", type=int, default=20, help="(oud) Niet opslaan als er te weinig snapshots zijn")

    # Foto's (optioneel)
    ap.add_argument("--unknown_photos", type=str, default="unknown_photos",
                    help="(oud) Map om optioneel een JPG te bewaren")
    ap.add_argument("--save_unknown_snapshot", action="store_true",
                    help="(oud) Bewaar ook 1 JPG (laatste frame) wanneer je een nieuwe persoon opslaat.")

    # Piper TTS (NL voice)
    ap.add_argument("--no_tts", action="store_true")
    ap.add_argument("--speak", type=str, default="True")
    ap.add_argument("--piper_model", type=str, default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "voices", "nl_BE-nathalie-medium.onnx"))
    ap.add_argument("--piper_rate", type=int, default=22050)
    ap.add_argument("--piper_rate_auto", action="store_true")
    ap.add_argument("--piper_length_scale", type=float, default=1.0)
    ap.add_argument("--voice_volume", type=int, default=100)
    ap.add_argument("--tts_queue_size", type=int, default=20)

    args = ap.parse_args()
    args.voice_volume = max(0, min(100, int(args.voice_volume)))

    yunet_path = os.path.join("models", "face_detection_yunet_2023mar.onnx")
    sface_path = os.path.join("models", "face_recognition_sface_2021dec.onnx")
    download_if_missing(YUNET_URL, yunet_path)
    download_if_missing(SFACE_URL, sface_path)

    os.makedirs(args.known, exist_ok=True)

    # TTS
    stop_event = mp.Event()
    tts_queue = None
    tts_proc = None
    speak_enabled = (not args.no_tts) and str2bool(args.speak)
    if speak_enabled:
        tts_queue = mp.Queue(maxsize=args.tts_queue_size)
        tts_proc = mp.Process(target=tts_worker_loop, args=(tts_queue, stop_event, args), daemon=True)
        tts_proc.start()
        tts_enqueue(tts_queue, "Gezichtsherkenning is gestart.")

    # Camera
    if (args.cam_url or "").strip():
        cap = open_camera_from_url(args.cam_url, args.width, args.height, args.fps)
    else:
        cap = open_camera_linux(args.cam, args.width, args.height, args.fps)
    if not cap.isOpened():
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
        cap.release()
        return

    h, w = frame.shape[:2]
    detector = cv2.FaceDetectorYN.create(yunet_path, "", (w, h), args.score_th, args.nms_th, args.topk)
    recognizer = cv2.FaceRecognizerSF.create(sface_path, "")

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
    last_announced_at = {}  # name -> time

    # Unknown state
    unknown_consec = 0
    unknown_started_at = None
    last_unknown_handled_at = 0.0

    # Unknown foto reeks (rechtstreeks in unknown/)
    unknown_dir = "unknown"
    unknown_photo_count = 0
    unknown_photo_interval = 60  # 20 foto's in ~4s (pas aan)
    unknown_last_photo_at = 0.0

    # Snapshot-spam preventie voor "snapshots/" (bekend + 1 bij onbekend start)
    last_person_photo_at = {}  # name -> time
    person_photo_cooldown = 300.0  # sec

    frame_id = 0
    print("[INFO] Headless actief. Ctrl+C om te stoppen.", flush=True)

    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                time.sleep(0.1)
                continue

            frame_id += 1
            if frame_id % args.infer_every != 0:
                continue

            now = time.time()

            detector.setInputSize((w, h))
            _, faces = detector.detect(frame)
            face = largest_face(faces)

            if face is None:
                # leave detection
                if present and (now - last_seen) >= args.lost_timeout:
                    print(f"[INFO] {present_name} is uit beeld.", flush=True)
                    present = False
                    present_name = None
                    consec_count = 0
                    candidate_name = None

                # reset unknown tracking
                unknown_consec = 0
                unknown_started_at = None
                unknown_dir = "unknown"
                unknown_photo_count = 0
                unknown_last_photo_at = 0.0
                continue

            x, y, fw, fh = face[:4].astype(int)
            if fw < args.min_face:
                if present and (now - last_seen) >= args.lost_timeout:
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

            aligned = recognizer.alignCrop(frame, face)
            feat = recognizer.feature(aligned).astype(np.float32)

            best_name, best_score, second_score = best_match(recognizer, feat, known) if known else (None, -1.0, -1.0)
            confident = (best_name is not None) and (best_score >= args.threshold) and ((best_score - second_score) >= args.margin)

            # -------------------------
            # ONBEKEND: 20 face-crops bewaren in unknown/
            # -------------------------
            if not confident:
                # als er net iemand "present" was, kan die verdwijnen
                if present and (now - last_seen) >= args.lost_timeout:
                    print(f"[INFO] {present_name} is uit beeld.", flush=True)
                    present = False
                    present_name = None
                    consec_count = 0
                    candidate_name = None

                # cooldown na vorige unknown sessie
                if (now - last_unknown_handled_at) < args.cooldown_after_unknown:
                    unknown_consec = 0
                    unknown_started_at = None
                    unknown_dir = "unknown"
                    unknown_photo_count = 0
                    unknown_last_photo_at = 0.0
                    continue

                unknown_consec += 1
                if unknown_consec < args.unknown_confirm_frames:
                    continue

                # start unknown sessie
                if unknown_started_at is None:
                    unknown_started_at = now
                    unknown_photo_count = 0
                    unknown_last_photo_at = 0.0
                    os.makedirs(unknown_dir, exist_ok=True)

                    print(f"[INFO] Onbekende persoon gedetecteerd -> map: {unknown_dir}", flush=True)
                    #if speak_enabled:
                    #    tts_enqueue(tts_queue, "Ik zie iemand die ik nog niet ken.")

                if unknown_dir is not None:
                    if (now - unknown_last_photo_at) >= unknown_photo_interval:
                        unknown_photo_count += 1
                        p = save_unknown_photo(frame, face, unknown_dir, unknown_photo_count)
                        unknown_last_photo_at = now
                        print(f"[OK] Unknown foto {unknown_photo_count}/20: {p}", flush=True)

                # klaar: reset + start cooldown
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
            # BEKEND: entry announcement + 1 FOTO bij "BINNEN"
            # -------------------------
            last_seen = now

            # reset unknown tracking zodra we weer confident zijn
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
            if (now - last_spoke) < args.reannounce_after:
                present = True
                present_name = candidate_name
                consec_count = 0
                candidate_name = None
                continue

            present = True
            present_name = candidate_name
            last_announced_at[present_name] = now

            print(f"[INFO] BINNEN: {present_name} {richting} (score={best_score:.2f}, tweede={second_score:.2f})", flush=True)

            # extra check (je had dit ook)
            if best_score > args.threshold:
                last_t = last_person_photo_at.get(present_name, 0.0)
                print(f"[DEBUG] Now={now:.1f}, last_t={last_t:.1f}, cooldown={person_photo_cooldown}s", flush=True)
                if (now - last_t) >= person_photo_cooldown:
                    p = save_person_snapshot(frame, present_name, out_dir="snapshots")
                    last_person_photo_at[present_name] = now
                    print("[OK] Snapshot opgeslagen:", p, flush=True)

                    if speak_enabled:
                        tts_enqueue(tts_queue, f"Hallo {present_name}")

                consec_count = 0
                candidate_name = None

    except KeyboardInterrupt:
        print("\n[INFO] Stoppen...", flush=True)

    finally:
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
