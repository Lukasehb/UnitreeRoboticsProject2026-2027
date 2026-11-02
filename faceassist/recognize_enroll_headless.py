#!/usr/bin/env python3
"""
Headless face recognition + "auto-enroll on unknown" (OpenCV YuNet + SFace)

UPDATED behavior (per your request):
- As soon as an unknown person is detected (after --unknown_confirm_frames),
  we immediately start collecting "snapshots" (SFace feature vectors) every
  --unknown_capture_interval seconds.
- Only after the unknown persists for --unknown_seconds, we ask:
    "New person detected, you want to save this person?"
- If YES:
    - Ask for a name
    - Save immediately using the already-collected snapshots (NO new snapshots taken)
    - Optionally save ONE photo from the last seen frame (no extra capture)
- If NO: discard the collected snapshots and go into cooldown.

This script combines logic from:
- enroll_sface.py (sampling + save .npz + piper TTS)
- recognize_headless.py (headless loop, open_camera_linux, best_match, entry logic)
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
# Shared helpers (from your scripts)
# -----------------------------

def download_if_missing(url: str, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if os.path.exists(path):
        return
    print(f"[INFO] Downloading {os.path.basename(path)} ...", flush=True)
    urllib.request.urlretrieve(url, path)
    print(f"[OK] Saved to {path}", flush=True)


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


def face_direction(x: int, w_face: int, frame_w: int) -> str:
    cx = x + (w_face // 2)
    if cx < frame_w / 3:
        return "is on your left"
    elif cx > 2 * frame_w / 3:
        return "is on your right"
    return "is in front of you"


def open_camera_linux(cam_index: int, width: int, height: int, fps: int):
    dev = f"/dev/video{cam_index}"

    gst_pipeline = (
        f"v4l2src device={dev} ! "
        f"image/jpeg,width={width},height={height},framerate={fps}/1 ! "
        f"jpegdec ! videoconvert ! appsink drop=true sync=false max-buffers=1"
    )

    cap = cv2.VideoCapture(gst_pipeline, cv2.CAP_GSTREAMER)
    if cap.isOpened():
        print("[INFO] Camera opened via GStreamer.", flush=True)
        return cap

    cap = cv2.VideoCapture(cam_index, cv2.CAP_V4L2)
    if cap.isOpened():
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        cap.set(cv2.CAP_PROP_FPS, fps)
        print("[INFO] Camera opened via V4L2 (OpenCV).", flush=True)
        return cap

    cap = cv2.VideoCapture(cam_index)
    if cap.isOpened():
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        cap.set(cv2.CAP_PROP_FPS, fps)
        print("[INFO] Camera opened via default backend (OpenCV).", flush=True)
        return cap

    return cap


def str2bool(s: str) -> bool:
    return str(s).strip().lower() in ("1", "true", "yes", "y", "on")


def sanitize_name(name: str) -> str:
    name = name.strip().replace("/", "_").replace("\\", "_")
    name = name.replace("..", ".")
    return name


def ask_input(prompt: str) -> str:
    sys.stdout.write(prompt)
    sys.stdout.flush()
    return input()


# -----------------------------
# Piper TTS (from your scripts, lightly unified)
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


def piper_say(text: str, model_path: str, sample_rate: int, length_scale: float = 0.6):
    p1 = subprocess.Popen(
        ["piper", "--model", model_path, "--output_raw", "--length_scale", str(length_scale)],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
    )

    p2 = subprocess.Popen(
        ["aplay", "-r", str(sample_rate), "-f", "S16_LE", "-t", "raw", "-"],
        stdin=p1.stdout,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    try:
        if p1.stdin is not None:
            p1.stdin.write((text + "\n").encode("utf-8"))
            p1.stdin.close()
    except Exception:
        pass

    if p1.stdout is not None:
        p1.stdout.close()

    try:
        p2.wait(timeout=60)
    except subprocess.TimeoutExpired:
        p2.kill()

    try:
        p1.wait(timeout=60)
    except subprocess.TimeoutExpired:
        p1.kill()


def tts_worker_loop(tts_queue: mp.Queue, stop_event: mp.Event, args):
    signal.signal(signal.SIGINT, signal.SIG_IGN)

    try:
        subprocess.run(["piper", "--help"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
    except FileNotFoundError:
        print("[WARN] 'piper' not found in PATH.", flush=True)
        return

    model_path = os.path.expanduser(args.piper_model)
    if not os.path.exists(model_path):
        print(f"[WARN] Piper model not found: {model_path}", flush=True)
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
            piper_say(text, model_path=model_path, sample_rate=sample_rate, length_scale=args.piper_length_scale)
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
# Snapshot storage
# -----------------------------

def save_snapshot(frame, out_dir: str, tag: str) -> str:
    os.makedirs(out_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_tag = sanitize_name(tag) if tag else "unknown"
    path = os.path.join(out_dir, f"{ts}_{safe_tag}.jpg")
    cv2.imwrite(path, frame)
    return path


# -----------------------------
# Main loop
# -----------------------------

def main():
    ap = argparse.ArgumentParser()

    # Camera + detection
    ap.add_argument("--cam", type=int, default=0)
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--fps", type=int, default=15)
    ap.add_argument("--infer_every", type=int, default=2)

    ap.add_argument("--min_face", type=int, default=50)
    ap.add_argument("--score_th", type=float, default=0.9)
    ap.add_argument("--nms_th", type=float, default=0.3)
    ap.add_argument("--topk", type=int, default=5000)

    # Recognition confidence
    ap.add_argument("--threshold", type=float, default=0.50)
    ap.add_argument("--margin", type=float, default=0.06)

    # Entry / leave behavior
    ap.add_argument("--lost_timeout", type=float, default=1.0)
    ap.add_argument("--enter_confirm_frames", type=int, default=3)
    ap.add_argument("--reannounce_after", type=float, default=6.0)

    # Unknown handling
    ap.add_argument("--unknown_seconds", type=float, default=5.0,
                    help="If an unknown face stays visible this long, prompt to enroll.")
    ap.add_argument("--unknown_confirm_frames", type=int, default=5,
                    help="How many consecutive 'unknown' frames before starting unknown logic.")
    ap.add_argument("--cooldown_after_unknown", type=float, default=10.0,
                    help="After handling an unknown (save or skip), ignore unknown triggers for this many seconds.")

    # NEW: unknown capture behavior (start capturing immediately)
    ap.add_argument("--unknown_capture_interval", type=float, default=0.5,
                    help="While unknown is present, capture a feature snapshot every N seconds.")
    ap.add_argument("--unknown_max_snaps", type=int, default=60,
                    help="Max number of unknown feature snapshots to keep before prompting.")

    # Enrollment / storage
    ap.add_argument("--known", type=str, default="known", help="Directory with .npz identities")
    ap.add_argument("--min_save_samples", type=int, default=8, help="Don't save if fewer snapshots were captured")

    # Snapshot storage
    ap.add_argument("--unknown_photos", type=str, default="unknown_photos",
                    help="Where to store optional snapshots")
    ap.add_argument("--save_unknown_snapshot", action="store_true",
                    help="Also save a JPG snapshot (using the last frame) when saving a new person.")

    # Piper TTS
    ap.add_argument("--no_tts", action="store_true")
    ap.add_argument("--speak", type=str, default="True")
    ap.add_argument("--piper_model", type=str, default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "voices", "en_GB-alan-medium.onnx"))
    ap.add_argument("--piper_rate", type=int, default=22050)
    ap.add_argument("--piper_rate_auto", action="store_true")
    ap.add_argument("--piper_length_scale", type=float, default=0.6)
    ap.add_argument("--tts_queue_size", type=int, default=20)

    args = ap.parse_args()

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
        tts_enqueue(tts_queue, "Face recognition started.")

    # Camera
    cap = open_camera_linux(args.cam, args.width, args.height, args.fps)
    if not cap.isOpened():
        print("[ERROR] Cannot open camera.", flush=True)
        stop_event.set()
        if tts_queue is not None:
            try:
                tts_queue.put_nowait(None)
            except Exception:
                pass
        return

    ok, frame = cap.read()
    if not ok or frame is None:
        print("[ERROR] Cannot read first frame.", flush=True)
        cap.release()
        return

    h, w = frame.shape[:2]
    detector = cv2.FaceDetectorYN.create(yunet_path, "", (w, h), args.score_th, args.nms_th, args.topk)
    recognizer = cv2.FaceRecognizerSF.create(sface_path, "")

    known = load_known(args.known)
    if known:
        print("[INFO] Known:", ", ".join(sorted(known.keys())), flush=True)
        if speak_enabled:
            tts_enqueue(tts_queue, f"{len(known)} identities loaded.")
    else:
        print(f"[WARN] No known identities in '{args.known}'. Unknown-trigger enrollment will still work.", flush=True)
        if speak_enabled:
            tts_enqueue(tts_queue, "No known identities loaded. I can enroll new people when I see them.")

    # Present/entry state (like recognize_headless.py)
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

    # NEW: collect snapshots (features) immediately for unknown
    unknown_feats = []          # list of np.float32 feature vectors
    unknown_last_cap = 0.0      # time of last feature snapshot
    unknown_last_frame = None   # last frame for optional JPG save

    frame_id = 0
    print("[INFO] Headless running. Ctrl+C to stop.", flush=True)

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
                    print(f"[INFO] {present_name} left the frame.", flush=True)
                    present = False
                    present_name = None
                    consec_count = 0
                    candidate_name = None

                # reset unknown tracking when face absent
                unknown_consec = 0
                unknown_started_at = None
                unknown_feats = []
                unknown_last_cap = 0.0
                unknown_last_frame = None
                continue

            x, y, fw, fh = face[:4].astype(int)
            if fw < args.min_face:
                # treat as absent-ish
                if present and (now - last_seen) >= args.lost_timeout:
                    print(f"[INFO] {present_name} left the frame.", flush=True)
                    present = False
                    present_name = None

                unknown_consec = 0
                unknown_started_at = None
                unknown_feats = []
                unknown_last_cap = 0.0
                unknown_last_frame = None
                consec_count = 0
                candidate_name = None
                continue

            direction = face_direction(x, fw, w)

            aligned = recognizer.alignCrop(frame, face)
            feat = recognizer.feature(aligned).astype(np.float32)

            best_name, best_score, second_score = best_match(recognizer, feat, known) if known else (None, -1.0, -1.0)
            confident = (best_name is not None) and (best_score >= args.threshold) and ((best_score - second_score) >= args.margin)

            # -------------------------
            # Handle UNKNOWN (start capturing immediately)
            # -------------------------
            if not confident:
                # update "present" leave logic (unknown doesn't refresh last_seen)
                if present and (now - last_seen) >= args.lost_timeout:
                    print(f"[INFO] {present_name} left the frame.", flush=True)
                    present = False
                    present_name = None

                # cooldown so we don't spam prompts
                if (now - last_unknown_handled_at) < args.cooldown_after_unknown:
                    unknown_consec = 0
                    unknown_started_at = None
                    unknown_feats = []
                    unknown_last_cap = 0.0
                    unknown_last_frame = None
                    continue

                # require a few consecutive unknown frames before starting unknown logic
                unknown_consec += 1
                if unknown_consec < args.unknown_confirm_frames:
                    continue

                # unknown begins now
                if unknown_started_at is None:
                    unknown_started_at = now
                    unknown_feats = []
                    unknown_last_cap = 0.0
                    unknown_last_frame = None
                    print("[INFO] Unknown face detected. Capturing snapshots...", flush=True)
                    if speak_enabled:
                        tts_enqueue(tts_queue, "I see someone I do not recognize.")

                # keep last frame for optional saving later (no extra capture)
                unknown_last_frame = frame

                # capture feature "snapshots" immediately while unknown is present
                if (now - unknown_last_cap) >= args.unknown_capture_interval:
                    try:
                        aligned_u = recognizer.alignCrop(frame, face)
                        feat_u = recognizer.feature(aligned_u).astype(np.float32)
                        unknown_feats.append(feat_u)
                        unknown_last_cap = now

                        if len(unknown_feats) > args.unknown_max_snaps:
                            unknown_feats = unknown_feats[-args.unknown_max_snaps:]

                        if len(unknown_feats) % 10 == 0:
                            print(f"[INFO] Unknown snapshots: {len(unknown_feats)}", flush=True)
                    except Exception:
                        pass

                # after unknown_seconds -> ask to save
                if (now - unknown_started_at) >= args.unknown_seconds:
                    print("[INFO] New person detected. Prompting to save...", flush=True)
                    if speak_enabled:
                        tts_enqueue(tts_queue, "New person detected, you want to save this person? Type y or n in the terminal.")

                    ans = ask_input("New person detected, you want to save this person? (y/n): ").strip().lower()

                    if ans.startswith("y"):
                        if speak_enabled:
                            tts_enqueue(tts_queue, "Please type the person's name and press Enter.")
                        name = sanitize_name(ask_input("Name: "))

                        if not name:
                            print("[INFO] No name entered. Skipping save.", flush=True)
                            if speak_enabled:
                                tts_enqueue(tts_queue, "No name entered. I will not save anything.")
                        else:
                            # save directly from collected snapshots (NO new captures!)
                            if len(unknown_feats) >= args.min_save_samples:
                                out_path = os.path.join(args.known, f"{name}.npz")
                                np.savez_compressed(out_path, features=np.stack(unknown_feats, axis=0))
                                print("[OK] Saved:", out_path, flush=True)

                                if speak_enabled:
                                    tts_enqueue(tts_queue, f"Saved. {name} has been added.")

                                # reload so recognition works immediately
                                known = load_known(args.known)

                                # optional: save ONE photo from the last frame we already have
                                if args.save_unknown_snapshot and unknown_last_frame is not None:
                                    tagged_path = save_snapshot(unknown_last_frame, args.unknown_photos, name)
                                    print("[OK] Snapshot saved:", tagged_path, flush=True)
                            else:
                                print(f"[WARN] Too few snapshots ({len(unknown_feats)}). Not saving.", flush=True)
                                if speak_enabled:
                                    tts_enqueue(tts_queue, "I could not capture enough snapshots. I will not save.")

                    else:
                        print("[INFO] User chose not to save.", flush=True)
                        if speak_enabled:
                            tts_enqueue(tts_queue, "Okay. I will not save anything.")

                    # reset unknown tracking + start cooldown
                    last_unknown_handled_at = time.time()
                    unknown_consec = 0
                    unknown_started_at = None
                    unknown_feats = []
                    unknown_last_cap = 0.0
                    unknown_last_frame = None

                continue  # done with unknown path

            # -------------------------
            # Handle KNOWN (entry announcement)
            # -------------------------
            last_seen = now

            # reset unknown tracking
            unknown_consec = 0
            unknown_started_at = None
            unknown_feats = []
            unknown_last_cap = 0.0
            unknown_last_frame = None

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

            print(f"[INFO] ENTER: {present_name} {direction} (score={best_score:.2f}, second={second_score:.2f})", flush=True)
            if speak_enabled:
                tts_enqueue(tts_queue, f"{present_name} {direction}")

            consec_count = 0
            candidate_name = None

    except KeyboardInterrupt:
        print("\n[INFO] Stopping...", flush=True)

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
