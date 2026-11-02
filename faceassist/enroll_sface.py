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


YUNET_URL = "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx"
SFACE_URL = "https://github.com/opencv/opencv_zoo/raw/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx"


def download_if_missing(url: str, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if os.path.exists(path):
        return
    print(f"[INFO] Downloading {os.path.basename(path)} ...", flush=True)
    urllib.request.urlretrieve(url, path)
    print(f"[OK] Saved to {path}", flush=True)


def largest_face(faces: np.ndarray):
    if faces is None or len(faces) == 0:
        return None
    areas = faces[:, 2] * faces[:, 3]
    return faces[int(np.argmax(areas))]


# -----------------------------
# Async TTS Process (Piper)
# -----------------------------

def read_piper_sample_rate(model_path: str, default_rate: int = 22050) -> int:
    """
    Try to read sample_rate from the accompanying .onnx.json.
    If not found, return default_rate.
    """
    json_path = model_path + ".json"
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        # Common keys in piper voice json
        for key in ("sample_rate", "audio.sample_rate", "audio_sample_rate"):
            if key in data and isinstance(data[key], int):
                return int(data[key])
        # Sometimes nested
        if isinstance(data.get("audio"), dict) and isinstance(data["audio"].get("sample_rate"), int):
            return int(data["audio"]["sample_rate"])
    except Exception:
        pass
    return default_rate


def piper_say(text: str, model_path: str, sample_rate: int):
    """
    Equivalent to:
      echo "text" | piper --model MODEL --output_raw | aplay -r <rate> -f S16_LE -t raw -
    Uses Popen to avoid shell quoting issues.
    """
    p1 = subprocess.Popen(
        ["piper", "--model", model_path, "--output_raw"],
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

    # Wait (avoid hanging forever)
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

    # Auto sample rate from json unless explicitly forced
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
            piper_say(text, model_path=model_path, sample_rate=sample_rate)
        except Exception:
            pass


def tts_enqueue(tts_queue, text):
    if tts_queue is None:
        return
    try:
        tts_queue.put_nowait(text)
    except pyqueue.Full:
        pass


def sanitize_name(name: str) -> str:
    name = name.strip().replace("/", "_").replace("\\", "_")
    name = name.replace("..", ".")
    return name


def ask_input(prompt: str) -> str:
    # ensure prompt shows immediately
    sys.stdout.write(prompt)
    sys.stdout.flush()
    return input()


# -----------------------------
# Main
# -----------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cam", type=int, default=0)
    ap.add_argument("--samples", type=int, default=20)
    ap.add_argument("--outdir", type=str, default="known")
    ap.add_argument("--min_face", type=int, default=80)
    ap.add_argument("--capture_interval", type=float, default=0.5)

    # Piper options
    ap.add_argument("--piper_model", type=str, default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "voices", "en_GB-alan-medium.onnx"))
    ap.add_argument("--piper_rate", type=int, default=22050, help="Fallback aplay sample rate")
    ap.add_argument("--piper_rate_auto", action="store_true", help="Read sample_rate from <model>.onnx.json")
    ap.add_argument("--tts_queue_size", type=int, default=20)

    args = ap.parse_args()

    yunet_path = os.path.join("models", "face_detection_yunet_2023mar.onnx")
    sface_path = os.path.join("models", "face_recognition_sface_2021dec.onnx")
    download_if_missing(YUNET_URL, yunet_path)
    download_if_missing(SFACE_URL, sface_path)
    os.makedirs(args.outdir, exist_ok=True)

    # Start TTS
    stop_event = mp.Event()
    tts_queue = mp.Queue(maxsize=args.tts_queue_size)
    tts_proc = mp.Process(target=tts_worker_loop, args=(tts_queue, stop_event, args), daemon=True)
    tts_proc.start()

    # Camera
    cap = cv2.VideoCapture(args.cam)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 15)

    ok, frame = cap.read()
    if not ok or frame is None:
        print("[ERROR] Camera not working.", flush=True)
        return

    h, w = frame.shape[:2]
    detector = cv2.FaceDetectorYN.create(yunet_path, "", (w, h))
    recognizer = cv2.FaceRecognizerSF.create(sface_path, "")

    tts_enqueue(tts_queue, "System started. I am waiting for a face. Press Control C to stop.")
    print("[INFO] Running. Ctrl+C to stop.", flush=True)

    try:
        while True:
            # 1) Wait for a sufficiently large face
            face = None
            while True:
                ok, frame = cap.read()
                if not ok or frame is None:
                    face = None
                    break

                detector.setInputSize((w, h))
                _, faces = detector.detect(frame)
                face = largest_face(faces)

                if face is not None:
                    x, y, fw, fh = face[:4].astype(int)
                    if fw >= args.min_face:
                        break  # face good enough

                time.sleep(0.05)

            if face is None:
                continue

            # 2) Ask for the person's name (terminal input)
            tts_enqueue(tts_queue, "Face detected.")
            tts_enqueue(tts_queue, "Please type the person's name in the terminal and press Enter.")

            name = sanitize_name(ask_input("Name: "))

            if not name:
                tts_enqueue(tts_queue, "No name entered. I will wait for a face again.")
                print("[INFO] No name. Back to waiting.\n", flush=True)
                time.sleep(0.3)
                continue

            tts_enqueue(tts_queue, f"Okay {name}. I will now record samples.")
            print(f"[INFO] Capturing samples for: {name}", flush=True)

            # 3) Capture samples
            features = []
            last_capture = 0.0

            while len(features) < args.samples:
                ok, frame = cap.read()
                if not ok or frame is None:
                    break

                detector.setInputSize((w, h))
                _, faces = detector.detect(frame)
                face = largest_face(faces)

                if face is None:
                    tts_enqueue(tts_queue, "I lost the face. Please look at the camera.")
                    time.sleep(0.4)
                    continue

                x, y, fw, fh = face[:4].astype(int)
                if fw < args.min_face:
                    tts_enqueue(tts_queue, "Please move a bit closer to the camera.")
                    time.sleep(0.6)
                    continue

                now = time.time()
                if now - last_capture >= args.capture_interval:
                    aligned = recognizer.alignCrop(frame, face)
                    feat = recognizer.feature(aligned).astype(np.float32)
                    features.append(feat)
                    last_capture = now

                    if len(features) % 5 == 0:
                        tts_enqueue(tts_queue, f"{len(features)} samples recorded.")

            tts_enqueue(tts_queue, "Recording complete.")
            print("[INFO] Capture done.", flush=True)

            # 4) Ask to save (terminal input)
            tts_enqueue(tts_queue, "May I save this person? Type y or n and press Enter.")

            ans = ask_input("Save? (y/n): ").strip().lower()

            if ans.startswith("y") and len(features) >= 8:
                out_path = os.path.join(args.outdir, f"{name}.npz")
                np.savez_compressed(out_path, features=np.stack(features, axis=0))
                tts_enqueue(tts_queue, f"Saved. {name} has been added.")
                print("[OK] Saved:", out_path, flush=True)
                return
            else:
                tts_enqueue(tts_queue, "Okay. I will not save anything.")
                print("[INFO] Not saved.", flush=True)
                return

    except KeyboardInterrupt:
        print("\n[INFO] Stopping...", flush=True)

    finally:
        cap.release()
        stop_event.set()
        try:
            tts_queue.put_nowait(None)
        except Exception:
            pass
        tts_proc.join(timeout=1.0)
        if tts_proc.is_alive():
            tts_proc.terminate()
            tts_proc.join()


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
