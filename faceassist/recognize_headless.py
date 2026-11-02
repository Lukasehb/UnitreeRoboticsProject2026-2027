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

YUNET_URL = "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx"
SFACE_URL = "https://github.com/opencv/opencv_zoo/raw/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx"


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


# -----------------------------
# Async TTS (Piper) process
# -----------------------------

def piper_say(text: str, model_path: str, sample_rate: int = 22050):
    """
    Equivalent to:
      echo "text" | piper --model MODEL --output_raw | aplay -r 22050 -f S16_LE -t raw -
    """
    p1 = subprocess.Popen(
        ["piper", "--model", model_path, "--output_raw", "--length_scale", "0.6"],
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
        p1.stdin.write((text + "\n").encode("utf-8"))
        p1.stdin.close()
    except Exception:
        pass

    if p1.stdout is not None:
        p1.stdout.close()

    try:
        p2.wait(timeout=30)
    except Exception:
        pass
    try:
        p1.wait(timeout=30)
    except Exception:
        pass


def tts_worker_loop(tts_queue: mp.Queue, stop_event: mp.Event, args):
    signal.signal(signal.SIGINT, signal.SIG_IGN)

    try:
        subprocess.run(["piper", "--help"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
    except FileNotFoundError:
        print("[WARN] 'piper' not found in PATH.", flush=True)
        return

    if not os.path.exists(args.piper_model):
        print(f"[WARN] Piper model not found: {args.piper_model}", flush=True)
        return

    while not stop_event.is_set():
        try:
            item = tts_queue.get(timeout=0.1)
        except pyqueue.Empty:
            continue

        if item is None:
            break

        text = str(item).strip()
        if not text:
            continue

        try:
            piper_say(text=text, model_path=args.piper_model, sample_rate=args.piper_rate)
        except Exception:
            pass


def tts_enqueue(tts_queue: mp.Queue, text: str):
    if tts_queue is None:
        return
    try:
        tts_queue.put_nowait(text)
    except pyqueue.Full:
        pass


def str2bool(s: str) -> bool:
    return str(s).strip().lower() in ("1", "true", "yes", "y", "on")


# -----------------------------
# Vision worker process (HEADLESS)
# -----------------------------

def worker_loop(args, stop_event: mp.Event, tts_queue: mp.Queue):
    signal.signal(signal.SIGINT, signal.SIG_IGN)

    yunet_path = os.path.join("models", "face_detection_yunet_2023mar.onnx")
    sface_path = os.path.join("models", "face_recognition_sface_2021dec.onnx")

    known = load_known(args.known)
    if not known:
        print(f"[ERROR] No known identities in '{args.known}'. Run enroll script first.", flush=True)
        return
    print("[INFO] Known:", ", ".join(sorted(known.keys())), flush=True)

    cap = open_camera_linux(args.cam, args.width, args.height, args.fps)
    if not cap.isOpened():
        print("[ERROR] Cannot open camera.", flush=True)
        return

    ok, frame = cap.read()
    if not ok or frame is None:
        print("[ERROR] Cannot read first frame.", flush=True)
        cap.release()
        return

    h, w = frame.shape[:2]
    detector = cv2.FaceDetectorYN.create(yunet_path, "", (w, h), args.score_th, args.nms_th, args.topk)
    recognizer = cv2.FaceRecognizerSF.create(sface_path, "")

    speak_enabled = (not args.no_tts) and str2bool(args.speak)

    if speak_enabled:
        tts_enqueue(tts_queue, f"Face recognition started. {len(known)} identities loaded.")

    # --- Announce-on-entry state ---
    # We only speak when someone "enters": transition absent -> present.
    present = False
    present_name = None
    present_direction = None
    present_since = 0.0
    last_seen = 0.0

    # To avoid announcing on a single noisy frame: require N consecutive confident frames
    consec_needed = args.enter_confirm_frames
    consec_count = 0
    candidate_name = None
    candidate_direction = None

    # Re-announce only if the person has been absent long enough
    last_announced_at = {}  # name -> time

    frame_id = 0

    try:
        while not stop_event.is_set():
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

            # If no face, handle "leave" timeout
            if face is None:
                if present and (now - last_seen) >= args.lost_timeout:
                    print(f"[INFO] {present_name} left the frame.", flush=True)
                    present = False
                    present_name = None
                    present_direction = None
                    consec_count = 0
                    candidate_name = None
                    candidate_direction = None
                continue

            x, y, fw, fh = face[:4].astype(int)
            if fw < args.min_face:
                # small face: treat like "not confidently present"
                if present and (now - last_seen) >= args.lost_timeout:
                    print(f"[INFO] {present_name} left the frame.", flush=True)
                    present = False
                    present_name = None
                    present_direction = None
                consec_count = 0
                candidate_name = None
                candidate_direction = None
                continue

            direction = face_direction(x, fw, w)

            aligned = recognizer.alignCrop(frame, face)
            feat = recognizer.feature(aligned).astype(np.float32)

            best_name, best_score, second_score = best_match(recognizer, feat, known)
            confident = (best_score >= args.threshold) and ((best_score - second_score) >= args.margin)

            if not confident or best_name is None:
                # Not confident: decay candidate/present state using lost_timeout
                if present and (now - last_seen) >= args.lost_timeout:
                    print(f"[INFO] {present_name} left the frame.", flush=True)
                    present = False
                    present_name = None
                    present_direction = None
                consec_count = 0
                candidate_name = None
                candidate_direction = None
                continue

            # Confident detection
            last_seen = now

            # If already present with the same person, do nothing (no repeated speech)
            if present and best_name == present_name:
                # Optionally keep latest direction for debug, but don't speak
                present_direction = direction
                continue

            # Otherwise: build up candidate confirmation
            if candidate_name == best_name:
                consec_count += 1
            else:
                candidate_name = best_name
                candidate_direction = direction
                consec_count = 1

            if consec_count < consec_needed:
                continue

            # Candidate confirmed as "entered"
            # Check re-announce rule (must be absent for reannounce_after)
            last_spoke = last_announced_at.get(candidate_name, 0.0)
            if (now - last_spoke) < args.reannounce_after:
                # Consider as present but don't speak again so soon
                present = True
                present_name = candidate_name
                present_direction = candidate_direction
                present_since = now
                print(f"[INFO] {candidate_name} detected again but not re-announced (reannounce_after={args.reannounce_after}s).", flush=True)
                continue

            present = True
            present_name = candidate_name
            present_direction = candidate_direction
            present_since = now
            last_announced_at[present_name] = now

            print(f"[INFO] ENTER: {present_name} {present_direction} (score={best_score:.2f}, second={second_score:.2f})", flush=True)
            if speak_enabled:
                tts_enqueue(tts_queue, f"{present_name} {present_direction}")

            # Reset candidate counter so we don't immediately re-trigger
            consec_count = 0
            candidate_name = None
            candidate_direction = None

    finally:
        cap.release()


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--speak", type=str, default="True")
    ap.add_argument("--cam", type=int, default=0)
    ap.add_argument("--known", type=str, default="known")

    ap.add_argument("--min_face", type=int, default=50)
    ap.add_argument("--threshold", type=float, default=0.50)
    ap.add_argument("--margin", type=float, default=0.06)

    # Only announce when someone enters the frame
    ap.add_argument("--lost_timeout", type=float, default=1.0,
                    help="Seconds without a confident face before considering the person 'gone'.")
    ap.add_argument("--enter_confirm_frames", type=int, default=3,
                    help="How many consecutive confident frames are needed to confirm entry.")
    ap.add_argument("--reannounce_after", type=float, default=6.0,
                    help="Minimum seconds before the same person can be announced again (after leaving).")

    ap.add_argument("--score_th", type=float, default=0.9)
    ap.add_argument("--nms_th", type=float, default=0.3)
    ap.add_argument("--topk", type=int, default=5000)
    ap.add_argument("--infer_every", type=int, default=2)

    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--fps", type=int, default=15)

    # Piper TTS
    ap.add_argument("--no_tts", action="store_true")
    ap.add_argument("--piper_model", type=str,
                    default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "voices", "en_GB-alan-medium.onnx"))
    ap.add_argument("--piper_rate", type=int, default=22050,
                    help="aplay sample rate (usually 22050 for piper voices)")
    ap.add_argument("--tts_queue_size", type=int, default=20)

    args = ap.parse_args()

    yunet_path = os.path.join("models", "face_detection_yunet_2023mar.onnx")
    sface_path = os.path.join("models", "face_recognition_sface_2021dec.onnx")
    download_if_missing(YUNET_URL, yunet_path)
    download_if_missing(SFACE_URL, sface_path)

    stop_event = mp.Event()

    if str2bool(args.speak) and (not args.no_tts):
        print("[INFO] Voice output enabled.", flush=True)
    else:
        print("[INFO] Voice output disabled.", flush=True)

    tts_queue = None
    tts_proc = None
    if not args.no_tts:
        tts_queue = mp.Queue(maxsize=args.tts_queue_size)
        tts_proc = mp.Process(target=tts_worker_loop, args=(tts_queue, stop_event, args), daemon=True)
        tts_proc.start()

    vision_proc = mp.Process(target=worker_loop, args=(args, stop_event, tts_queue), daemon=True)
    vision_proc.start()

    print("[INFO] Headless running. Ctrl+C om te stoppen.", flush=True)
    try:
        while vision_proc.is_alive():
            vision_proc.join(timeout=0.2)
    except KeyboardInterrupt:
        print("\n[INFO] Ctrl+C received, stopping...", flush=True)
        stop_event.set()
        if tts_queue is not None:
            try:
                tts_queue.put_nowait(None)
            except Exception:
                pass

        vision_proc.join(timeout=1.0)
        if vision_proc.is_alive():
            vision_proc.terminate()
            vision_proc.join()

        if tts_proc is not None:
            tts_proc.join(timeout=1.0)
            if tts_proc.is_alive():
                tts_proc.terminate()
                tts_proc.join()

    print("[INFO] Exited.", flush=True)


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
