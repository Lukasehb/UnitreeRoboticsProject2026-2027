import cv2
import numpy as np

import asyncio
import logging
import sys
import termios
import threading
import time
import tty
from queue import Queue
from unitree_webrtc_connect.webrtc_driver import UnitreeWebRTCConnection, WebRTCConnectionMethod
from unitree_webrtc_connect.constants import RTC_TOPIC, SPORT_CMD
from aiortc import MediaStreamTrack
from pathlib import Path

UNITREE_IP_ADDRESS = "192.168.12.1"


try:
    from ultralytics import YOLO
except ImportError as exc:
    raise SystemExit(
        "Missing dependency: ultralytics. Install it with: pip install ultralytics"
    ) from exc


# Enable logging for debugging
logging.basicConfig(level=logging.FATAL)


DETECTION_CONFIDENCE = 0.3
SCAN_HEIGHTS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
ALLOWED_PATH_LABELS = {"path", "path-oxod"}
TARGET_HEADING = 90.0
HEADING_DEADBAND = 2.0
FORWARD_SPEED = 0.2
TURN_SPEED = 0.3
COMMAND_INTERVAL_SECONDS = 0.25

model = YOLO(str(Path(__file__).resolve().parent.parent / "signaling" / "models" / "thuis.pt"), verbose=False)


def print_controls():
    print("""
================= CONTROLS =================
o : opstaan (segmentatie start)
l : neerliggen (segmentatie stopt)
spatie : pauze aan/uit (robot beweegt niet)
p : print controls
(ctrl + c) : exit
============================================
""", flush=True)


def get_allowed_mask_indices(result, model_names):
    if result.boxes is None or result.boxes.cls is None:
        return []

    allowed_indices = []
    class_ids = result.boxes.cls.cpu().numpy().astype(int).tolist()
    for index, class_id in enumerate(class_ids):
        label = str(model_names.get(class_id, "")).strip().lower()
        if label in ALLOWED_PATH_LABELS:
            allowed_indices.append(index)
    return allowed_indices


def compute_heading_to_point(frame, target_x, target_y):
    h, w = frame.shape[:2]
    start_x = w // 2
    start_y = h
    dx = target_x - start_x
    dy = start_y - target_y
    return float(np.degrees(np.arctan2(dy, dx)))


def compute_heading(frame, model):
    h, w = frame.shape[:2]
    result = model(frame, conf=DETECTION_CONFIDENCE, verbose=False)[0]
    model_names = getattr(model, "names", {})
    midpoints = []

    if result.masks is None or len(result.masks.data) == 0:
        return 90.0

    for mask_index in get_allowed_mask_indices(result, model_names):
        if mask_index >= len(result.masks.data):
            continue

        mask = result.masks.data[mask_index].cpu().numpy()
        mask = (mask * 255).astype(np.uint8)
        mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)

        for row_ratio in SCAN_HEIGHTS:
            y = int(h * row_ratio)
            if y >= h:
                continue
            filled_x = np.where(mask[y, :] > 0)[0]
            if len(filled_x) > 0:
                midpoints.append((int(np.mean(filled_x)), y))

    if not midpoints:
        return 90.0

    avg_x = int(np.mean([point[0] for point in midpoints]))
    target_y = min(point[1] for point in midpoints)
    return compute_heading_to_point(frame, avg_x, target_y)


async def send_move(conn, x=0, y=0, z=0):
    await conn.datachannel.pub_sub.publish_request_new(
        RTC_TOPIC["SPORT_MOD"],
        {
            "api_id": SPORT_CMD["Move"],
            "parameter": {"x": x, "y": y, "z": z},
        },
    )


async def send_stand_up(conn):
    await conn.datachannel.pub_sub.publish_request_new(
        RTC_TOPIC["SPORT_MOD"],
        {"api_id": SPORT_CMD["RecoveryStand"], "parameter": {"data": False}},
    )


async def send_stand_down(conn):
    await conn.datachannel.pub_sub.publish_request_new(
        RTC_TOPIC["SPORT_MOD"],
        {"api_id": SPORT_CMD["StandDown"]},
    )


def turn_speed_for_heading(heading):
    if heading == TARGET_HEADING:
        return 0.0

    error = heading - TARGET_HEADING
    if abs(error) <= HEADING_DEADBAND:
        return 0.0
    if heading < TARGET_HEADING:
        return -TURN_SPEED
    return TURN_SPEED


def forward_speed_for_heading(heading):
    if heading == TARGET_HEADING:
        return 0.0
    return FORWARD_SPEED


def report_move_result(future):
    try:
        future.result()
    except Exception as exc:
        print(f"Move command failed: {exc}", flush=True)


def get_key():
    """Read a single keypress from stdin (works over SSH)."""
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        ch = sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
    return ch


def keyboard_listener(conn, loop, robot_state, stop_event):
    """Blocking keyboard loop running in a separate thread."""
    while not stop_event.is_set():
        try:
            k = get_key().lower()
        except Exception:
            break

        # Ctrl+C
        if k == "\x03":
            print("\nExit requested", flush=True)
            stop_event.set()
            break

        # Opstaan -> segmentatie mag draaien
        elif k == "o":
            asyncio.run_coroutine_threadsafe(send_stand_up(conn), loop)
            robot_state["paused"] = False
            robot_state["standing"] = True
            robot_state["last_command"] = None
            print("Opstaan: segmentatie actief", flush=True)

        # Neerliggen -> segmentatie stopt
        elif k == "l":
            robot_state["standing"] = False
            robot_state["last_command"] = None
            asyncio.run_coroutine_threadsafe(send_move(conn), loop)
            asyncio.run_coroutine_threadsafe(send_stand_down(conn), loop)
            print("Neerliggen: segmentatie gestopt", flush=True)

        # Pauze aan/uit
        elif k == " ":
            robot_state["paused"] = not robot_state["paused"]
            robot_state["last_command"] = None
            if robot_state["paused"]:
                asyncio.run_coroutine_threadsafe(send_move(conn), loop)
                print("PAUZE: robot beweegt niet", flush=True)
            else:
                print("Pauze opgeheven", flush=True)

        # Print controls
        elif k == "p":
            print_controls()


def main():
    frame_queue = Queue()
    robot_state = {
        "standing": False,
        "paused": False,
        "last_sent_at": 0.0,
        "last_command": None,
    }
    stop_event = threading.Event()

    # Choose a connection method (uncomment the correct one)
    conn = UnitreeWebRTCConnection(WebRTCConnectionMethod.LocalSTA, ip=UNITREE_IP_ADDRESS)
    # conn = UnitreeWebRTCConnection(WebRTCConnectionMethod.LocalSTA, serialNumber="B42D2000XXXXXXXX")
    # conn = UnitreeWebRTCConnection(WebRTCConnectionMethod.Remote, serialNumber="B42D2000XXXXXXXX", username="email@gmail.com", password="pass")
    # conn = UnitreeWebRTCConnection(WebRTCConnectionMethod.LocalAP)

    # Async function to receive video frames and put them in the queue
    async def recv_camera_stream(track: MediaStreamTrack):
        while True:
            frame = await track.recv()
            # Convert the frame to a NumPy array
            img = frame.to_ndarray(format="bgr24")
            frame_queue.put(img)

    def run_asyncio_loop(loop):
        asyncio.set_event_loop(loop)
        async def setup():
            try:
                # Connect to the device
                await conn.connect()

                # Switch video channel on and start receiving video frames
                conn.video.switchVideoChannel(True)

                # Add callback to handle received video frames
                conn.video.add_track_callback(recv_camera_stream)
            except Exception as e:
                logging.error(f"Error in WebRTC connection: {e}")

        # Run the setup coroutine and then start the event loop
        loop.run_until_complete(setup())
        loop.run_forever()

    # Create a new event loop for the asyncio code
    loop = asyncio.new_event_loop()

    # Start the asyncio event loop in a separate thread
    asyncio_thread = threading.Thread(target=run_asyncio_loop, args=(loop,))
    asyncio_thread.start()

    print_controls()

    # Run keyboard listener in a background thread so it doesn't block the main loop
    kb_thread = threading.Thread(
        target=keyboard_listener,
        args=(conn, loop, robot_state, stop_event),
        daemon=True,
    )
    kb_thread.start()

    try:
        while not stop_event.is_set():
            if frame_queue.empty():
                # Sleep briefly to prevent high CPU usage
                time.sleep(0.01)
                continue

            img = frame_queue.get()

            # Segmentatie draait enkel wanneer de robot rechtstaat en niet gepauzeerd is
            if not robot_state["standing"] or robot_state["paused"]:
                continue

            heading = compute_heading(img, model)
            x_speed = forward_speed_for_heading(heading)
            z_speed = turn_speed_for_heading(heading)
            print(
                f"Heading: {heading:.2f} deg, "
                f"forward_x={x_speed:.2f}, turn_z={z_speed:.2f}"
            )

            # Toestand kan tijdens de (trage) segmentatie veranderd zijn
            if not robot_state["standing"] or robot_state["paused"]:
                continue

            now = time.monotonic()
            command_due = now - robot_state["last_sent_at"] >= COMMAND_INTERVAL_SECONDS
            command = (x_speed, z_speed)
            command_changed = command != robot_state["last_command"]
            if command_due or command_changed:
                future = asyncio.run_coroutine_threadsafe(
                    send_move(conn, x=x_speed, z=z_speed),
                    loop,
                )
                future.add_done_callback(report_move_result)
                robot_state["last_sent_at"] = now
                robot_state["last_command"] = command
    except KeyboardInterrupt:
        print("\nProgram interrupted", flush=True)
    finally:
        stop_event.set()
        try:
            asyncio.run_coroutine_threadsafe(send_move(conn), loop).result(timeout=2)
        except Exception as exc:
            print(f"Stop command failed: {exc}", flush=True)
        # Stop the asyncio event loop
        loop.call_soon_threadsafe(loop.stop)
        asyncio_thread.join()

if __name__ == "__main__":
    main()
