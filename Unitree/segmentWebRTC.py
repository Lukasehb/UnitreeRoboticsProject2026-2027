import cv2
import numpy as np

# Create an OpenCV window and display a blank image
height, width = 720, 1280  # Adjust the size as needed
img = np.zeros((height, width, 3), dtype=np.uint8)
cv2.imshow('Video', img)
cv2.waitKey(1)  # Ensure the window is created

import asyncio
import logging
import threading
import time
from queue import Queue
from unitree_webrtc_connect.webrtc_driver import UnitreeWebRTCConnection, WebRTCConnectionMethod
from aiortc import MediaStreamTrack
from pathlib import Path

try:
    from ultralytics import YOLO
except ImportError as exc:
    raise SystemExit(
        "Missing dependency: ultralytics. Install it with: pip install ultralytics"
    ) from exc


# Enable logging for debugging
logging.basicConfig(level=logging.FATAL)


DETECTION_CONFIDENCE = 0.8
SCAN_HEIGHTS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
ALLOWED_PATH_LABELS = {"path", "path-oxod"}

model = YOLO(str(Path(__file__).resolve().parent.parent / "signaling" / "models" / "thuis.pt"), verbose=False)


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

def main():
    frame_queue = Queue()

    # Choose a connection method (uncomment the correct one)
    conn = UnitreeWebRTCConnection(WebRTCConnectionMethod.LocalSTA, ip="unitree.local")
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

    try:
        while True:
            if not frame_queue.empty():
                img = frame_queue.get()
                heading = compute_heading(img, model)
                if (heading != 90.0):
                    print(f"Heading: {heading:.2f} deg")

                cv2.imshow('Video', img)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            else:
                # Sleep briefly to prevent high CPU usage
                time.sleep(0.01)
    finally:
        cv2.destroyAllWindows()
        # Stop the asyncio event loop
        loop.call_soon_threadsafe(loop.stop)
        asyncio_thread.join()

if __name__ == "__main__":
    main()