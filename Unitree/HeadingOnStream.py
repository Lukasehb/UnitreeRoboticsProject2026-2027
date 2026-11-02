import cv2
import numpy as np

import argparse
import asyncio
import logging
import os
import threading
import time
from queue import Empty, Queue
from unitree_webrtc_connect.webrtc_driver import UnitreeWebRTCConnection, WebRTCConnectionMethod
from unitree_webrtc_connect.constants import RTC_TOPIC, SPORT_CMD
from aiortc import MediaStreamTrack
from ultralytics import YOLO
from HeadingOnVideo import (
    DEFAULT_BOTTOM_CONNECT_RATIO,
    DEFAULT_BOTTOM_SEED_X_RATIO,
    DEFAULT_LATERAL_STEP_RATIO,
    DEFAULT_ROW_WEIGHT_POWER,
    calculate_heading,
    draw_row_transition_arrows,
    draw_rowwise_midpoints,
    draw_start_to_first_midpoint_arrow,
    extract_rowwise_midpoints,
    first_near_midpoint,
    segmentation_mask_from_result,
)

MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "signaling", "models", "denham.pt")
DETECTION_CONFIDENCE = 0.3
SCAN_HEIGHTS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
PREDICTION_COLOR = (0, 255, 255)
PREDICTION_ALPHA = 0.35
ROW_WEIGHT_POWER = DEFAULT_ROW_WEIGHT_POWER
LATERAL_STEP_RATIO = DEFAULT_LATERAL_STEP_RATIO
BOTTOM_CONNECT_RATIO = DEFAULT_BOTTOM_CONNECT_RATIO
BOTTOM_SEED_X_RATIO = DEFAULT_BOTTOM_SEED_X_RATIO

model = YOLO(MODEL_PATH, verbose=False)


# Enable logging for debugging
logging.basicConfig(level=logging.FATAL)

ROBOT_IP = "unitree.local"
ARUCO_DICTIONARY = "DICT_4X4_50"
RECOVERY_DELAY_SECONDS = 3.0
LIGHT_TOGGLE_MARKER_ID = 26
LIGHT_BRIGHTNESS = 1
FRAME_WIDTH = 640
MAX_PROCESSING_FPS = 10
MIN_PROCESS_INTERVAL_SECONDS = 1.0 / MAX_PROCESSING_FPS

def parse_args():
    parser = argparse.ArgumentParser(
        description="Control the Unitree with ArUco markers from the WebRTC video stream."
    )
    parser.add_argument(
        "--preview",
        action="store_true",
        help="Show the OpenCV video window. Without this, the script runs terminal-only.",
    )
    return parser.parse_args()

ARUCO_ACTIONS = {
    22: ("h / Hello", {"api_id": SPORT_CMD["Hello"]}, True),
    23: ("x / Stretch", {"api_id": SPORT_CMD["Stretch"], "parameter": {"data": False}}, True),
    24: ("y / Sit", {"api_id": 1009, "parameter": {"data": False}}, False),
    25: ("t / Rize sit", {"api_id": 1010, "parameter": {"data": False}}, True),
    26: ("toggle light", None, False),
    27: ("g / Front Jump", {"api_id": 1031, "parameter": {"data": False}}, True),
}

def resize_to_width(img, width):
    height, current_width = img.shape[:2]
    if current_width == width:
        return img

    scale = width / current_width
    new_height = int(height * scale)
    return cv2.resize(img, (width, new_height), interpolation=cv2.INTER_AREA)

def drain_frame_queue(frame_queue):
    latest_img = None
    while True:
        try:
            latest_img = frame_queue.get_nowait()
        except Empty:
            return latest_img

def draw_inference_time(img, total_ms):
    cv2.putText(
        img,
        f"Inference: {total_ms:.1f} ms",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 255),
        2,
        cv2.LINE_AA,
    )


def draw_heading_info(img, heading_angle, midpoint_count):
    heading_text = "Heading: n/a"
    if heading_angle is not None:
        heading_text = f"Heading: {heading_angle:.1f} deg"

    cv2.putText(
        img,
        f"{heading_text} | Midpoints: {midpoint_count}",
        (10, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )


def draw_prediction_polygons(img, result):
    masks = getattr(result, "masks", None)
    if masks is None or masks.xy is None:
        return 0

    height, width = img.shape[:2]
    overlay = img.copy()
    polygons = []

    for polygon in masks.xy:
        if polygon is None or len(polygon) < 3:
            continue

        points = np.round(polygon).astype(np.int32).reshape((-1, 1, 2))
        points[:, 0, 0] = np.clip(points[:, 0, 0], 0, width - 1)
        points[:, 0, 1] = np.clip(points[:, 0, 1], 0, height - 1)
        polygons.append(points)

    if not polygons:
        return 0

    cv2.fillPoly(overlay, polygons, PREDICTION_COLOR)
    cv2.addWeighted(overlay, PREDICTION_ALPHA, img, 1.0 - PREDICTION_ALPHA, 0, dst=img)

    for points in polygons:
        cv2.polylines(img, [points], True, PREDICTION_COLOR, 2, cv2.LINE_AA)

    return len(polygons)


def draw_heading_arrows(img, result):
    mask = segmentation_mask_from_result(
        result,
        img.shape[:2],
        BOTTOM_CONNECT_RATIO,
        BOTTOM_SEED_X_RATIO,
    )
    midpoints = extract_rowwise_midpoints(mask, SCAN_HEIGHTS) if mask is not None else []
    heading_angle, _weighted_target, arrow_start = calculate_heading(
        midpoints,
        img.shape[:2],
        ROW_WEIGHT_POWER,
    )
    first_midpoint = first_near_midpoint(midpoints)

    draw_row_transition_arrows(img, midpoints, LATERAL_STEP_RATIO)
    draw_rowwise_midpoints(img, midpoints)
    draw_start_to_first_midpoint_arrow(img, arrow_start, first_midpoint)
    draw_heading_info(img, heading_angle, len(midpoints))

    return heading_angle, len(midpoints)


def create_aruco_detector(dictionary_name):
    if not hasattr(cv2, "aruco"):
        raise RuntimeError(
            "This OpenCV install has no cv2.aruco module. Install opencv-contrib-python."
        )

    dictionary_id = getattr(cv2.aruco, dictionary_name, None)
    if dictionary_id is None:
        raise RuntimeError(f"Unknown ArUco dictionary: {dictionary_name}")

    if hasattr(cv2.aruco, "getPredefinedDictionary"):
        dictionary = cv2.aruco.getPredefinedDictionary(dictionary_id)
    else:
        dictionary = cv2.aruco.Dictionary_get(dictionary_id)

    if hasattr(cv2.aruco, "DetectorParameters"):
        parameters = cv2.aruco.DetectorParameters()
    else:
        parameters = cv2.aruco.DetectorParameters_create()

    if hasattr(cv2.aruco, "ArucoDetector"):
        detector = cv2.aruco.ArucoDetector(dictionary, parameters)
        return lambda frame: detector.detectMarkers(frame)

    return lambda frame: cv2.aruco.detectMarkers(
        frame,
        dictionary,
        parameters=parameters,
    )

def detect_and_draw_aruco(img, detect_markers):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    corners, ids, _rejected = detect_markers(gray)

    if ids is None:
        return [], {}

    cv2.aruco.drawDetectedMarkers(img, corners, ids)
    marker_ids = [int(marker_id[0]) for marker_id in ids]
    marker_info = {}

    for marker_id, marker_corners in zip(marker_ids, corners):
        points = marker_corners.reshape((4, 2)).astype(int)
        center_x = int(points[:, 0].mean())
        center_y = int(points[:, 1].mean())
        side_lengths = [
            np.linalg.norm(points[0] - points[1]),
            np.linalg.norm(points[1] - points[2]),
            np.linalg.norm(points[2] - points[3]),
            np.linalg.norm(points[3] - points[0]),
        ]
        marker_info[marker_id] = {
            "center_x": center_x,
            "center_y": center_y,
            "side_px": float(np.mean(side_lengths)),
        }
        cv2.putText(
            img,
            f"ID {marker_id}",
            (center_x - 30, center_y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )
        if marker_id in ARUCO_ACTIONS:
            action_name, _payload, _needs_recovery = ARUCO_ACTIONS[marker_id]
            cv2.putText(
                img,
                action_name,
                (center_x - 55, center_y + 22),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )
    return marker_ids, marker_info

async def send_light(conn, brightness):
    await conn.datachannel.pub_sub.publish_request_new(
        RTC_TOPIC["VUI"],
        {
            "api_id": 1005,
            "parameter": {"brightness": brightness},
        },
    )

async def send_action(conn, payload, needs_recovery):
    action_error = None

    try:
        await conn.datachannel.pub_sub.publish_request_new(
            RTC_TOPIC["SPORT_MOD"],
            payload,
        )
    except Exception as exc:
        action_error = exc

    if needs_recovery:
        await asyncio.sleep(RECOVERY_DELAY_SECONDS)

        recovery_payload = {
            "api_id": SPORT_CMD["RecoveryStand"],
            "parameter": {"data": False},
        }
        await conn.datachannel.pub_sub.publish_request_new(
            RTC_TOPIC["SPORT_MOD"],
            recovery_payload,
        )

    if action_error is not None:
        raise action_error

def report_action_result(future, marker_id, action_name, needs_recovery, action_state):
    try:
        future.result()
        recovery_text = " + RecoveryStand" if needs_recovery else ""
        print(f"Action completed: marker {marker_id} ({action_name}){recovery_text}", flush=True)
    except Exception as exc:
        print(
            f"Action failed for marker {marker_id} ({action_name}): {exc}",
            flush=True,
        )
    finally:
        action_state["in_progress"] = False

def report_light_result(future, brightness, light_state, action_state):
    try:
        future.result()
        light_state["on"] = brightness > 0
        status = "on" if light_state["on"] else "off"
        print(f"Light toggled {status}", flush=True)
    except Exception as exc:
        print(f"Light toggle failed: {exc}", flush=True)
    finally:
        action_state["in_progress"] = False

def main():
    args = parse_args()
    frame_queue = Queue(maxsize=1)
    detect_markers = create_aruco_detector(ARUCO_DICTIONARY)
    last_printed_ids = None
    executed_marker_ids = set()
    action_state = {"in_progress": False}
    light_state = {"on": False}
    previous_marker_ids = set()

    # Choose a connection method (uncomment the correct one)
    conn = UnitreeWebRTCConnection(WebRTCConnectionMethod.LocalSTA, ip="unitree.local")
    # conn = UnitreeWebRTCConnection(WebRTCConnectionMethod.LocalSTA, serialNumber="B42D2000XXXXXXXX")
    # conn = UnitreeWebRTCConnection(WebRTCConnectionMethod.Remote, serialNumber="B42D2000XXXXXXXX", username="email@gmail.com", password="pass")
    # conn = UnitreeWebRTCConnection(WebRTCConnectionMethod.LocalAP)

    # Async function to receive video frames and put them in the queue
    async def recv_camera_stream(track: MediaStreamTrack):
        while True:
            frame = await track.recv()
            # Convert the frame to a NumPy array and keep only the newest frame.
            img = frame.to_ndarray(format="bgr24")
            img = resize_to_width(img, FRAME_WIDTH)

            if frame_queue.full():
                try:
                    frame_queue.get_nowait()
                except Empty:
                    pass
            frame_queue.put_nowait(img)

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


    predict_kwargs = {
        "conf": 0.25,
        "imgsz": 640,
        "verbose": False,
    }
    last_processed_at = 0.0


    try:
        while True:
            now = time.perf_counter()
            if now - last_processed_at < MIN_PROCESS_INTERVAL_SECONDS:
                drain_frame_queue(frame_queue)
                time.sleep(0.01)
                continue

            img = drain_frame_queue(frame_queue)
            if img is not None:
                last_processed_at = time.perf_counter()

                start = time.perf_counter()
                result = model.predict(img, **predict_kwargs)[0]
                total_ms = (time.perf_counter() - start) * 1000.0
                polygon_count = draw_prediction_polygons(img, result)
                heading_angle, midpoint_count = draw_heading_arrows(img, result)
                draw_inference_time(img, total_ms)



                marker_ids, marker_info = detect_and_draw_aruco(img, detect_markers)
                if marker_ids != last_printed_ids:
                    #print(f"ArUco markers: {marker_ids}", flush=True)
                    last_printed_ids = marker_ids

                current_marker_ids = set(marker_ids)
                if (
                    not action_state["in_progress"]
                    and LIGHT_TOGGLE_MARKER_ID in current_marker_ids
                    and LIGHT_TOGGLE_MARKER_ID not in previous_marker_ids
                ):
                    action_state["in_progress"] = True
                    brightness = 0 if light_state["on"] else LIGHT_BRIGHTNESS
                    future = asyncio.run_coroutine_threadsafe(
                        send_light(conn, brightness),
                        loop,
                    )
                    future.add_done_callback(
                        lambda done, seen_brightness=brightness: report_light_result(
                            done,
                            seen_brightness,
                            light_state,
                            action_state,
                        )
                    )
                    print("Action triggered: marker 26 -> toggle light", flush=True)

                if not action_state["in_progress"]:
                    marker_id = next(
                        (
                            current_id for current_id in sorted(marker_ids)
                            if current_id in ARUCO_ACTIONS
                            and current_id != LIGHT_TOGGLE_MARKER_ID
                            and current_id not in executed_marker_ids
                        ),
                        None,
                    )
                    if marker_id is not None:
                        executed_marker_ids.add(marker_id)
                        action_state["in_progress"] = True
                        action_name, payload, needs_recovery = ARUCO_ACTIONS[marker_id]
                        future = asyncio.run_coroutine_threadsafe(
                            send_action(conn, payload.copy(), needs_recovery),
                            loop,
                        )
                        future.add_done_callback(
                            lambda done, seen_id=marker_id, seen_action=action_name, seen_recovery=needs_recovery: report_action_result(
                                done,
                                seen_id,
                                seen_action,
                                seen_recovery,
                                action_state,
                            )
                        )
                        print(
                            f"Action triggered once: marker {marker_id} -> {action_name}",
                            flush=True,
                        )

                previous_marker_ids = current_marker_ids
                if args.preview:
                    cv2.imshow('Video', img)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
            else:
                # Sleep briefly to prevent high CPU usage
                time.sleep(0.01)
    finally:
        if args.preview:
            cv2.destroyAllWindows()
        # Stop the asyncio event loop
        loop.call_soon_threadsafe(loop.stop)
        asyncio_thread.join()

if __name__ == "__main__":
    main()
