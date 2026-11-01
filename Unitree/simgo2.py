"""
De Botopia-simulator (robotics.dinno.be) in plaats van de echte Go2.

urlcontrol.py praat met de Go2 via de WebRTC-datachannel: {topic, api_id,
parameter}. De simulator neemt exact dezelfde berichten aan als actie
{"webrtc": {...}}, dus hier vervangen we enkel de verbinding:

  * SimConnection.datachannel.pub_sub.publish_request_new() stuurt het bericht
    via RemoteSession.command() naar de simulator;
  * een thread haalt het camerabeeld van de robot op (RemoteSession.frame)
    en geeft het aan de segmentatie;
  * een thread zet de positie van de robot (pose in meter) om naar een
    nep-GPS rond SIM_ORIGIN en stuurt die naar stayontrails (/api/gps), net
    zoals de gsm dat met zijn echte GPS doet.

Open de simulator in een browservenster met ?remote=CODE en hou dat venster
zichtbaar (een verborgen tab pauzeert). Start dan op de Jetson:

    ROBOT_SIM=CODE python3 urlcontrol.py

Een testroute opnemen: rij de robot met de hand rond in de simulator en laat
dit script om de paar meter een routepunt bewaren in stayontrails/paths/:

    python3 simgo2.py record CODE "Sim park test"

botopia.py komt uit de robotSimulatorAPI-repository. We zoeken hem in
BOTOPIA_DIR, naast dit script, en in ../../robotSimulatorAPI/python (de
repository naast deze gekloond).
"""

import asyncio
import json
import math
import os
import random
import re
import ssl
import sys
import threading
import time
import types
import urllib.request
from datetime import datetime, timezone

HERE = os.path.dirname(os.path.abspath(__file__))

SIM_API_URL = os.environ.get("SIM_API_URL", "https://robotics.dinno.be/api/index.php")
BOTOPIA_DIR = os.environ.get("BOTOPIA_DIR", "")

# Camera: 1..30 fps, 160..640 breed (4:3). Een camera heeft hooguit 2
# ontvangers tegelijk, dus naast dit script kan er nog één browser meekijken.
SIM_CAMERA_FPS = 10
SIM_CAMERA_WIDTH = 640
SIM_CAMERA_RETRY = 0.5        # s wachten na "no picture yet" (404) of een fout

# Nep-GPS: de simulator geeft een pose in meter (x langs de lanes, y omhoog,
# z opzij; yaw positief = naar links). We leggen de oorsprong van de
# simulatorwereld op SIM_ORIGIN en rekenen x als oost en -z als noord: links
# van x is dan noord, dus de kaart staat niet gespiegeld en links blijft links.
SIM_ORIGIN_LAT = float(os.environ.get("SIM_ORIGIN_LAT", "50.841746"))
SIM_ORIGIN_LON = float(os.environ.get("SIM_ORIGIN_LON", "4.322936"))
SIM_GPS_INTERVAL = 1.0        # s tussen twee GPS-fixen, zoals de gsm
SIM_GPS_NOISE_M = float(os.environ.get("SIM_GPS_NOISE_M", "0"))  # ruis in meter, 0 = perfect
SIM_GPS_ACCURACY_M = 1.0      # gemelde nauwkeurigheid zonder ruis

# Route opnemen
RECORD_SPACING_M = 3.0        # zoveel meter tussen twee routepunten
RECORD_INTERVAL = 0.5         # s tussen twee uitlezingen van de pose
PATHS_DIR = os.path.join(os.path.dirname(HERE), "stayontrails", "paths")

METERS_PER_DEG_LAT = 111320.0


def load_remote_session_class():
    """RemoteSession uit botopia.py (BOTOPIA_DIR, naast dit script, of de
    robotSimulatorAPI-repository naast deze)."""
    sibling = os.path.join(os.path.dirname(os.path.dirname(HERE)), "robotSimulatorAPI", "python")
    for folder in (sibling, HERE, BOTOPIA_DIR):
        if folder and folder not in sys.path:
            sys.path.insert(0, folder)
    try:
        from botopia import RemoteSession
    except ImportError as exc:
        raise RuntimeError(
            "botopia.py niet gevonden: kopieer python/botopia.py uit robotSimulatorAPI "
            "naast simgo2.py of zet BOTOPIA_DIR") from exc
    return RemoteSession


def pose_to_latlon(pose):
    """Pose van de simulator (meter) naar (lat, lon) rond SIM_ORIGIN."""
    east = float(pose.get("x", 0.0))
    north = -float(pose.get("z", 0.0))
    lat = SIM_ORIGIN_LAT + north / METERS_PER_DEG_LAT
    lon = SIM_ORIGIN_LON + east / (METERS_PER_DEG_LAT * math.cos(math.radians(SIM_ORIGIN_LAT)))
    return lat, lon


def add_noise(lat, lon, meters):
    """Verschuif een fix willekeurig, om GPS-ruis na te bootsen."""
    if meters <= 0:
        return lat, lon
    north = random.gauss(0.0, meters)
    east = random.gauss(0.0, meters)
    return (lat + north / METERS_PER_DEG_LAT,
            lon + east / (METERS_PER_DEG_LAT * math.cos(math.radians(lat))))


def post_json(url, payload, timeout=3):
    """JSON posten naar stayontrails (zelfondertekend certificaat)."""
    context = ssl.create_default_context()
    context.check_hostname = False
    context.verify_mode = ssl.CERT_NONE
    request = urllib.request.Request(
        url, data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(request, timeout=timeout, context=context) as resp:
        return json.loads(resp.read().decode("utf-8"))


class _PubSub:
    """Zelfde aanroep als conn.datachannel.pub_sub van unitree_webrtc_connect."""

    def __init__(self, connection):
        self._connection = connection

    async def publish_request_new(self, topic, payload):
        message = dict(payload or {})
        message["topic"] = topic
        loop = asyncio.get_running_loop()
        # RemoteSession doet een gewone HTTP-aanroep: niet in de event loop blokkeren
        return await loop.run_in_executor(
            None, self._connection.command, {"webrtc": message})


class SimConnection:
    """Staat in voor UnitreeWebRTCConnection zolang er een simulator draait."""

    def __init__(self, code):
        RemoteSession = load_remote_session_class()
        self.code = code
        self.session = RemoteSession(SIM_API_URL, code)
        # Een RemoteSession houdt één HTTP-verbinding open en is niet
        # thread-safe: de camera krijgt er een eigen, de rest deelt er een
        # achter een lock
        self.camera_session = RemoteSession(SIM_API_URL, code)
        self.datachannel = types.SimpleNamespace(pub_sub=_PubSub(self))
        self._lock = threading.Lock()

    def command(self, action):
        with self._lock:
            return self.session.command(action)

    def state(self):
        with self._lock:
            return self.session.state()

    def pose(self):
        """De pose van de robot, of None als de simulator er (nog) geen geeft."""
        obs = (self.state() or {}).get("obs") or {}
        return obs.get("pose")

    # -- camera --------------------------------------------------------------

    def start_camera(self, on_frame):
        """Haal blijvend het camerabeeld op en geef elk beeld (BGR) aan on_frame."""
        threading.Thread(target=self._run_camera, args=(on_frame,), daemon=True).start()

    def _run_camera(self, on_frame):
        import cv2
        import numpy as np

        interval = 1.0 / SIM_CAMERA_FPS
        last_error = None
        while True:
            started = time.monotonic()
            try:
                jpeg = self.camera_session.frame(fps=SIM_CAMERA_FPS, width=SIM_CAMERA_WIDTH)
            except Exception as exc:  # noqa: BLE001 - 429/503 of netwerk: later opnieuw
                if str(exc) != last_error:
                    print("Simulatorcamera: " + str(exc), flush=True)
                    last_error = str(exc)
                time.sleep(SIM_CAMERA_RETRY)
                continue

            if not jpeg:                  # "no picture yet": even wachten
                time.sleep(SIM_CAMERA_RETRY)
                continue

            image = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
            if image is not None:
                if last_error is not None:
                    print("Simulatorcamera: beeld komt binnen", flush=True)
                    last_error = None
                on_frame(image)

            time.sleep(max(0.0, interval - (time.monotonic() - started)))

    # -- nep-GPS -------------------------------------------------------------

    def start_gps(self, freewalk_url):
        """Stuur elke seconde de positie van de robot als GPS naar stayontrails."""
        threading.Thread(target=self._run_gps, args=(freewalk_url,), daemon=True).start()

    def _run_gps(self, freewalk_url):
        url = freewalk_url.rstrip("/") + "/api/gps"
        last_error = None
        while True:
            try:
                pose = self.pose()
                if pose is not None:
                    lat, lon = add_noise(*pose_to_latlon(pose), SIM_GPS_NOISE_M)
                    post_json(url, {
                        "latitude": lat,
                        "longitude": lon,
                        "gps_accuracy": max(SIM_GPS_ACCURACY_M, 2 * SIM_GPS_NOISE_M),
                        "source": "sim",
                    })
                if last_error is not None:
                    print("Simulator-GPS: weer verbonden", flush=True)
                    last_error = None
            except Exception as exc:  # noqa: BLE001 - stayontrails nog niet op? Gewoon opnieuw
                if str(exc) != last_error:
                    print("Simulator-GPS: " + str(exc), flush=True)
                    last_error = str(exc)
            time.sleep(SIM_GPS_INTERVAL)


# ------------------------------------------------------------- route opnemen

def record_route(code, name):
    """Neem een route op terwijl je de robot in de simulator rondrijdt.

    Elke RECORD_SPACING_M meter komt er een routepunt bij. Ctrl+C bewaart de
    route als stayontrails/paths/<slug>.json."""
    connection = SimConnection(code)
    slug = re.sub(r"[^a-z0-9]+", "-", name.strip().lower()).strip("-") or "sim-route"
    points = []
    length = 0.0
    print("Rij de robot rond in de simulator; Ctrl+C bewaart de route.", flush=True)

    try:
        while True:
            pose = connection.pose()
            if pose is not None:
                east, north = float(pose.get("x", 0.0)), float(pose.get("z", 0.0))
                if not points or math.hypot(east - points[-1][0], north - points[-1][1]) >= RECORD_SPACING_M:
                    if points:
                        length += math.hypot(east - points[-1][0], north - points[-1][1])
                    points.append((east, north, pose_to_latlon(pose)))
                    print("Punt %d (%.1f m)" % (len(points), length), flush=True)
            time.sleep(RECORD_INTERVAL)
    except KeyboardInterrupt:
        pass

    if len(points) < 2:
        print("Te weinig punten, niets bewaard.")
        return

    stamp = int(time.time() * 1000)
    doc = {
        "name": name,
        "slug": slug,
        "language": "nl-BE",
        "isPublic": False,
        "arrivalRadiusMeters": 3,
        "routeLengthMeters": round(length, 1),
        "pointCount": len(points),
        "updatedAt": datetime.now(timezone.utc).isoformat(),
        "arucoMarkers": [],
        "points": [
            {
                "id": "point-%d-%d" % (stamp, index + 1),
                "lat": round(lat, 7), "lng": round(lon, 7), "lon": round(lon, 7),
                "instruction": "Start" if index == 0 else ("Einde" if index == len(points) - 1 else ""),
            }
            for index, (_east, _north, (lat, lon)) in enumerate(points)
        ],
    }
    os.makedirs(PATHS_DIR, exist_ok=True)
    target = os.path.join(PATHS_DIR, slug + ".json")
    with open(target, "w", encoding="utf-8") as bestand:
        json.dump(doc, bestand, indent=2, ensure_ascii=False)
    print("Bewaard: %s (%d punten, %.1f m)" % (target, len(points), length))


if __name__ == "__main__":
    if len(sys.argv) >= 3 and sys.argv[1] == "record":
        record_route(sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else "Sim route")
    else:
        print('Gebruik: python3 simgo2.py record CODE "Naam van de route"')
