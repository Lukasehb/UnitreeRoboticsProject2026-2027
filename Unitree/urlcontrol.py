"""
Webbediening voor de Unitree Go2.

Zelfde commando's als keyboard2.py, maar bediend via een webpagina die
geoptimaliseerd is voor een smartphone. Start dit script op de Jetson en
surf met je gsm naar http://<ip-van-de-jetson>:8080/

    python3 urlcontrol.py
"""

import asyncio
import json
import logging
import math
import os
import random
import ssl
import threading
import time
import urllib.parse
import urllib.request

from flask import Flask, Response, jsonify, render_template_string, request

try:
    import cv2
    import numpy as np
except ImportError:           # zonder opencv/numpy werkt enkel de handbediening
    cv2 = None
    np = None

from unitree_webrtc_connect.webrtc_driver import (
    UnitreeWebRTCConnection,
    WebRTCConnectionMethod
)
from unitree_webrtc_connect.constants import RTC_TOPIC, SPORT_CMD, VUI_COLOR

logging.basicConfig(level=logging.FATAL)

ROBOT_IP = os.environ.get("ROBOT_IP", "10.2.172.215")   # of: ROBOT_IP=x.x.x.x python3 urlcontrol.py
# Sessiecode van de Botopia-simulator (zie simgo2.py). Leeg = de echte Go2.
ROBOT_SIM = os.environ.get("ROBOT_SIM", "").strip()     # ROBOT_SIM=729AM9 python3 urlcontrol.py
WEB_HOST = "0.0.0.0"
WEB_PORT = 8080

MOVE_SPEED = 0.5
TURN_SPEED = 1
BRIGHTNESS_LVL = 1

# Heading-bediening: 90 = recht vooruit, meer = naar rechts, minder = naar links
HEADING_FORWARD = 90.0
HEADING_MOVE_SPEED = 0.2      # traag vooruit terwijl de robot draait
HEADING_FORWARD_TIME = 1.0    # s rechtdoor stappen als de heading precies 90 is
HEADING_MAX_TURN = 180.0      # nooit meer dan een halve draai in één commando
TURN_INTERVAL = 0.1           # s tussen twee move-commando's tijdens het draaien
TURN_CALIBRATION = 1.0        # verhoog als de robot te weinig draait, verlaag als te veel

# Camera + segmentatie
USE_CAMERA = True             # zet op False om zonder camera/YOLO te draaien
MODEL_PATH = os.path.join(                 # relatief aan de repo, waar die ook staat
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "signaling", "models", "laerbeekbos.pt")
DETECTION_CONFIDENCE = 0.5    # startwaarde, op de webpagina aanpasbaar
CONFIDENCE_CHOICES = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
SCAN_HEIGHTS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
MODEL_DIR = os.path.dirname(MODEL_PATH)   # hier zoeken we de andere .pt-bestanden
ALLOWED_PATH_LABELS = {"path", "path-oxod"}
FRAME_INTERVAL = 0.2          # s tussen twee frames die we bijhouden (5 fps volstaat)
SEGMENTATION_INTERVAL = 0.2   # s wachten na een fout voor we opnieuw proberen
FRAME_WAIT_TIMEOUT = 0.5      # s wachten op een beeld voor de lus toch rondgaat
MARKER_CACHE_FRAMES = 16      # zoveel markerresultaten houden we bij voor de segmentatie
HEADING_MAX_AGE = 3.0         # s waarna we een heading als verouderd beschouwen

# Foto's van het ruwe camerabeeld: op vraag of om de zoveel tijd
PHOTO_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "fotos")
PHOTO_QUALITY = 90            # JPEG-kwaliteit van een bewaarde foto
PHOTO_TICK = 0.5              # s tussen twee controles of er een foto moet
PHOTO_INTERVALS = [           # (seconden, wat er op de webpagina staat)
    (0, "geen"),
    (5, "5 seconden"),
    (10, "10 seconden"),
    (30, "30 seconden"),
    (60, "1 minuut"),
    (120, "2 minuten"),
    (300, "5 minuten"),
    (600, "10 minuten"),
]

# Live beeld op de webpagina (/video)
STREAM_SIZE = (640, 480)      # het beeld wordt hierin gepast, met zwarte randen
STREAM_QUALITY = 70           # JPEG-kwaliteit van de stream
OVERLAY_MAX_AGE = 2.0         # s dat we het getekende beeld blijven tonen
STREAM_IDLE_TIMEOUT = 1.0     # s wachten op een nieuw beeld voor we iets sturen
MASK_COLOR = (0, 255, 0)      # BGR: het pad
MASK_ALPHA = 0.35             # hoe hard het masker het beeld inkleurt
MIDPOINT_COLOR = (0, 200, 255)
HEADING_COLOR = (0, 255, 255)

# Het commando dat de robot nu uitvoert, groot in het midden van het beeld
COMMAND_SECONDS = 3.0         # s dat een commando blijft staan
COMMAND_WIDTH = 0.6           # deel van de beeldbreedte dat de tekst inneemt
COMMAND_COLOR = (255, 255, 255)
ESTOP_COLOR = (0, 0, 255)     # BGR: rood

# ArUco: we tonen enkel de grootste marker in beeld
ARUCO_DICTIONARY = "DICT_4X4_50"
ARUCO_COLOR = (255, 0, 255)   # BGR: magenta kader
ARUCO_LABEL_SCALE = 2.1       # groot genoeg om van wat verder af te lezen
ARUCO_LABEL_THICKNESS = 3

# IJkpunten om de afstand te schatten: (oppervlakte in pixels, afstand in meter)
ARUCO_CALIBRATION = [(2100, 1.00), (25000, 0.30)]

# Markers die een commando uitvoeren zodra ze lang genoeg in beeld liggen
ARUCO_ESTOP = "estop"         # dit commando reageert meteen, zonder wachttijd
ARUCO_COMMANDS = {
    10: ARUCO_ESTOP,      # noodstop: breekt meteen af wat er bezig is
    25: "stand_down",     # neerliggen
    23: "stand_up",       # opstaan
    22: "hello",
    29: "stretch",
}
ARUCO_MIN_FRAMES = 3          # zoveel beelden na elkaar zichtbaar voor we reageren
ARUCO_TURN_FRAMES = 2         # draairegels reageren sneller: 2 beelden op de juiste afstand
ARUCO_COOLDOWN = 10        # s voor we hetzelfde commando opnieuw laten uitvoeren

# Draairegels: markers die de robot laten draaien, in te stellen via /aruco
ARUCO_RULES_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "aruco_turn_rules.json")
ARUCO_MARKERS_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "aruco_commands.json")
TURN_DIRECTIONS = ("links", "rechts")
MAX_TURN_SECONDS = 10.0       # grens op de duur van een draai
MAX_RULE_DISTANCE = 10.0      # grens op de afstand waarop een regel afgaat

# Een route volgen met de GPS van de gsm (zie stayontrails/freewalkFlask.py).
# De gsm zit op/bij de robot en draait de stayontrails-pagina; die stuurt zijn
# GPS naar /api/gps. De camera blijft het pad volgen, de GPS beslist wanneer
# een routepunt bereikt is en of we nog de goede kant op lopen.
FREEWALK_URL = os.environ.get("FREEWALK_URL", "https://localhost:5003")
ROUTE_POLL_INTERVAL = 1.0     # s tussen twee GPS-uitlezingen
ROUTE_GPS_MAX_AGE = 5.0       # s zonder verse GPS voor we stoppen
ROUTE_MAX_ACCURACY = 15.0     # m: slechtere fixes tellen niet mee voor aankomst
ROUTE_MIN_COURSE_M = 3.0      # m verplaatsing nodig om de looprichting te schatten
ROUTE_GPS_STEERING = True     # bijsturen als de looprichting ver van het volgende punt wijkt
ROUTE_TURN_THRESHOLD = 50.0   # graden afwijking waarboven we bijsturen
ROUTE_MAX_CORRECTION = 90.0   # nooit meer dan dit in één correctie draaien
ROUTE_CORRECTION_COOLDOWN = 10.0  # s tussen twee correcties

# Het pad volgen: dit loopt door tot er iets is om voor te stoppen
FOLLOW_DEADBAND = 3.0         # graden verschil waarbinnen we niet bijsturen
FOLLOW_FULL_TURN = 90.0       # graden verschil waarbij we op volle bijstuursnelheid zitten
FOLLOW_TURN_SPEED = 0.6       # rad/s bij die afwijking; los van de draaiknoppen
FOLLOW_INTERVAL = 0.15        # s tussen twee move-commando's tijdens het volgen
FOLLOW_NO_PATH_FRAMES = 5     # zoveel beelden na elkaar zonder pad voor we stoppen (~1 s)
MARKER_MAX_AGE = 0.6          # s waarna we een geziene marker als verdwenen beschouwen
DISCO_COLOURS = [VUI_COLOR.RED, VUI_COLOR.YELLOW, VUI_COLOR.GREEN,
                 VUI_COLOR.CYAN, VUI_COLOR.BLUE, VUI_COLOR.PURPLE]


# ---------------------------------------------------------------- robot state

def clamp(value, minimum, maximum):
    return max(minimum, min(maximum, value))


class RobotController:
    """Stuurt commando's naar de Go2 vanuit de Flask threads."""

    def __init__(self):
        self.conn = None
        self.loop = None
        self.connected = False
        self._disco_thread = None
        self._heading_lock = threading.Lock()

        # Staat er een noodstop aan, dan breken lopende bewegingen af en
        # blijven ze weg tot er een nieuw commando komt
        self._abort = threading.Event()

        # Wat de robot nu doet, om op het beeld te zetten
        self.command = None
        self.command_at = 0.0

    # -- laag niveau ---------------------------------------------------------

    def publish(self, topic, payload):
        """Publiceer een bericht op de datachannel (thread-safe)."""
        if not self.connected:
            raise RuntimeError("Geen verbinding met de robot")
        return asyncio.run_coroutine_threadsafe(
            self.conn.datachannel.pub_sub.publish_request_new(topic, payload),
            self.loop
        )

    @property
    def stopped(self):
        """Staat er een noodstop aan?"""
        return self._abort.is_set()

    def resume(self):
        """Een nieuw commando heft de noodstop op."""
        self._abort.clear()

    def note_command(self, label):
        """Onthoud wat de robot nu aan het doen is, voor op het beeld."""
        self.command = label
        self.command_at = time.monotonic()

    def current_command(self):
        """Wat er nu groot op het beeld hoort te staan, of None.

        Een noodstop blijft staan zolang hij duurt; een gewoon commando dooft
        na COMMAND_SECONDS uit."""
        if self.stopped:
            return "NOODSTOP"
        if not self.command:
            return None
        if time.monotonic() - self.command_at > COMMAND_SECONDS:
            return None
        return self.command

    def run_command(self, command):
        """Voer een commando uit de tabel uit.

        Elk commando behalve de noodstop zelf heft een lopende noodstop op; zo
        blijft de robot stil tot je hem iets nieuws vraagt."""
        if command != ARUCO_ESTOP:
            self.resume()

        # Een ander commando betekent dat je zelf de leiding neemt
        if command != "follow":
            path_follower.stop("commando " + command)
            route_follower.stop("commando " + command)

        self.note_command(command)
        return COMMANDS[command]()

    def sport(self, api_id, parameter=None):
        payload = {"api_id": api_id}
        if parameter is not None:
            payload["parameter"] = parameter
        return self.publish(RTC_TOPIC["SPORT_MOD"], payload)

    def vui(self, api_id, parameter):
        return self.publish(RTC_TOPIC["VUI"], {"api_id": api_id, "parameter": parameter})

    # -- bewegen -------------------------------------------------------------

    def move(self, x=0, y=0, z=0):
        return self.sport(SPORT_CMD["Move"], {"x": x, "y": y, "z": z})

    def steer_to(self, heading):
        """Vooruit stappen en evenredig bijsturen naar een heading.

        Hoe verder het doel van recht vooruit ligt, hoe harder we bijsturen:
        de draaisnelheid loopt recht evenredig met de afwijking. Een heading
        ligt tussen 0 en 180, dus de afwijking blijft binnen 90 graden; met
        FOLLOW_FULL_TURN op 90 loopt die lijn over het hele bereik en zit de
        robot nooit tegen zijn maximum aan te schuren.

        We rekenen met FOLLOW_TURN_SPEED en niet met TURN_SPEED: dat laatste
        is de draaisnelheid van de knoppen links en rechts, een bewuste snelle
        draai. Voor bijsturen is dat veel te hard."""
        error = heading - HEADING_FORWARD
        if abs(error) < FOLLOW_DEADBAND:
            z = 0.0
        else:
            # Evenredig met de afwijking; positieve z is links
            z = -clamp(error / FOLLOW_FULL_TURN, -1.0, 1.0) * FOLLOW_TURN_SPEED

        return self.move(x=MOVE_SPEED, z=z)

    def follow_path(self):
        """Het pad volgen zoals de camera het ziet.

        Ziet de camera niets bruikbaars, dan stappen we gewoon rechtdoor."""
        heading = segmentation.current_heading()
        if heading is None:
            return self.move(x=MOVE_SPEED)
        return self.steer_to(heading)

    # -- heading -------------------------------------------------------------

    def follow_heading(self, degrees):
        """Volg een heading: draai het gegeven aantal graden (positief =
        rechts, negatief = links) terwijl de robot traag vooruit gaat. Is de
        heading precies recht vooruit (0 graden verschil), dan stapt de robot
        HEADING_FORWARD_TIME seconden rechtdoor.

        De Go2 beweegt zolang hij move-commando's krijgt, dus we blijven
        herhalen tot de berekende tijd voorbij is en sturen daarna een stop.

        Geeft de hoek en de duur terug, of None als er al een beweging bezig
        is."""
        if not self._heading_lock.acquire(blocking=False):
            return None

        try:
            degrees = clamp(degrees, -HEADING_MAX_TURN, HEADING_MAX_TURN)
            self.note_command("heading %+.0f" % degrees)

            if degrees == 0:
                # Recht vooruit: gewoon een seconde stappen, niet draaien
                duration = HEADING_FORWARD_TIME
                z = 0.0
            else:
                duration = math.radians(abs(degrees)) / TURN_SPEED * TURN_CALIBRATION
                # In de Go2 is een positieve z een draai naar links
                z = -TURN_SPEED if degrees > 0 else TURN_SPEED

            self.resume()
            deadline = time.monotonic() + duration
            while time.monotonic() < deadline:
                if self._abort.is_set():      # noodstop: meteen afbreken
                    break
                self.move(x=HEADING_MOVE_SPEED, z=z)
                time.sleep(TURN_INTERVAL)
            self.move(x=0, y=0, z=0)

            return {
                "degrees": round(degrees, 1),
                "forward": HEADING_MOVE_SPEED,
                "duration": round(duration, 2),
            }
        finally:
            self._heading_lock.release()

    def turn_for(self, direction, seconds):
        """Draai een aantal seconden naar links of naar rechts, en stop daarna.

        De Go2 beweegt zolang hij move-commando's krijgt, dus we blijven
        herhalen tot de tijd om is. Is er al een beweging bezig, dan doen we
        niets en geven we None terug."""
        if not self._heading_lock.acquire(blocking=False):
            return None

        try:
            self.note_command("draai %s %g s" % (direction, seconds))

            # In de Go2 is een positieve z een draai naar links
            z = TURN_SPEED if direction == "links" else -TURN_SPEED

            deadline = time.monotonic() + seconds
            while time.monotonic() < deadline:
                if self._abort.is_set():      # noodstop: meteen afbreken
                    break
                self.move(z=z)
                time.sleep(TURN_INTERVAL)
            self.move(x=0, y=0, z=0)

            return {"direction": direction, "seconds": seconds}
        finally:
            self._heading_lock.release()

    # -- licht ---------------------------------------------------------------

    def colour(self, colour):
        return self.vui(1007, {"color": colour})

    def disco(self):
        """Laat de kleuren wisselen in een aparte thread, zodat de webpagina
        niet blijft wachten."""
        if self._disco_thread and self._disco_thread.is_alive():
            return

        def run():
            for _ in range(30):
                try:
                    self.colour(random.choice(DISCO_COLOURS))
                except Exception:
                    break
                time.sleep(0.2)

        self._disco_thread = threading.Thread(target=run, daemon=True)
        self._disco_thread.start()

    # -- noodstop ------------------------------------------------------------

    def emergency_stop(self):
        """Zet de robot meteen stil en breek af wat er bezig is.

        De vlag blijft staan tot er een nieuw commando komt, zodat een draai of
        een heading die nog aan het lopen was niet verder gaat."""
        self._abort.set()
        self.move(x=0, y=0, z=0)


robot = RobotController()


# ---------------------------------------------------------- camera + segmentatie

def create_aruco_detector(dictionary_name):
    """Maak een detector die werkt met zowel de oude als de nieuwe OpenCV-API
    (zelfde opzet als in arucoContro.py)."""
    if not hasattr(cv2, "aruco"):
        raise RuntimeError("deze OpenCV heeft geen cv2.aruco (opencv-contrib-python)")

    dictionary_id = getattr(cv2.aruco, dictionary_name, None)
    if dictionary_id is None:
        raise RuntimeError("onbekende ArUco-dictionary: " + dictionary_name)

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
        return lambda image: detector.detectMarkers(image)

    return lambda image: cv2.aruco.detectMarkers(image, dictionary, parameters=parameters)


_aruco_detector = None
_aruco_tried = False


def detect_markers(frame):
    """Zoek alle ArUco-markers in het beeld, van groot naar klein.

    Elke marker is een dict met zijn id, zijn hoekpunten en zijn oppervlakte."""
    global _aruco_detector, _aruco_tried

    if not _aruco_tried:
        _aruco_tried = True
        try:
            _aruco_detector = create_aruco_detector(ARUCO_DICTIONARY)
        except Exception as exc:
            print("ArUco niet beschikbaar: " + str(exc), flush=True)

    if _aruco_detector is None:
        return []

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    corners, ids, _rejected = _aruco_detector(gray)
    if ids is None or len(ids) == 0:
        return []

    markers = []
    for marker_id, marker_corners in zip(ids, corners):
        points = np.round(marker_corners.reshape((4, 2))).astype(np.int32)
        markers.append({
            "id": int(marker_id[0]),
            "points": points,
            "area": float(cv2.contourArea(points)),
        })

    markers.sort(key=lambda marker: marker["area"], reverse=True)
    return markers


def detect_largest_marker(frame):
    """De grootste marker in beeld, of None.

    De grootste is doorgaans ook de dichtste, en één marker tegelijk houdt het
    beeld rustig. Enkel de noodstop kijkt naar alle markers."""
    markers = detect_markers(frame)
    return markers[0] if markers else None


def estop_marker(markers):
    """De noodstopmarker tussen de markers in beeld, of None.

    Hier telt de grootte niet mee: ook een kleine marker ver weg moet de robot
    stilzetten."""
    for marker in markers:
        if command_markers.get_command(marker["id"]) == ARUCO_ESTOP:
            return marker
    return None


def marker_distance(area):
    """Schat de afstand tot de marker uit zijn oppervlakte in pixels.

    Een marker die twee keer zo ver staat is half zo breed, en beslaat dus een
    kwart van de oppervlakte. De zijde (de wortel van de oppervlakte) is met
    andere woorden omgekeerd evenredig met de afstand, dus interpoleren we
    rechtlijnig in 1/wortel(oppervlakte) door de twee ijkpunten.

    Geeft meters terug, of None als er niets te rekenen valt."""
    (area_a, distance_a), (area_b, distance_b) = ARUCO_CALIBRATION
    if area <= 0 or area_a <= 0 or area_b <= 0 or area_a == area_b:
        return None

    x = 1.0 / math.sqrt(area)
    x_a, x_b = 1.0 / math.sqrt(area_a), 1.0 / math.sqrt(area_b)
    slope = (distance_a - distance_b) / (x_a - x_b)
    return max(0.0, distance_b + slope * (x - x_b))


def draw_command(frame, text, colour):
    """Zet het commando groot in het midden van het beeld.

    De tekst wordt zo geschaald dat ze COMMAND_WIDTH van de breedte inneemt,
    zodat ze even groot oogt op elke resolutie."""
    h, w = frame.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    text = str(text).upper()

    (base_w, _base_h), _baseline = cv2.getTextSize(text, font, 1.0, 2)
    scale = w * COMMAND_WIDTH / max(base_w, 1)
    thickness = max(2, int(round(scale * 1.5)))

    (text_w, text_h), _baseline = cv2.getTextSize(text, font, scale, thickness)
    x = (w - text_w) // 2
    y = (h + text_h) // 2

    # Eerst een zwarte rand, zo blijft het leesbaar op eender welke achtergrond
    cv2.putText(frame, text, (x, y), font, scale, (0, 0, 0), thickness + 4, cv2.LINE_AA)
    cv2.putText(frame, text, (x, y), font, scale, colour, thickness, cv2.LINE_AA)


def validate_rule(marker_id, direction, seconds, distance, margin):
    """Controleer de waarden van een draairegel en geef ze opgekuist terug.

    Bij een fout komt er een ValueError met een boodschap die op de pagina
    getoond kan worden."""
    try:
        marker_id = int(marker_id)
    except (TypeError, ValueError):
        raise ValueError("de marker moet een geheel getal zijn")

    if not 0 <= marker_id <= 999:
        raise ValueError("de marker moet tussen 0 en 999 liggen")
    used_by = command_markers.get_command(marker_id)
    if used_by is not None:
        raise ValueError("marker %d is al een vast commando (%s)"
                         % (marker_id, used_by))

    direction = str(direction).strip().lower()
    if direction not in TURN_DIRECTIONS:
        raise ValueError("de richting moet 'links' of 'rechts' zijn")

    def number(value, name, low, high):
        try:
            value = float(value)
        except (TypeError, ValueError):
            raise ValueError(name + " moet een getal zijn")
        if not low <= value <= high:
            raise ValueError("%s moet tussen %g en %g liggen" % (name, low, high))
        return value

    return {
        "id": marker_id,
        "direction": direction,
        "seconds": round(number(seconds, "de duur", 0.1, MAX_TURN_SECONDS), 2),
        "distance": round(number(distance, "de afstand", 0.1, MAX_RULE_DISTANCE), 2),
        "margin": round(number(margin, "de marge", 0.0, 100.0), 1),
    }


def rule_range(rule):
    """Tussen welke afstanden de regel afgaat."""
    edge = rule["distance"] * rule["margin"] / 100.0
    return rule["distance"] - edge, rule["distance"] + edge


def rule_matches(rule, distance):
    """Ligt de marker op de afstand waarop deze regel moet afgaan?"""
    if distance is None:
        return False
    low, high = rule_range(rule)
    return low <= distance <= high


def validate_command_marker(command, marker_id):
    """Controleer een koppeling tussen een marker en een vast commando."""
    command = str(command).strip()
    if command not in COMMANDS:
        raise ValueError("onbekend commando: " + command)

    try:
        marker_id = int(marker_id)
    except (TypeError, ValueError):
        raise ValueError("de marker moet een geheel getal zijn")
    if not 0 <= marker_id <= 999:
        raise ValueError("de marker moet tussen 0 en 999 liggen")

    return command, marker_id


class CommandMarkers:
    """Welke marker welk vast commando uitvoert.

    Bij een eerste start gelden de koppelingen uit ARUCO_COMMANDS. Wat je via
    /aruco wijzigt komt in een json-bestand naast dit script en heeft daarna
    voorrang."""

    def __init__(self, path, defaults):
        self.path = path
        self._lock = threading.Lock()

        markers = {}
        for marker_id, command in defaults.items():
            # Een tikfout in ARUCO_COMMANDS moet meteen opvallen
            command, marker_id = validate_command_marker(command, marker_id)
            markers[command] = marker_id
        self._markers = markers

        self.load()

    def load(self):
        try:
            with open(self.path) as bestand:
                stored = json.load(bestand)
        except FileNotFoundError:
            return
        except (OSError, ValueError) as exc:
            print("Markercommando's lezen mislukt: " + str(exc), flush=True)
            return

        markers = {}
        for command, marker_id in (stored.items() if isinstance(stored, dict) else []):
            try:
                command, marker_id = validate_command_marker(command, marker_id)
            except ValueError as exc:
                print("Markercommando overgeslagen: " + str(exc), flush=True)
                continue
            markers[command] = marker_id

        with self._lock:
            self._markers = markers
        print("Markercommando's geladen: %d" % len(markers), flush=True)

    def save(self):
        with self._lock:
            markers = dict(self._markers)
        try:
            with open(self.path, "w") as bestand:
                json.dump(markers, bestand, indent=2)
        except OSError as exc:
            print("Markercommando's bewaren mislukt: " + str(exc), flush=True)

    def all(self):
        """Alle koppelingen, op markernummer gesorteerd."""
        with self._lock:
            items = list(self._markers.items())
        return [{"command": command, "id": marker_id}
                for command, marker_id in sorted(items, key=lambda item: item[1])]

    def as_dict(self):
        """De koppelingen als marker -> commando, zoals ARUCO_COMMANDS."""
        with self._lock:
            return dict((marker_id, command)
                        for command, marker_id in self._markers.items())

    def get_command(self, marker_id):
        """Welk commando hoort bij deze marker? None als er geen is."""
        with self._lock:
            for command, other_id in self._markers.items():
                if other_id == marker_id:
                    return command
        return None

    def set(self, command, marker_id):
        """Koppel een marker aan een commando, of verplaats een bestaande."""
        command, marker_id = validate_command_marker(command, marker_id)

        in_use = self.get_command(marker_id)
        if in_use is not None and in_use != command:
            raise ValueError("marker %d is al gekoppeld aan %s" % (marker_id, in_use))
        if turn_rules.get(marker_id) is not None:
            raise ValueError("marker %d heeft al een draairegel" % marker_id)

        with self._lock:
            self._markers[command] = marker_id
        self.save()
        return {"command": command, "id": marker_id}

    def delete(self, command):
        command = str(command).strip()
        with self._lock:
            removed = self._markers.pop(command, None)
        if removed is None:
            raise ValueError("er is geen marker gekoppeld aan " + command)
        self.save()
        return {"command": command, "id": removed}


class TurnRules:
    """De draairegels van de pagina /aruco.

    Ze worden in een json-bestand naast dit script bewaard, zodat ze een
    herstart overleven."""

    def __init__(self, path):
        self.path = path
        self._lock = threading.Lock()
        self._rules = {}
        self.load()

    def load(self):
        try:
            with open(self.path) as bestand:
                stored = json.load(bestand)
        except FileNotFoundError:
            return
        except (OSError, ValueError) as exc:
            print("Draairegels lezen mislukt: " + str(exc), flush=True)
            return

        rules = {}
        for item in stored if isinstance(stored, list) else []:
            try:
                rule = validate_rule(item.get("id"), item.get("direction"),
                                     item.get("seconds"), item.get("distance"),
                                     item.get("margin"))
            except (ValueError, AttributeError) as exc:
                print("Draairegel overgeslagen: " + str(exc), flush=True)
                continue
            rules[rule["id"]] = rule

        with self._lock:
            self._rules = rules
        print("Draairegels geladen: %d" % len(rules), flush=True)

    def save(self):
        try:
            with open(self.path, "w") as bestand:
                json.dump(self.all(), bestand, indent=2)
        except OSError as exc:
            print("Draairegels bewaren mislukt: " + str(exc), flush=True)

    def all(self):
        with self._lock:
            return [self._rules[key] for key in sorted(self._rules)]

    def get(self, marker_id):
        with self._lock:
            return self._rules.get(marker_id)

    def set(self, marker_id, direction, seconds, distance, margin):
        """Voeg een regel toe of pas een bestaande aan."""
        rule = validate_rule(marker_id, direction, seconds, distance, margin)
        with self._lock:
            self._rules[rule["id"]] = rule
        self.save()
        return rule

    def delete(self, marker_id):
        try:
            marker_id = int(marker_id)
        except (TypeError, ValueError):
            raise ValueError("de marker moet een geheel getal zijn")

        with self._lock:
            removed = self._rules.pop(marker_id, None)
        if removed is None:
            raise ValueError("er is geen regel voor marker %d" % marker_id)
        self.save()
        return removed


def draw_marker(frame, marker):
    """Teken een kader rond de marker, met zijn nummer en oppervlakte erboven."""
    h, w = frame.shape[:2]
    points = marker["points"]
    cv2.polylines(frame, [points.reshape((-1, 1, 2))], True, ARUCO_COLOR, 3, cv2.LINE_AA)

    distance = marker_distance(marker["area"])
    label = "aruco %d" % marker["id"]
    if distance is not None:
        label += " - %.2f m" % distance
    (text_w, text_h), _baseline = cv2.getTextSize(
        label, cv2.FONT_HERSHEY_SIMPLEX, ARUCO_LABEL_SCALE, ARUCO_LABEL_THICKNESS)

    # Net boven het kader, maar altijd binnen het beeld
    x = clamp(int(points[:, 0].min()), 5, max(5, w - text_w - 5))
    y = clamp(int(points[:, 1].min()) - 10, text_h + 5, h - 5)

    cv2.putText(frame, label, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX,
                ARUCO_LABEL_SCALE, ARUCO_COLOR, ARUCO_LABEL_THICKNESS, cv2.LINE_AA)


def available_models():
    """De .pt-bestanden die naast het ingestelde model staan."""
    try:
        names = [name for name in os.listdir(MODEL_DIR or ".")
                 if name.lower().endswith(".pt")]
    except OSError:
        names = []
    return sorted(names)


def get_allowed_mask_indices(result, model_names):
    """Houd enkel de maskers over die een pad voorstellen."""
    if result.boxes is None or result.boxes.cls is None:
        return []

    allowed_indices = []
    class_ids = result.boxes.cls.cpu().numpy().astype(int).tolist()
    for index, class_id in enumerate(class_ids):
        label = str(model_names.get(class_id, "")).strip().lower()
        if label in ALLOWED_PATH_LABELS:
            allowed_indices.append(index)
    return allowed_indices


def largest_allowed_mask(result, model_names, shape):
    """Zoek van alle maskers die een pad voorstellen het grootste.

    We werken bewust met één vlak: een tweede pad, een zijweggetje of een
    los stukje berm zou de heading anders doen verspringen.

    Geeft de index en het masker op beeldformaat terug, of (None, None)."""
    h, w = shape
    best_index, best_mask, best_area = None, None, 0

    for index in get_allowed_mask_indices(result, model_names):
        if index >= len(result.masks.data):
            continue

        mask = result.masks.data[index].cpu().numpy()
        mask = (mask * 255).astype(np.uint8)
        mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)

        area = int(np.count_nonzero(mask))
        if area > best_area:
            best_index, best_mask, best_area = index, mask, area

    return best_index, best_mask


def compute_heading_to_point(frame, target_x, target_y):
    """Hoek van het midden onderaan het beeld naar een punt, in de conventie van
    deze pagina: 90 is recht vooruit, meer dan 90 is naar rechts.

    arctan2 telt tegen de klok in, dus een punt rechts van het midden geeft daar
    minder dan 90 graden (zo staat het ook in segmentAndControlGo2.py). Daarom
    spiegelen we het resultaat rond 90."""
    h, w = frame.shape[:2]
    start_x = w // 2
    start_y = h
    dx = target_x - start_x
    dy = start_y - target_y
    angle = float(np.degrees(np.arctan2(dy, dx)))
    return 2 * HEADING_FORWARD - angle


def draw_path_overlay(frame, result, mask_index, midpoints, heading):
    """Teken het grootste pad, de meetpunten en de heading op het beeld."""
    h, w = frame.shape[:2]
    masks_xy = getattr(result.masks, "xy", None) if result.masks is not None else None
    polygon = None

    if masks_xy is not None and mask_index is not None and mask_index < len(masks_xy):
        polygon = masks_xy[mask_index]

    if polygon is not None and len(polygon) >= 3:
        points = np.round(polygon).astype(np.int32).reshape((-1, 1, 2))
        points[:, 0, 0] = np.clip(points[:, 0, 0], 0, w - 1)
        points[:, 0, 1] = np.clip(points[:, 0, 1], 0, h - 1)

        shaded = frame.copy()
        cv2.fillPoly(shaded, [points], MASK_COLOR)
        cv2.addWeighted(shaded, MASK_ALPHA, frame, 1.0 - MASK_ALPHA, 0, dst=frame)
        cv2.polylines(frame, [points], True, MASK_COLOR, 2, cv2.LINE_AA)

    for x, y in midpoints:
        cv2.circle(frame, (x, y), 4, MIDPOINT_COLOR, -1, cv2.LINE_AA)

    if midpoints:
        avg_x = int(np.mean([point[0] for point in midpoints]))
        target_y = min(point[1] for point in midpoints)
        cv2.arrowedLine(frame, (w // 2, h - 1), (avg_x, target_y),
                        HEADING_COLOR, 3, cv2.LINE_AA, tipLength=0.08)

    text = "geen pad" if heading is None else "heading: %.1f" % heading
    cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                HEADING_COLOR, 2, cv2.LINE_AA)


def segment_frame(frame, model, confidence):
    """Zoek het pad in het beeld en geef de heading ernaartoe, samen met een
    kopie van het beeld waarop het masker getekend staat (voor de live stream).

    We houden enkel het grootste pad over.  Daarvan kijken we op een aantal
    hoogtes waar het zit, en we mikken op het gemiddelde van die punten. Wordt
    er niets gevonden, dan geven we recht vooruit terug.  Hoe lager de
    confidence, hoe sneller het model iets een pad noemt."""
    h, w = frame.shape[:2]
    result = model(frame, conf=confidence, verbose=False)[0]
    model_names = getattr(model, "names", {})
    midpoints = []
    mask_index, mask = None, None

    if result.masks is not None and len(result.masks.data) > 0:
        mask_index, mask = largest_allowed_mask(result, model_names, (h, w))

    if mask is not None:
        for row_ratio in SCAN_HEIGHTS:
            y = int(h * row_ratio)
            if y >= h:
                continue
            filled_x = np.where(mask[y, :] > 0)[0]
            if len(filled_x) > 0:
                midpoints.append((int(np.mean(filled_x)), y))

    if midpoints:
        avg_x = int(np.mean([point[0] for point in midpoints]))
        target_y = min(point[1] for point in midpoints)
        heading = compute_heading_to_point(frame, avg_x, target_y)
    else:
        # Geen pad in beeld: geen heading. Zo weet wie volgt dat hij moet
        # stoppen, in plaats van blind op 90 graden rechtdoor te lopen.
        heading = None

    overlay = frame.copy()
    draw_path_overlay(overlay, result, mask_index, midpoints, heading)
    return heading, overlay


def fit_to_stream_size(image):
    """Schaal het beeld naar STREAM_SIZE en vul de rest zwart op, zodat de
    stream altijd even groot is en de verhoudingen kloppen."""
    width, height = STREAM_SIZE
    h, w = image.shape[:2]
    if (w, h) == STREAM_SIZE:
        return image

    scale = min(width / w, height / h)
    new_w, new_h = max(1, int(w * scale)), max(1, int(h * scale))
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)

    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    x, y = (width - new_w) // 2, (height - new_h) // 2
    canvas[y:y + new_h, x:x + new_w] = resized
    return canvas


def placeholder_frame(text):
    """Zwart beeld met een boodschap, zolang er geen camerabeeld binnenkomt."""
    width, height = STREAM_SIZE
    image = np.zeros((height, width, 3), dtype=np.uint8)
    cv2.putText(image, str(text)[:40], (20, height // 2), cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (160, 160, 160), 2, cv2.LINE_AA)
    return image


class Segmentation:
    """Berekent voortdurend de heading uit de camerabeelden van de robot.

    Het model wordt in een achtergrondthread geladen, zodat de webpagina meteen
    bruikbaar is en niet op YOLO moet wachten."""

    def __init__(self):
        self._frame = None
        self._frame_seq = 0
        self._frame_cond = threading.Condition()
        self._last_frame_at = 0.0
        self._raw_frame = None            # het laatste beeld zoals het binnenkwam
        self._raw_lock = threading.Lock()
        self.heading = None
        self.updated_at = 0.0
        self.status = "uit" if not USE_CAMERA else "wachten op beeld"
        self.error = None
        self.confidence = DETECTION_CONFIDENCE
        self.model_name = os.path.basename(MODEL_PATH)
        self.marker = None
        self.marker_id = None         # de marker die de snelle thread net zag
        self.marker_at = 0.0

        # ArUco-commando's: tellen hoe lang dezelfde marker in beeld ligt
        self._seen_id = None
        self._seen_count = 0
        self._range_count = 0
        self._triggered_at = {}
        self.last_command = None
        self.command_count = 0

        # Voor de noodstopthread, die elk binnenkomend beeld bekijkt
        self._marker_cond = threading.Condition()
        self._marker_frame = None
        self._marker_seq = 0
        self._markers_by_seq = {}     # wat die thread per beeld vond, om te hergebruiken
        self._aims = (0.0, ())        # (tijdstip, richtingsmarkers in beeld)

        # Voor de live stream op /video
        self._stream_cond = threading.Condition()
        self._stream_image = None
        self._stream_seq = 0
        self._overlay_at = 0.0

    # -- beelden binnenkrijgen ----------------------------------------------

    def put_frame(self, image):
        """Bewaar enkel het laatste beeld; oudere beelden zijn toch achterhaald."""
        # Het kale beeld apart bijhouden: daar maken de foto's gebruik van
        with self._raw_lock:
            self._raw_frame = image

        self._publish_stream(image, annotated=False)

        # De noodstopthread krijgt elk beeld, die mag niet wachten
        with self._marker_cond:
            self._marker_seq += 1
            seq = self._marker_seq
            self._marker_frame = image
            self._marker_cond.notify_all()

        now = time.monotonic()
        if now - self._last_frame_at < FRAME_INTERVAL:
            return
        self._last_frame_at = now
        with self._frame_cond:
            self._frame = image
            self._frame_seq = seq
            self._frame_cond.notify_all()

    def take_frame(self, timeout=None):
        """Wacht op het volgende beeld voor de segmentatie: (beeld, volgnummer).

        Wachten in plaats van pollen: het tempo ligt al vast bij put_frame, met
        FRAME_INTERVAL. Een sleep achteraf kwam daar bovenop en maakte de
        segmentatie trager dan die ene rem. Komt er niets binnen de timeout, dan
        geven we (None, 0) terug zodat de lus toch rondgaat en bijvoorbeeld een
        nieuw gekozen model opmerkt.

        Het volgnummer hoort bij hetzelfde beeld als _marker_seq, zodat we de
        markers kunnen ophalen die de snelle thread er al op vond."""
        with self._frame_cond:
            if self._frame is None:
                self._frame_cond.wait(timeout)
            if self._frame is None:
                return None, 0
            image, self._frame = self._frame, None
            return image, self._frame_seq

    def raw_frame(self):
        """Het laatste camerabeeld zonder masker, kaders of tekst, of None.

        Niemand tekent op een binnengekomen beeld: de segmentatie werkt op een
        kopie. Dit blijft dus het beeld zoals de camera het aanleverde."""
        with self._raw_lock:
            return self._raw_frame

    def put_markers(self, seq, markers):
        """Bewaar wat de markerthread op dit beeld vond, voor de segmentatie.

        We houden er een handvol bij: de segmentatie vraagt ze pas op als YOLO
        klaar is, en tegen dan staat de markerthread al een paar beelden
        verder."""
        with self._marker_cond:
            self._markers_by_seq[seq] = markers
            while len(self._markers_by_seq) > MARKER_CACHE_FRAMES:
                del self._markers_by_seq[next(iter(self._markers_by_seq))]

    def _publish_aims(self, image, markers):
        """Zet klaar waar de richtingsmarkers liggen, voor wie erop wil mikken.

        Een richtingsmarker is er een met een draairegel. We rekenen er
        dezelfde heading voor uit als voor het pad, vanaf het midden onderaan
        het beeld naar het midden van de marker: 90 is recht vooruit, meer is
        naar rechts. Zo stuurt de robot met dezelfde regeling naar een marker
        als naar een pad.

        De lijst staat op volgorde van grootte, dus de dichtste vooraan."""
        aims = []
        for marker in markers:
            if turn_rules.get(marker["id"]) is None:
                continue
            points = marker["points"]
            heading = compute_heading_to_point(
                image, float(np.mean(points[:, 0])), float(np.mean(points[:, 1])))
            aims.append({
                "id": marker["id"],
                "heading": clamp(heading, 0.0, 180.0),
                "distance": marker_distance(marker["area"]),
            })
        self._aims = (time.monotonic(), tuple(aims))

    def aims(self):
        """De richtingsmarkers in beeld, de dichtste eerst.

        Een lege tuple als er geen zijn of als het laatste beeld ouder is dan
        MARKER_MAX_AGE: dan beschouwen we ze als uit beeld."""
        at, aims = self._aims
        if not aims or time.monotonic() - at > MARKER_MAX_AGE:
            return ()
        return aims

    def aim_for(self, marker_id):
        """De richtingsmarker met dit nummer, of None als hij niet in beeld is."""
        for aim in self.aims():
            if aim["id"] == marker_id:
                return aim
        return None

    def turn_rule_on_cooldown(self, marker_id):
        """Is de draairegel van deze marker net afgegaan?

        Dezelfde cooldown als in marker_action, en met dezelfde boekhouding:
        zowel de segmentatie als de volger kan een regel laten afgaan, en na
        een draai staat de marker vaak nog in beeld. Zonder dit zou hij meteen
        opnieuw afgaan."""
        last = self._triggered_at.get(marker_id)
        return last is not None and time.monotonic() - last < ARUCO_COOLDOWN

    def note_turn_rule(self, marker_id):
        """Onthoud dat deze draairegel nu afgaat, voor de cooldown."""
        self._triggered_at[marker_id] = time.monotonic()

    def markers_for(self, seq):
        """De markers die de markerthread op dit beeld vond, of None.

        None betekent enkel dat die thread er nog niet aan toe was; de
        segmentatie zoekt dan zelf. Zo hoort het kader altijd bij het beeld
        waarop het getekend wordt, zonder dezelfde detectie twee keer te
        doen."""
        with self._marker_cond:
            return self._markers_by_seq.get(seq)

    # -- commando's van een marker -------------------------------------------

    def marker_action(self, marker):
        """Geef de actie die bij de marker hoort, of None.

        Een marker uit ARUCO_COMMANDS geeft een vast commando, zodra hij
        ARUCO_MIN_FRAMES beelden na elkaar in beeld ligt; zo zet een marker die
        even voorbijflitst de robot niet aan het werk.

        Bij een draairegel moet het sneller gaan: daar volstaan
        ARUCO_TURN_FRAMES beelden na elkaar waarop de marker ook nog eens op de
        ingestelde afstand ligt, marge inbegrepen. Een beeld buiten die afstand
        zet die teller weer op nul.

        In beide gevallen houden we de marker daarna ARUCO_COOLDOWN seconden
        tegen, zodat hij niet telkens opnieuw afgaat."""
        marker_id = marker["id"] if marker else None

        if marker_id != self._seen_id:
            self._seen_id = marker_id
            self._seen_count = 0
            self._range_count = 0
        self._seen_count += 1

        if marker_id is None:
            return None

        # Zonder verbinding valt er niets uit te voeren; we wachten gewoon af
        if not robot.connected:
            return None

        command = command_markers.get_command(marker_id)
        if command is not None:
            if self._seen_count < ARUCO_MIN_FRAMES:
                return None
            action = {"kind": "command", "command": command, "label": command}
        else:
            rule = turn_rules.get(marker_id)
            if rule is None:
                return None

            # Lijnt de volger op een richtingsmarker uit, dan regelt die het
            # draaien: hij rijdt er zelf naartoe en weet wanneer hij er is.
            # Andere richtingsmarkers tellen zolang niet mee.
            if path_follower.aligning_id() is not None:
                return None

            if rule_matches(rule, marker_distance(marker["area"])):
                self._range_count += 1
            else:
                self._range_count = 0

            if self._range_count < ARUCO_TURN_FRAMES:
                return None

            action = {
                "kind": "turn",
                "rule": rule,
                "label": "draai %s, %g s" % (rule["direction"], rule["seconds"]),
            }

        now = time.monotonic()
        last = self._triggered_at.get(marker_id)
        if last is not None and now - last < ARUCO_COOLDOWN:
            return None

        self._triggered_at[marker_id] = now
        return action

    def trigger_estop(self, marker):
        """Zet de robot meteen stil omdat de noodstopmarker in beeld ligt.

        Geen tellen, geen cooldown en geen afstandsvoorwaarde: dit gebeurt bij
        het eerste beeld waarop de marker gezien wordt. Staat de robot al stil,
        dan valt er niets meer te doen."""
        if robot.stopped or not robot.connected:
            return

        self.last_command = {"id": marker["id"], "command": "noodstop"}
        self.command_count += 1
        print("ArUco %d: noodstop" % marker["id"], flush=True)

        try:
            robot.run_command(ARUCO_ESTOP)
        except Exception as exc:
            print("Noodstop mislukt: " + str(exc), flush=True)

    def run_marker_command(self, marker):
        """Voer de actie van de marker uit, als er een aan de beurt is."""
        if marker is not None and \
                command_markers.get_command(marker["id"]) == ARUCO_ESTOP:
            self.trigger_estop(marker)
            return

        action = self.marker_action(marker)
        if action is None:
            return

        self.last_command = {"id": marker["id"], "command": action["label"]}
        self.command_count += 1
        print("ArUco %d: %s" % (marker["id"], action["label"]), flush=True)

        try:
            if action["kind"] == "command":
                robot.run_command(action["command"])
            else:
                # Volgt de robot een pad, dan zetten we hem eerst stil
                resume = path_follower.pause_for_turn()
                robot.resume()
                # Draaien duurt seconden, dus dat gebeurt naast de segmentatie
                threading.Thread(
                    target=self.turn_and_resume,
                    args=(action["rule"], resume),
                    daemon=True).start()
        except Exception as exc:
            print("ArUco-commando mislukt: " + str(exc), flush=True)

    def turn_and_resume(self, rule, resume):
        """Draai zoals de regel zegt en zet het volgen daarna terug aan.

        Enkel hervatten als het volgen aanstond toen de regel afging: wie met
        de hand aan het rijden was, blijft met de hand rijden.

        Een noodstop tijdens het draaien gaat voor. path_follower.start() heft
        een noodstop op, dus die mogen we dan niet oproepen."""
        try:
            robot.turn_for(rule["direction"], rule["seconds"])
        except Exception as exc:
            print("Draairegel mislukt: " + str(exc), flush=True)
            return

        if resume and not robot.stopped:
            path_follower.start()

    # -- live stream ---------------------------------------------------------

    def _publish_stream(self, image, annotated):
        """Zet een beeld klaar voor /video.

        Zolang het model beelden aflevert tonen we die met het masker erop; het
        kale camerabeeld dient enkel als terugval, bijvoorbeeld terwijl YOLO nog
        aan het laden is of wanneer de segmentatie stilvalt."""
        now = time.monotonic()
        with self._stream_cond:
            if annotated:
                self._overlay_at = now
            elif now - self._overlay_at < OVERLAY_MAX_AGE:
                return
            self._stream_image = image
            self._stream_seq += 1
            self._stream_cond.notify_all()

    def stream_frames(self):
        """Blijf JPEG's opleveren voor de MJPEG-stream van /video."""
        last_seq = -1
        while True:
            with self._stream_cond:
                if self._stream_seq == last_seq:
                    self._stream_cond.wait(timeout=STREAM_IDLE_TIMEOUT)
                image = None
                if self._stream_seq != last_seq:
                    image, last_seq = self._stream_image, self._stream_seq

            # Niets nieuws? Toch iets sturen: zo merken we dat de browser weg is
            if image is None:
                image = placeholder_frame(self.error or self.status)

            ok, buffer = cv2.imencode(
                ".jpg", fit_to_stream_size(image),
                [int(cv2.IMWRITE_JPEG_QUALITY), STREAM_QUALITY])
            if ok:
                yield buffer.tobytes()

    # -- berekening ----------------------------------------------------------

    def start(self):
        if not USE_CAMERA:
            return
        threading.Thread(target=self._run, daemon=True).start()
        threading.Thread(target=self._run_markers, daemon=True).start()

    def _run_markers(self):
        """Bekijk elk binnenkomend beeld op ArUco-markers.

        Dit staat los van de segmentatie, die maar een paar keer per seconde
        rekent: de noodstop moet afgaan bij het eerste beeld waarop de marker
        te zien is, ook terwijl het YOLO-model nog aan het laden is, en
        ongeacht hoe klein of ver de marker in beeld staat. Wie het pad volgt
        gebruikt dezelfde thread om te weten dat er een marker in zicht komt."""
        if cv2 is None or np is None:
            return

        last_seq = -1
        while True:
            with self._marker_cond:
                while self._marker_seq == last_seq or self._marker_frame is None:
                    self._marker_cond.wait()
                image, last_seq = self._marker_frame, self._marker_seq

            try:
                markers = detect_markers(image)
                marker = estop_marker(markers)
            except Exception as exc:
                print("Markercontrole mislukt: " + str(exc), flush=True)
                time.sleep(SEGMENTATION_INTERVAL)
                continue

            # De segmentatie werkt op hetzelfde beeld en hoeft straks niet
            # opnieuw te zoeken
            self.put_markers(last_seq, markers)

            # Wie een richtingsmarker wil naderen, stuurt op deze thread: die
            # loopt op het tempo van de camera, de segmentatie veel trager
            self._publish_aims(image, markers)

            if markers:
                self.marker_id = markers[0]["id"]
                self.marker_at = time.monotonic()

            if marker is not None:
                self.trigger_estop(marker)

    def _run(self):
        self.status = "model laden"
        try:
            if cv2 is None or np is None:
                raise RuntimeError("opencv of numpy ontbreekt")
            # Ultralytics importeren duurt een tiental seconden
            from ultralytics import YOLO
        except Exception as exc:
            self.status = "fout"
            self.error = str(exc)
            print("Segmentatie niet beschikbaar: " + str(exc), flush=True)
            return

        model = None
        loaded_name = None

        while True:
            # Een ander model gekozen op de webpagina? Dan laden we dat nu
            wanted = self.model_name
            if wanted != loaded_name:
                self.status = "model laden"
                loaded_name = wanted
                try:
                    model = YOLO(os.path.join(MODEL_DIR, wanted), verbose=False)
                    self.status = "wachten op beeld"
                    self.error = None
                    print("Model geladen: " + wanted, flush=True)
                except Exception as exc:
                    model = None
                    self.status = "fout"
                    self.error = str(exc)
                    print("Model laden mislukt: " + str(exc), flush=True)

            # Zonder model wachten we tot er een ander gekozen wordt
            if model is None:
                time.sleep(SEGMENTATION_INTERVAL)
                continue

            image, seq = self.take_frame(FRAME_WAIT_TIMEOUT)
            if image is None:
                continue

            try:
                heading, overlay = segment_frame(image, model, self.confidence)

                # De grootste ArUco-marker krijgt een kader op hetzelfde beeld,
                # behalve als de noodstopmarker in beeld ligt: die gaat voor,
                # hoe klein hij ook is, en houdt elke andere actie tegen.
                # De markerthread bekeek dit beeld al; enkel als zijn antwoord
                # er nog niet is zoeken we zelf
                markers = self.markers_for(seq)
                if markers is None:
                    markers = detect_markers(image)
                marker = estop_marker(markers)
                if marker is None and markers:
                    marker = markers[0]

                if marker is not None:
                    draw_marker(overlay, marker)
                    distance = marker_distance(marker["area"])
                    self.marker = {
                        "id": marker["id"],
                        "distance": round(distance, 2) if distance is not None else None,
                        "area": int(round(marker["area"])),
                        "size": int(round(math.sqrt(marker["area"]))),
                    }
                else:
                    self.marker = None

                self.run_marker_command(marker)

                # Wat de robot nu doet, groot in het midden
                command = robot.current_command()
                if command:
                    draw_command(overlay, command,
                                 ESTOP_COLOR if robot.stopped else COMMAND_COLOR)
            except Exception as exc:
                self.status = "fout"
                self.error = str(exc)
                time.sleep(SEGMENTATION_INTERVAL)
                continue

            self._publish_stream(overlay, annotated=True)
            self.heading = None if heading is None else clamp(float(heading), 0.0, 180.0)
            self.updated_at = time.monotonic()
            self.status = "actief"
            self.error = None

    # -- uitlezen ------------------------------------------------------------

    def model_choices(self):
        """Alle modellen om uit te kiezen, met het huidige er zeker bij."""
        return sorted(set(available_models()) | {self.model_name})

    def set_model(self, name):
        """Kies een ander .pt-bestand. De achtergrondthread laadt het daarna.

        Enkel bestandsnamen uit MODEL_DIR zijn toegelaten, zo kan er via de URL
        geen ander bestand van de Jetson geopend worden."""
        name = os.path.basename(str(name))
        if name not in available_models():
            raise ValueError("onbekend model: " + name)

        if name != self.model_name:
            self.model_name = name
            # De oude heading hoort bij het oude model, dus die gooien we weg
            self.heading = None
            self.updated_at = 0.0
        return self.model_name

    def set_confidence(self, value):
        """Zet de drempel waarboven het model een masker meetelt."""
        value = clamp(float(value), min(CONFIDENCE_CHOICES), max(CONFIDENCE_CHOICES))
        self.confidence = round(value, 2)
        return self.confidence

    def marker_in_view(self):
        """Het nummer van de marker die net nog in beeld lag, of None."""
        if not self.marker_at or time.monotonic() - self.marker_at > MARKER_MAX_AGE:
            return None
        return self.marker_id

    def current_heading(self):
        """De laatste heading, of None als er nog geen of enkel een verouderde is."""
        if self.heading is None or not self.updated_at:
            return None
        if time.monotonic() - self.updated_at > HEADING_MAX_AGE:
            return None
        return self.heading

    def snapshot(self):
        """Toestand voor de webpagina."""
        heading = self.current_heading()
        age = time.monotonic() - self.updated_at if self.updated_at else None

        return {
            "status": self.status,
            "error": self.error,
            "heading": round(heading, 1) if heading is not None else None,
            "age": round(age, 1) if age is not None else None,
            "confidence": round(self.confidence, 2),
            "model": self.model_name,
            "marker": self.marker,
            "last_command": self.last_command,
            "command_count": self.command_count,
        }


segmentation = Segmentation()


# ------------------------------------------------------------------- foto's

class PhotoRecorder:
    """Bewaart het ruwe camerabeeld als JPEG in PHOTO_DIR.

    Op de webpagina staat er een knop om er meteen een te nemen, en een
    keuzelijst om er vanzelf om de zoveel tijd een te laten bewaren. De map
    wordt aangemaakt zodra er een eerste foto in moet."""

    def __init__(self):
        self.interval = 0         # s tussen twee foto's; 0 = enkel op de knop
        self.count = 0
        self.last_name = None
        self.error = None
        self._due_at = 0.0
        self._lock = threading.Lock()

    def interval_choices(self):
        """De keuzes voor de lijst op de webpagina."""
        return [{"seconds": seconds, "label": label}
                for seconds, label in PHOTO_INTERVALS]

    def set_interval(self, value):
        """Kies om de hoeveel seconden er vanzelf een foto bewaard wordt.

        Enkel de waarden uit PHOTO_INTERVALS zijn toegelaten; 0 zet het af."""
        try:
            seconds = int(float(value))
        except (TypeError, ValueError):
            raise ValueError("interval moet een getal zijn: %s" % (value,))

        if seconds not in [choice for choice, _ in PHOTO_INTERVALS]:
            raise ValueError("onbekend interval: %s" % (value,))

        with self._lock:
            self.interval = seconds
            # Het wachten begint nu opnieuw, ook als er net een foto genomen is
            self._due_at = time.monotonic() + seconds
        return seconds

    def save_now(self):
        """Bewaar het laatste ruwe beeld en geef de bestandsnaam terug."""
        if cv2 is None:
            raise RuntimeError("opencv ontbreekt")

        image = segmentation.raw_frame()
        if image is None:
            raise RuntimeError("nog geen camerabeeld")

        os.makedirs(PHOTO_DIR, exist_ok=True)
        now = time.time()
        name = "%s_%03d.jpg" % (time.strftime("%Y%m%d_%H%M%S", time.localtime(now)),
                                int(now % 1 * 1000))
        path = os.path.join(PHOTO_DIR, name)

        if not cv2.imwrite(path, image, [int(cv2.IMWRITE_JPEG_QUALITY), PHOTO_QUALITY]):
            raise RuntimeError("bewaren mislukt: " + path)

        with self._lock:
            self.count += 1
            self.last_name = name
            self.error = None
            self._due_at = time.monotonic() + self.interval
        return name

    def start(self):
        threading.Thread(target=self._run, daemon=True).start()

    def _run(self):
        """Neem om de zoveel tijd een foto, zolang er een interval ingesteld is."""
        while True:
            time.sleep(PHOTO_TICK)

            with self._lock:
                due = self.interval and time.monotonic() >= self._due_at
            if not due:
                continue

            try:
                self.save_now()
            except Exception as exc:
                with self._lock:
                    self.error = str(exc)
                    # Niet blijven proberen: pas bij de volgende beurt opnieuw
                    self._due_at = time.monotonic() + self.interval
                print("Foto mislukt: " + str(exc), flush=True)

    def snapshot(self):
        """Toestand voor de webpagina."""
        with self._lock:
            return {
                "interval": self.interval,
                "count": self.count,
                "last": self.last_name,
                "error": self.error,
                "directory": PHOTO_DIR,
            }


photos = PhotoRecorder()


# ------------------------------------------------------------- het pad volgen

class PathFollower:
    """Volgt het pad tot er een reden is om te stoppen.

    Het volgen loopt hier op de Jetson door, niet op de webpagina: die klikt
    het enkel aan en uit. We stoppen zodra er een ArUco-marker in beeld komt,
    als het model FOLLOW_NO_PATH_FRAMES beelden na elkaar geen pad ziet, bij
    een noodstop, en natuurlijk als je opnieuw op de knop klikt of een ander
    commando geeft."""

    def __init__(self):
        self._thread = None
        self._stop = threading.Event()
        self.active = False
        self.reason = None
        self.changes = 0          # zo ziet de webpagina dat er iets veranderd is
        self._no_path_count = 0   # beelden na elkaar zonder pad
        self._last_frame_at = 0.0 # updated_at van het beeld dat we al beoordeeld hebben
        self._aim_id = None       # de richtingsmarker waarop we nu uitlijnen
        self._arrived_count = 0   # beelden na elkaar op de ingestelde afstand

    def toggle(self):
        """Aan- of uitzetten, wat de knop 'volg pad' doet."""
        if self.active:
            self.stop("op de knop geklikt")
            return {"following": False}
        return self.start()

    def start(self):
        if self.active:
            return {"following": True}

        robot.resume()            # een noodstop van daarnet mag dit niet tegenhouden
        self._stop.clear()
        self.active = True
        self.reason = None
        self.changes += 1
        self._no_path_count = 0   # elke nieuwe rit begint met een schone teller
        self._last_frame_at = 0.0
        self._aim_id = None
        self._arrived_count = 0

        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        print("Pad volgen gestart", flush=True)
        return {"following": True}

    def stop(self, reason):
        """Zet het volgen stil. Doet niets als er niet gevolgd wordt."""
        if not self.active:
            return

        self.active = False
        self.reason = reason
        self.changes += 1
        self._stop.set()
        print("Pad volgen gestopt: " + reason, flush=True)

    def pause_for_turn(self):
        """Zet het volgen stil voor een draairegel.

        Geeft terug of het volgen aanstond. Was dat zo, dan hoort de beller het
        na de draai weer aan te zetten; stond het af, dan blijft het af.

        We wachten tot de volglus echt uitgebold is: die stuurt bij het
        afsluiten nog een move(0,0,0), en dat moet gebeurd zijn voor het
        draaien begint. Anders stuurt de volger nog een stapcommando midden in
        de draai."""
        if not self.active:
            return False

        self.stop("draairegel")
        self._aim_id = None
        self._arrived_count = 0
        thread = self._thread
        if thread is not None:
            thread.join(timeout=FOLLOW_INTERVAL * 4)
        return True

    def aligning_id(self):
        """De richtingsmarker waarop we nu uitlijnen, of None.

        De segmentatie leest dit om te weten dat zij het draaien niet hoeft te
        regelen zolang de volger er zelf naartoe rijdt."""
        return self._aim_id

    def _aim(self):
        """De richtingsmarker waar we naartoe rijden, of None voor het pad.

        Eenmaal gekozen blijven we op dezelfde marker mikken tot hij uit beeld
        verdwijnt of zijn regel uitgevoerd is: andere richtingsmarkers tellen
        tijdens het uitlijnen niet mee."""
        if self._aim_id is not None:
            aim = segmentation.aim_for(self._aim_id)
            if aim is not None:
                return aim

            # Uit beeld: terug het pad volgen, met schone tellers
            print("Aruco %d uit beeld, terug naar het pad" % self._aim_id,
                  flush=True)
            self._aim_id = None
            self._arrived_count = 0
            self._no_path_count = 0
            self._last_frame_at = 0.0
            return None

        for aim in segmentation.aims():
            # Net gedraaid voor deze marker? Dan laten we hem met rust, anders
            # mikken we er na het hervatten meteen opnieuw op
            if segmentation.turn_rule_on_cooldown(aim["id"]):
                continue

            self._aim_id = aim["id"]
            self._arrived_count = 0
            print("Uitlijnen op aruco %d" % aim["id"], flush=True)
            return aim

        return None

    def _arrived(self, aim):
        """De regel van deze marker als we er zijn, anders None.

        We slaan toe zodra de marker binnen de bovengrens van de marge komt,
        niet pas als hij precies in het bandje ligt: de robot rijdt er recht
        naartoe, dus de afstand loopt enkel terug, en een smal bandje zou
        tussen twee beelden door kunnen wegvallen.

        Een paar beelden na elkaar, want de afstand komt uit de oppervlakte
        van de marker in beeld en die schommelt."""
        rule = turn_rules.get(aim["id"])
        distance = aim["distance"]
        if rule is None or distance is None or distance > rule_range(rule)[1]:
            self._arrived_count = 0
            return None

        self._arrived_count += 1
        if self._arrived_count < ARUCO_TURN_FRAMES:
            return None
        return rule

    def _start_turn(self, marker_id, rule):
        """We zijn er: stilzetten en de draairegel laten uitvoeren.

        Dit loopt in de volgthread zelf, dus pause_for_turn kan hier niet:
        die wacht op deze thread. We zetten onszelf stil, sturen de robot
        eerst echt naar nul en laten het draaien in een eigen thread lopen.
        Die zet het volgen daarna weer aan."""
        print("Aruco %d op afstand: draai %s" % (marker_id, rule["direction"]),
              flush=True)
        segmentation.note_turn_rule(marker_id)
        self.stop("draairegel")
        self._aim_id = None
        self._arrived_count = 0
        try:
            robot.move(x=0, y=0, z=0)
        except Exception:
            pass
        threading.Thread(target=segmentation.turn_and_resume,
                         args=(rule, True), daemon=True).start()

    def _run(self):
        try:
            while not self._stop.is_set():
                aim = self._aim()

                reason = self._reason_to_stop(aim)
                if reason is not None:
                    self.stop(reason)
                    break

                if aim is not None:
                    rule = self._arrived(aim)
                    if rule is not None:
                        self._start_turn(aim["id"], rule)
                        break
                    robot.note_command("naar aruco %d" % aim["id"])
                    robot.steer_to(aim["heading"])
                else:
                    robot.note_command("volg pad")
                    robot.follow_path()

                self._stop.wait(FOLLOW_INTERVAL)
        except Exception as exc:
            self.stop("fout: " + str(exc))

        # Altijd afsluiten met een stop, behalve als de noodstop dat al deed
        try:
            if not robot.stopped:
                robot.move(x=0, y=0, z=0)
        except Exception:
            pass

    def _reason_to_stop(self, aim=None):
        """Waarom moeten we stoppen? None als we gewoon verder mogen."""
        if not robot.connected:
            return "geen verbinding"
        if robot.stopped:
            return "noodstop"

        # Tijdens het uitlijnen mikken we op de marker en overrulen we de
        # segmentatie: of er een pad ligt doet dan niet ter zake
        if aim is not None:
            return None

        #marker_id = segmentation.marker_in_view()
        #if marker_id is not None:
        #    return "aruco %d" % marker_id

        return self._no_path_reason()

    def _no_path_reason(self):
        """Stoppen omdat er geen pad meer is? None als we verder mogen.

        Een enkel beeld zonder pad zegt niet veel: het model mist er wel eens
        een, en dan stopt de robot midden op een pad dat er gewoon ligt. We
        stoppen daarom pas na FOLLOW_NO_PATH_FRAMES beelden na elkaar zonder
        pad. Ondertussen stapt follow_path rechtdoor.

        We tellen beelden, geen lusrondes: deze lus draait op FOLLOW_INTERVAL
        en is daarmee sneller dan de segmentatie, dus hetzelfde beeld zou
        anders meermaals meetellen. Een nieuw beeld herkennen we aan
        updated_at, dat elke ronde van de segmentatie opnieuw gezet wordt.

        Valt de segmentatie helemaal stil, dan komt er ook geen nieuw beeld
        meer om op te tellen. Daar helpt wachten niet, dus stoppen we meteen:
        anders zou de robot blijven doorstappen op een verouderd beeld."""
        updated_at = segmentation.updated_at
        if not updated_at or time.monotonic() - updated_at > HEADING_MAX_AGE:
            return "geen beeld"

        # Enkel een beeld dat we nog niet bekeken hebben telt mee
        if updated_at != self._last_frame_at:
            self._last_frame_at = updated_at
            if segmentation.current_heading() is None:
                self._no_path_count += 1
            else:
                self._no_path_count = 0

        if self._no_path_count >= FOLLOW_NO_PATH_FRAMES:
            return "geen pad op %d beelden" % self._no_path_count
        return None


path_follower = PathFollower()

# ------------------------------------------------------------- route via GPS

def freewalk_get(path, timeout=3):
    """JSON ophalen bij de stayontrails-app (zelfondertekend certificaat)."""
    context = ssl.create_default_context()
    context.check_hostname = False
    context.verify_mode = ssl.CERT_NONE
    with urllib.request.urlopen(FREEWALK_URL.rstrip("/") + path,
                                timeout=timeout, context=context) as resp:
        return json.loads(resp.read().decode("utf-8"))


def haversine_m(lat1, lon1, lat2, lon2):
    p1, p2 = math.radians(lat1), math.radians(lat2)
    a = (math.sin((p2 - p1) / 2) ** 2 +
         math.cos(p1) * math.cos(p2) * math.sin(math.radians(lon2 - lon1) / 2) ** 2)
    return 2 * 6371000.0 * math.asin(math.sqrt(a))


def bearing_deg(lat1, lon1, lat2, lon2):
    """Kompasrichting (0 = noord, 90 = oost) van punt 1 naar punt 2."""
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dl = math.radians(lon2 - lon1)
    y = math.sin(dl) * math.cos(p2)
    x = math.cos(p1) * math.sin(p2) - math.sin(p1) * math.cos(p2) * math.cos(dl)
    return (math.degrees(math.atan2(y, x)) + 360) % 360


def angle_diff(target, current):
    """Kleinste hoek van current naar target, -180..180 (positief = rechts)."""
    return (target - current + 180) % 360 - 180


class RouteFollower:
    """Loopt een route uit stayontrails af met de GPS van de gsm.

    De camera-volger (path_follower) blijft het pad volgen; deze klasse houdt
    de GPS in het oog: ze zet het actieve routepunt door als we binnen de
    aankomststraal komen, stopt aan het einde of als de GPS wegvalt, en stuurt
    bij als de gelopen richting duidelijk niet naar het volgende punt wijst.
    Elk ander commando, een noodstop of het stoppen van de camera-volger zet
    de route ook stil."""

    def __init__(self):
        self._thread = None
        self._stop = threading.Event()
        self.active = False
        self.reason = None
        self.changes = 0
        self.slug = None
        self.name = None
        self.points = []
        self.radius = 3.0
        self.index = 0
        self.distance = None
        self.bearing = None
        self.course = None
        self.accuracy = None
        self._track = []              # (monotonic, lat, lon) sinds de laatste correctie
        self._last_correction = 0.0

    # -- starten en stoppen --------------------------------------------------

    def start(self, slug):
        if self.active:
            return {"ok": False, "error": "er loopt al een route"}
        if not robot.connected:
            return {"ok": False, "error": "geen verbinding met de robot"}

        try:
            payload = freewalk_get("/?action=load_path&slug=" + urllib.parse.quote(slug))
        except Exception as exc:
            return {"ok": False, "error": "route laden mislukt: %s" % exc}
        if not payload.get("ok"):
            return {"ok": False, "error": payload.get("error", "route niet gevonden")}

        path = payload["path"]
        points = [(float(p["lat"]), float(p["lng"]), p.get("instruction", ""))
                  for p in path.get("points", [])]
        if len(points) < 2:
            return {"ok": False, "error": "de route heeft te weinig punten"}

        # De camera-volger heeft een vers segmentatiebeeld nodig; zonder zou de
        # route meteen stoppen met een vage melding. Zeg dus wat er schort.
        camera = segmentation.snapshot()
        if camera["status"] != "actief" or camera["age"] is None or camera["age"] > HEADING_MAX_AGE:
            reason = {
                "model laden": "het model wordt nog geladen, wacht even",
                "wachten op beeld": "er komt geen camerabeeld van de robot binnen",
                "fout": "segmentatiefout: %s" % camera["error"],
            }.get(camera["status"], "geen vers camerabeeld (status: %s)" % camera["status"])
            return {"ok": False, "error": "camera niet klaar: " + reason}

        gps = self._gps()
        if gps is None:
            return {"ok": False, "error": "geen verse GPS van de gsm: open de stayontrails-pagina"}

        self.slug = path.get("slug", slug)
        self.name = path.get("name", slug)
        self.points = points
        self.radius = float(path.get("arrivalRadiusMeters") or 3)
        self.index = 0
        self.distance = self.bearing = self.course = None
        self._track = []
        self._last_correction = 0.0
        self.reason = None
        self.active = True
        self.changes += 1
        self._stop.clear()

        path_follower.start()         # de camera loopt mee op het pad
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        print("Route gestart: %s (%d punten)" % (self.name, len(points)), flush=True)
        return {"ok": True, "route": self.name, "points": len(points)}

    def stop(self, reason):
        if not self.active:
            return
        self.active = False
        self.reason = reason
        self.changes += 1
        self._stop.set()
        print("Route gestopt: " + reason, flush=True)
        path_follower.stop("route: " + reason)

    # -- GPS -----------------------------------------------------------------

    def _gps(self):
        """De verse GPS van de gsm, of None."""
        try:
            data = freewalk_get("/api/gps")
        except Exception:
            return None
        if not data.get("fix") or data.get("age_s", 99) > ROUTE_GPS_MAX_AGE:
            return None
        return data

    # -- lus -----------------------------------------------------------------

    def _run(self):
        gps_lost_since = None
        try:
            while not self._stop.is_set():
                reason = self._reason_to_stop()
                if reason is not None:
                    self.stop(reason)
                    return

                gps = self._gps()
                if gps is None:
                    gps_lost_since = gps_lost_since or time.monotonic()
                    if time.monotonic() - gps_lost_since > ROUTE_GPS_MAX_AGE:
                        self.stop("GPS weggevallen")
                        return
                else:
                    gps_lost_since = None
                    self._update(gps)

                self._stop.wait(ROUTE_POLL_INTERVAL)
        except Exception as exc:
            self.stop("fout: " + str(exc))

    def _reason_to_stop(self):
        if robot.stopped:
            return "noodstop"
        if not robot.connected:
            return "geen verbinding"
        # Het volgen stond even stil voor een draairegel? Dan wachten we.
        if not path_follower.active and path_follower.reason != "draairegel":
            return "camera-volger gestopt (%s)" % path_follower.reason
        return None

    def _update(self, gps):
        lat, lon = gps["lat"], gps["lon"]
        self.accuracy = gps.get("accuracy_m")
        target_lat, target_lon, _ = self.points[self.index]
        self.distance = haversine_m(lat, lon, target_lat, target_lon)
        self.bearing = bearing_deg(lat, lon, target_lat, target_lon)

        reliable = self.accuracy is None or self.accuracy <= ROUTE_MAX_ACCURACY
        if reliable and self.distance <= self.radius:
            self.index += 1
            self.changes += 1
            self._track = []
            self.course = None
            print("Routepunt %d bereikt" % self.index, flush=True)
            if self.index >= len(self.points):
                self.stop("bestemming bereikt")
                robot.note_command("route klaar")
                try:
                    robot.move(x=0, y=0, z=0)
                except Exception:
                    pass
            return

        if reliable:
            self._track.append((time.monotonic(), lat, lon))
            self._track = self._track[-30:]
        self._steer()

    def _steer(self):
        """Bijsturen als we al een tijd de verkeerde kant op lopen."""
        if not ROUTE_GPS_STEERING or len(self._track) < 2:
            return
        _, lat0, lon0 = self._track[0]
        _, lat1, lon1 = self._track[-1]
        if haversine_m(lat0, lon0, lat1, lon1) < ROUTE_MIN_COURSE_M:
            return                      # te weinig verplaatst: GPS-ruis

        self.course = bearing_deg(lat0, lon0, lat1, lon1)
        error = angle_diff(self.bearing, self.course)
        if abs(error) < ROUTE_TURN_THRESHOLD:
            return
        if time.monotonic() - self._last_correction < ROUTE_CORRECTION_COOLDOWN:
            return

        degrees = clamp(error, -ROUTE_MAX_CORRECTION, ROUTE_MAX_CORRECTION)
        print("Route: koers %.0f, doel %.0f, stuur %+.0f graden bij" %
              (self.course, self.bearing, degrees), flush=True)
        self._last_correction = time.monotonic()
        self._track = []                # opnieuw een looprichting opbouwen
        self.course = None

        # Even de camera-volger stilzetten, draaien, en weer aanzetten
        resume = path_follower.pause_for_turn()
        try:
            robot.follow_heading(degrees)
        finally:
            if resume and self.active and not robot.stopped:
                path_follower.start()

    # -- voor de webpagina ---------------------------------------------------

    def snapshot(self):
        return {
            "active": self.active,
            "reason": self.reason,
            "changes": self.changes,
            "name": self.name,
            "point": min(self.index + 1, len(self.points)) if self.points else None,
            "points": len(self.points),
            "distance": None if self.distance is None else round(self.distance, 1),
            "bearing": None if self.bearing is None else round(self.bearing),
            "course": None if self.course is None else round(self.course),
            "accuracy": self.accuracy,
        }


route_follower = RouteFollower()



# -------------------------------------------------------------- commandotabel

COMMANDS = {
    # de basis
    "forward":      lambda: robot.move(x=MOVE_SPEED),
    "follow":       lambda: path_follower.toggle(),
    "backward":     lambda: robot.move(x=-MOVE_SPEED),
    "turn_left":    lambda: robot.move(z=TURN_SPEED),
    "turn_right":   lambda: robot.move(z=-TURN_SPEED),
    "strafe_left":  lambda: robot.move(y=MOVE_SPEED),
    "strafe_right": lambda: robot.move(y=-MOVE_SPEED),
    "stop":         lambda: robot.move(x=0, y=0, z=0),

    "stand_up":     lambda: robot.sport(SPORT_CMD["RecoveryStand"], {"data": False}),
    "stand_down":   lambda: robot.sport(SPORT_CMD["StandDown"]),

    # extra bewegingen
    "hello":        lambda: robot.sport(SPORT_CMD["Hello"]),
    "stretch":      lambda: robot.sport(SPORT_CMD["Stretch"], {"data": False}),
    "sit":          lambda: robot.sport(1009, {"data": False}),
    "rise_sit":     lambda: robot.sport(1010, {"data": False}),
    "scrape":       lambda: robot.sport(1029, {"data": False}),

    # licht
    "torch_on":     lambda: robot.vui(1005, {"brightness": BRIGHTNESS_LVL}),
    "torch_off":    lambda: robot.vui(1005, {"brightness": 0}),
    "white":        lambda: robot.colour(VUI_COLOR.WHITE),
    "red":          lambda: robot.colour(VUI_COLOR.RED),
    "yellow":       lambda: robot.colour(VUI_COLOR.YELLOW),
    "green":        lambda: robot.colour(VUI_COLOR.GREEN),
    "cyan":         lambda: robot.colour(VUI_COLOR.CYAN),
    "blue":         lambda: robot.colour(VUI_COLOR.BLUE),
    "purple":       lambda: robot.colour(VUI_COLOR.PURPLE),
    "disco":        lambda: robot.disco(),
    "flash":        lambda: robot.vui(1007, {"color": VUI_COLOR.WHITE, "flash_cycle": 500}),

    # noodstop
    "estop":        lambda: robot.emergency_stop(),
}


# De koppelingen kennen COMMANDS, dus die maken we hier pas aan. De vaste
# commando's eerst: de draairegels controleren erop of een marker al bezet is.
command_markers = CommandMarkers(ARUCO_MARKERS_PATH, ARUCO_COMMANDS)
turn_rules = TurnRules(ARUCO_RULES_PATH)

for _rule in turn_rules.all():
    _command = command_markers.get_command(_rule["id"])
    if _command is not None:
        print("Let op: marker %d heeft zowel een draairegel als het commando %s; "
              "het commando gaat voor" % (_rule["id"], _command), flush=True)


# ------------------------------------------------------------------ webpagina

PAGE_CSS = """
    :root {
      --bg: #111418;
      --panel: #1c2128;
      --panel-2: #262d36;
      --line: #333b45;
      --text: #f2f4f7;
      --muted: #9aa4b2;
      --accent: #2f81f7;
      --danger: #e5484d;
      --ok: #2ea043;
    }
    * { box-sizing: border-box; -webkit-tap-highlight-color: transparent; }
    body {
      margin: 0;
      padding: 12px 12px calc(12px + env(safe-area-inset-bottom));
      background: var(--bg);
      color: var(--text);
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Arial, sans-serif;
      font-size: 15px;
      user-select: none;
      -webkit-user-select: none;
      touch-action: manipulation;
      overscroll-behavior: none;
    }
    header {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 10px;
      margin-bottom: 12px;
    }
    h1 { font-size: 17px; margin: 0; }
    .status {
      font-size: 12px;
      color: var(--muted);
      display: flex;
      align-items: center;
      gap: 6px;
      min-width: 0;
    }
    #msg { overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
    .dot {
      width: 9px; height: 9px; border-radius: 50%;
      background: var(--muted);
      flex: none;
    }
    .dot.ok { background: var(--ok); }
    .dot.err { background: var(--danger); }

    section {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 14px;
      padding: 12px;
      margin-bottom: 12px;
    }
    h2 {
      font-size: 12px;
      letter-spacing: .08em;
      text-transform: uppercase;
      color: var(--muted);
      margin: 0 0 10px;
    }

    button {
      font: inherit;
      color: var(--text);
      background: var(--panel-2);
      border: 1px solid var(--line);
      border-radius: 12px;
      padding: 12px 6px;
      min-height: 58px;
      width: 100%;
      display: flex;
      flex-direction: column;
      align-items: center;
      justify-content: center;
      gap: 3px;
      cursor: pointer;
      transition: transform .05s ease, background .1s ease;
    }
    button .ico { font-size: 20px; line-height: 1; }
    button .lbl { font-size: 12px; color: var(--muted); text-align: center; }
    button .aruco {
      font-size: 10px;
      color: #ff7bff;
      letter-spacing: .04em;
    }
    button:active { background: var(--accent); transform: scale(.96); }
    button:active .lbl { color: #fff; }

    .grid { display: grid; gap: 8px; }
    .cols-2 { grid-template-columns: repeat(2, 1fr); }
    .cols-3 { grid-template-columns: repeat(3, 1fr); }
    .cols-4 { grid-template-columns: repeat(4, 1fr); }

    .dpad { display: grid; grid-template-columns: repeat(3, 1fr); gap: 8px; }
    .dpad button { min-height: 66px; }
    .dpad .stop { background: #3a2224; border-color: #5a2c30; }
    .dpad button.bezig {
      background: var(--accent);
      border-color: var(--accent);
      animation: pulse 1.2s ease-in-out infinite;
    }
    .dpad button.bezig .lbl { color: #fff; }
    @keyframes pulse { 50% { opacity: .65; } }

    .estop {
      background: var(--danger);
      border-color: var(--danger);
      min-height: 62px;
      font-weight: 700;
      letter-spacing: .04em;
    }
    .estop .lbl { color: #ffe9ea; font-size: 15px; font-weight: 700; }
    .estop:active { background: #ff5a5f; }

    .swatch { min-height: 52px; }
    .swatch .ico { font-size: 22px; }

    .row { display: grid; grid-template-columns: 1fr auto; gap: 8px; align-items: stretch; }
    .row button { width: auto; padding: 12px 22px; }
    input[type=number] {
      font: inherit;
      user-select: text;
      -webkit-user-select: text;
      font-size: 20px;
      text-align: center;
      color: var(--text);
      background: var(--panel-2);
      border: 1px solid var(--line);
      border-radius: 12px;
      padding: 12px;
      min-height: 58px;
      width: 100%;
    }

    .check {
      display: flex;
      align-items: center;
      gap: 10px;
      margin-top: 10px;
      font-size: 13px;
      color: var(--muted);
    }
    .check input[type=checkbox] { width: 22px; height: 22px; accent-color: var(--accent); }
    .check select {
      max-width: 60%;
      font: inherit;
      color: var(--text);
      background: var(--panel-2);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 6px 8px;
      min-height: 34px;
      margin-left: auto;
    }
    .check b { color: var(--text); font-variant-numeric: tabular-nums; }

    .camera {
      position: relative;
      width: 100%;
      max-width: 640px;        /* 640x480, op een gsm schaalt het mee naar beneden */
      aspect-ratio: 4 / 3;
      margin: 0 auto;
      background: #000;
      border: 1px solid var(--line);
      border-radius: 10px;
      overflow: hidden;
    }
    .camera.verborgen { display: none; }
    .camera img { width: 100%; height: 100%; object-fit: contain; display: block; }
    .camera .badge {
      position: absolute;
      left: 8px; bottom: 8px;
      background: rgba(17, 20, 24, .75);
      border-radius: 8px;
      padding: 3px 8px;
      font-size: 11px;
      color: var(--muted);
      font-variant-numeric: tabular-nums;
    }

    .hint { font-size: 11px; color: var(--muted); margin: 8px 2px 0; }
    .hint code { background: var(--panel-2); padding: 1px 5px; border-radius: 5px; }
    footer { text-align: center; color: var(--muted); font-size: 11px; padding: 4px 0 8px; }
    a.terug { color: var(--accent); text-decoration: none; font-size: 13px; line-height: 2; }
"""


HTML_PAGE = """
<!doctype html>
<html lang="nl">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1, maximum-scale=1, user-scalable=no, viewport-fit=cover">
  <meta name="theme-color" content="#111418">
  <meta name="mobile-web-app-capable" content="yes">
  <title>Go2 bediening</title>
  <style>
""" + PAGE_CSS + """  </style>
</head>
<body>

<header>
  <h1>🐕 Go2 bediening</h1>
  <div class="status"><span class="dot" id="dot"></span><span id="msg">klaar</span></div>
</header>

<section>
  <h2>Noodstop</h2>
  <button class="estop" data-cmd="estop"><span class="lbl">⛔ EMERGENCY STOP</span></button>
</section>

<section>
  <h2>Camera</h2>
  <div class="camera" id="cam-box">
    <img id="cam" alt="camerabeeld">
    <span class="badge" id="cam-badge">—</span>
  </div>
  <label class="check">
    <input type="checkbox" id="show-camera" checked>
    <span>toon het beeld</span>
  </label>
</section>

<section>
  <h2>Bewegen — houd ingedrukt</h2>
  <div class="dpad">
    <button data-cmd="strafe_left" data-hold><span class="ico">⬅️</span><span class="lbl">zijwaarts</span></button>
    <button data-cmd="forward" data-follow data-hold><span class="ico">⬆️</span><span class="lbl">vooruit</span></button>
    <button data-cmd="strafe_right" data-hold><span class="ico">➡️</span><span class="lbl">zijwaarts</span></button>

    <button data-cmd="turn_left" data-hold><span class="ico">↩️</span><span class="lbl">draai links</span></button>
    <button class="stop" data-cmd="stop"><span class="ico">⏹️</span><span class="lbl">stop</span></button>
    <button data-cmd="turn_right" data-hold><span class="ico">↪️</span><span class="lbl">draai rechts</span></button>

    <div></div>
    <button data-cmd="backward" data-hold><span class="ico">⬇️</span><span class="lbl">achteruit</span></button>
    <div></div>
  </div>
</section>

<section>
  <h2>Heading</h2>
  <div class="row">
    <input id="heading" type="number" inputmode="decimal" step="1" value="90">
    <button id="heading-go"><span class="ico">🧭</span><span class="lbl">draai</span></button>
  </div>
  <label class="check">
    <input type="checkbox" id="auto-heading" checked>
    <span>volg de camera <b id="camera-heading">—</b></span>
  </label>
</section>

<section>
  <h2>Route (GPS)</h2>
  <div class="row">
    <select id="route-select"></select>
    <button id="route-go"><span class="ico">🗺️</span><span class="lbl">volg route</span></button>
  </div>
  <div id="route-info">—</div>
</section>

<section>
  <h2>Houding</h2>
  <div class="grid cols-2">
    <button data-cmd="stand_up"><span class="ico">🦮</span><span class="lbl">opstaan</span></button>
    <button data-cmd="stand_down"><span class="ico">🛌</span><span class="lbl">neerliggen</span></button>
  </div>
</section>

<section>
  <h2>Bewegingen</h2>
  <div class="grid cols-3">
    <button data-cmd="hello"><span class="ico">👋</span><span class="lbl">hello</span></button>
    <button data-cmd="stretch"><span class="ico">🤸</span><span class="lbl">stretch</span></button>
    <button data-cmd="sit"><span class="ico">🪑</span><span class="lbl">zitten</span></button>
    <button data-cmd="rise_sit"><span class="ico">🚶</span><span class="lbl">rechtstaan</span></button>
  </div>
</section>

<section>
  <h2>Zaklamp</h2>
  <div class="grid cols-2">
    <button data-cmd="torch_on"><span class="ico">🔦</span><span class="lbl">aan</span></button>
    <button data-cmd="torch_off"><span class="ico">🌑</span><span class="lbl">uit</span></button>
  </div>
</section>

<section>
  <h2>Kleuren</h2>
  <div class="grid cols-4">
    <button class="swatch" data-cmd="white"><span class="ico">🤍</span></button>
    <button class="swatch" data-cmd="red"><span class="ico">❤️</span></button>
    <button class="swatch" data-cmd="yellow"><span class="ico">💛</span></button>
    <button class="swatch" data-cmd="green"><span class="ico">💚</span></button>
    <button class="swatch" data-cmd="cyan"><span class="ico">🩵</span></button>
    <button class="swatch" data-cmd="blue"><span class="ico">💙</span></button>
    <button class="swatch" data-cmd="purple"><span class="ico">💜</span></button>
    <button class="swatch" data-cmd="disco"><span class="ico">🪩</span></button>
  </div>
  <div class="grid" style="margin-top:8px">
    <button data-cmd="flash"><span class="ico">⚡</span><span class="lbl">flash</span></button>
  </div>
</section>

<section>
  <h2>Foto's</h2>
  <label class="check">
    <span>foto om de</span>
    <select id="photo-interval">
      {% for choice in photo_intervals %}
      <option value="{{ choice.seconds }}"{% if choice.seconds == photo_interval %} selected{% endif %}>{{ choice.label }}</option>
      {% endfor %}
    </select>
  </label>
  <div class="row" style="margin-top:10px">
    <span class="check" style="margin-top:0">opgenomen beelden <b id="photo-count">0</b></span>
    <button id="photo-go"><span class="ico">&#128247;</span><span class="lbl">foto</span></button>
  </div>
</section>

<section>
  <h2>Model</h2>
  <label class="check">
    <span>model</span>
    <select id="model">
      {% for name in model_choices %}
      <option value="{{ name }}"{% if name == model %} selected{% endif %}>{{ name }}</option>
      {% endfor %}
    </select>
  </label>
  <label class="check">
    <span>confidence</span>
    <select id="confidence">
      {% for value in confidence_choices %}
      <option value="{{ value }}"{% if value == confidence %} selected{% endif %}>{{ "%.1f"|format(value) }}</option>
      {% endfor %}
    </select>
  </label>
</section>

<footer><a class="terug" href="/aruco">🎯 draairegels van de markers</a><br>Robot: {{ robot_ip }}</footer>

<script>
const dot = document.getElementById('dot');
const msg = document.getElementById('msg');
const HOLD_INTERVAL = 150;   // ms tussen twee move-commando's tijdens indrukken

function setStatus(text, state) {
  msg.textContent = text;
  dot.className = 'dot' + (state ? ' ' + state : '');
}

async function send(cmd) {
  try {
    const res = await fetch('/cmd/' + cmd, { method: 'POST' });
    const data = await res.json();
    setStatus(data.ok ? cmd : (data.error || 'fout'), data.ok ? 'ok' : 'err');
  } catch (err) {
    setStatus('geen verbinding', 'err');
  }
}

// heading: draai naar een hoek (90 = vooruit)
document.getElementById('heading-go').addEventListener('click', async () => {
  const value = document.getElementById('heading').value;
  setStatus('draaien naar ' + value + '...');
  try {
    const res = await fetch('/heading/?heading=' + encodeURIComponent(value), { method: 'POST' });
    const data = await res.json();
    if (data.ok) {
      setStatus(data.turned + '° ' + data.direction, 'ok');
    } else {
      setStatus(data.error || 'fout', 'err');
    }
  } catch (err) {
    setStatus('geen verbinding', 'err');
  }
});

// gewone knoppen: één commando per tik
document.querySelectorAll('button[data-cmd]:not([data-hold])').forEach(btn => {
  btn.addEventListener('click', () => send(btn.dataset.cmd));
});

// Staat "volg de camera" aan, dan is de vooruitknop een schakelaar in plaats
// van een houdknop: het volgen loopt dan op de robot zelf door
function isFollowToggle(btn) {
  return btn.dataset.follow !== undefined && autoHeading.checked;
}

// houd-knoppen: herhaal het commando tot je loslaat, daarna stoppen
document.querySelectorAll('button[data-hold]').forEach(btn => {
  let timer = null;

  const start = (ev) => {
    if (isFollowToggle(btn)) return;   // dan doet de klik hieronder het werk
    ev.preventDefault();
    if (timer) return;
    send(btn.dataset.cmd);
    timer = setInterval(() => send(btn.dataset.cmd), HOLD_INTERVAL);
  };

  const end = () => {
    if (!timer) return;
    clearInterval(timer);
    timer = null;
    send('stop');
  };

  btn.addEventListener('pointerdown', start);
  btn.addEventListener('pointerup', end);
  btn.addEventListener('pointercancel', end);
  btn.addEventListener('pointerleave', end);
  btn.addEventListener('contextmenu', ev => ev.preventDefault());
});

// Stoppen wanneer de pagina naar de achtergrond gaat of je wegsurft. Volgt de
// robot een pad, dan laten we hem doorgaan: dat is net de bedoeling van de modus.
document.addEventListener('visibilitychange', () => {
  if (document.hidden && !following) send('stop');
});
window.addEventListener('pagehide', () => { if (!following) send('stop'); });

// verbindingstoestand + heading van de camera ophalen
const headingInput = document.getElementById('heading');
const autoHeading = document.getElementById('auto-heading');
const cameraHeading = document.getElementById('camera-heading');
const followButton = document.querySelector('button[data-follow]');

// Toon op de knop zelf of hij gewoon vooruit gaat of het pad volgt
let following = false;
let lastFollowChanges = null;

function showForwardMode() {
  const lbl = followButton.querySelector('.lbl');
  if (!autoHeading.checked) {
    lbl.textContent = 'vooruit';
  } else {
    lbl.textContent = following ? 'volgt pad…' : 'volg pad';
  }
  followButton.classList.toggle('bezig', following);
}

autoHeading.addEventListener('change', () => {
  if (following) send('stop');       // van modus wisselen zet het volgen stil
  showForwardMode();
});

// Eén klik zet het volgen aan, nog een klik zet het weer uit
followButton.addEventListener('click', () => {
  if (!isFollowToggle(followButton)) return;
  send('follow');
});

showForwardMode();

// ---- route volgen met de GPS van de gsm
const routeSelect = document.getElementById('route-select');
const routeGo = document.getElementById('route-go');
const routeInfo = document.getElementById('route-info');
let routeActive = false;
let lastRouteChanges = null;

async function loadRoutes() {
  try {
    const res = await fetch('/route/paths');
    const data = await res.json();
    routeSelect.innerHTML = '';
    (data.paths || []).forEach(p => {
      const opt = document.createElement('option');
      opt.value = p.slug;
      opt.textContent = p.name + ' (' + p.pointCount + ')';
      routeSelect.appendChild(opt);
    });
    if (!data.ok) routeInfo.textContent = data.error;
    else if (!(data.paths || []).length) routeInfo.textContent = 'geen routes gevonden';
  } catch (e) {
    routeInfo.textContent = 'routes laden mislukt';
  }
}
loadRoutes();

routeGo.addEventListener('click', async () => {
  const url = routeActive ? '/route/stop' : '/route/start?slug=' + encodeURIComponent(routeSelect.value);
  if (!routeActive && !routeSelect.value) return;
  const res = await fetch(url, { method: 'POST' });
  const data = await res.json();
  if (!data.ok) setStatus(data.error, 'err');
});

function showRoute(r) {
  if (!r) return;
  routeActive = !!r.active;
  routeGo.querySelector('.lbl').textContent = routeActive ? 'stop route' : 'volg route';
  routeGo.classList.toggle('bezig', routeActive);
  if (routeActive) {
    routeInfo.textContent = r.name + ' — punt ' + r.point + '/' + r.points +
      (r.distance !== null ? ', nog ' + r.distance + ' m' : '') +
      (r.accuracy ? ' (GPS ±' + Math.round(r.accuracy) + ' m)' : '');
  } else if (r.reason) {
    routeInfo.textContent = 'route gestopt: ' + r.reason;
  }
  lastRouteChanges = r.changes;
}

function showFollowing(data) {
  following = !!data.following;
  showForwardMode();

  // Meld waarom het volgen gestopt is, maar maar één keer
  if (lastFollowChanges !== null && data.follow_changes !== lastFollowChanges &&
      !data.following && data.follow_reason) {
    setStatus('volgen gestopt: ' + data.follow_reason, 'ok');
  }
  lastFollowChanges = data.follow_changes;
}

// live beeld met het masker erop: gewoon een MJPEG-stream in een <img>
const camImg = document.getElementById('cam');
const camBadge = document.getElementById('cam-badge');
const showCameraBox = document.getElementById('show-camera');
const camBox = document.getElementById('cam-box');

// Staat "toon het beeld" af, dan tonen we ook geen leeg zwart vak. Dit volgt
// enkel de schakelaar: een pagina die even naar de achtergrond gaat stopt wel
// de stream, maar laat het vak staan waar het stond.
function updateCamBox() {
  camBox.classList.toggle('verborgen', !showCameraBox.checked);
}

function startStream() {
  if (!showCameraBox.checked || document.hidden) return;
  if (camImg.getAttribute('src')) return;
  camImg.src = '/video?t=' + Date.now();   // nieuwe url, anders hergebruikt de browser de oude stream
}

function stopStream() {
  if (!camImg.getAttribute('src')) return;
  camImg.removeAttribute('src');           // sluit de verbinding met /video
}

showCameraBox.addEventListener('change', () => {
  updateCamBox();
  if (showCameraBox.checked) startStream(); else stopStream();
});

// De browser kan de schakelaar bij een herlaadbeurt op zijn oude stand zetten
updateCamBox();

// De stream niet laten doorlopen als de pagina toch niet zichtbaar is
document.addEventListener('visibilitychange', () => {
  if (document.hidden) stopStream(); else startStream();
});
// de markers die een knop bedienen: zet hun nummer op die knop
const ARUCO_COMMANDS = {{ aruco_commands|tojson }};

Object.keys(ARUCO_COMMANDS).forEach(markerId => {
  const btn = document.querySelector('button[data-cmd="' + ARUCO_COMMANDS[markerId] + '"]');
  if (!btn) return;
  const tag = document.createElement('span');
  tag.className = 'aruco';
  tag.textContent = 'aruco ' + markerId;
  btn.appendChild(tag);
});

// het model kiezen: de .pt-bestanden uit de modelmap
const modelBox = document.getElementById('model');

modelBox.addEventListener('change', async () => {
  setStatus('model laden...');
  try {
    const res = await fetch('/model/?name=' + encodeURIComponent(modelBox.value), { method: 'POST' });
    const data = await res.json();
    setStatus(data.ok ? data.model : (data.error || 'fout'), data.ok ? 'ok' : 'err');
  } catch (err) {
    setStatus('geen verbinding', 'err');
  }
});

// de drempel waarboven het model een masker meetelt
const confidenceBox = document.getElementById('confidence');

confidenceBox.addEventListener('change', async () => {
  const value = confidenceBox.value;
  try {
    const res = await fetch('/confidence/?value=' + encodeURIComponent(value), { method: 'POST' });
    const data = await res.json();
    setStatus(data.ok ? 'confidence ' + data.confidence : (data.error || 'fout'),
              data.ok ? 'ok' : 'err');
  } catch (err) {
    setStatus('geen verbinding', 'err');
  }
});

// foto's van het ruwe beeld: meteen op de knop of vanzelf om de zoveel tijd
const photoButton = document.getElementById('photo-go');
const photoIntervalBox = document.getElementById('photo-interval');
const photoCount = document.getElementById('photo-count');

photoButton.addEventListener('click', async () => {
  try {
    const res = await fetch('/photo/', { method: 'POST' });
    const data = await res.json();
    if (data.ok) {
      photoCount.textContent = data.count;
      setStatus('foto ' + data.photo, 'ok');
    } else {
      setStatus(data.error || 'fout', 'err');
    }
  } catch (err) {
    setStatus('geen verbinding', 'err');
  }
});

photoIntervalBox.addEventListener('change', async () => {
  const value = photoIntervalBox.value;
  try {
    const res = await fetch('/photo/interval/?value=' + encodeURIComponent(value),
                            { method: 'POST' });
    const data = await res.json();
    setStatus(data.ok ? 'foto om de ' + data.label : (data.error || 'fout'),
              data.ok ? 'ok' : 'err');
  } catch (err) {
    setStatus('geen verbinding', 'err');
  }
});

function showPhotos(info) {
  if (!info) return;
  photoCount.textContent = info.count;

  // De keuzelijst gelijk houden met de server, bv. na een herstart of een URL
  if (document.activeElement !== photoIntervalBox &&
      photoIntervalBox.value !== String(info.interval)) {
    photoIntervalBox.value = info.interval;
  }
}

camImg.addEventListener('error', () => {
  stopStream();
  setTimeout(startStream, 2000);        // opnieuw proberen, bv. na een herstart
});
startStream();

function showCamera(cam) {
  if (!cam) return;

  let badge = cam.heading === null || cam.heading === undefined
    ? cam.status
    : cam.status + ' · ' + cam.heading.toFixed(1) + '°';
  if (cam.marker) {
    badge += ' · aruco ' + cam.marker.id;
    if (cam.marker.distance !== null) badge += ' op ' + cam.marker.distance.toFixed(2) + ' m';
  }
  camBadge.textContent = badge;

  // De keuzelijsten gelijk houden met de server, bv. na een herstart of een URL
  if (cam.confidence !== undefined && document.activeElement !== confidenceBox) {
    confidenceBox.value = cam.confidence.toFixed(1);
  }
  if (cam.model && document.activeElement !== modelBox && modelBox.value !== cam.model) {
    modelBox.value = cam.model;
  }

  if (cam.heading === null || cam.heading === undefined) {
    cameraHeading.textContent = '(' + cam.status + ')';
    return;
  }

  cameraHeading.textContent = cam.heading.toFixed(1) + '°';

  // Niet overschrijven terwijl je zelf een waarde aan het typen bent
  if (autoHeading.checked && document.activeElement !== headingInput) {
    headingInput.value = cam.heading;
  }
}

let lastCommandCount = null;

function showMarkerCommand(cam) {
  if (!cam || cam.command_count === undefined) return;

  // De eerste ronde tonen we niets, anders meldt de pagina een oud commando
  if (lastCommandCount !== null && cam.command_count > lastCommandCount && cam.last_command) {
    setStatus('aruco ' + cam.last_command.id + ': ' + cam.last_command.command, 'ok');
  }
  lastCommandCount = cam.command_count;
}

async function poll() {
  try {
    const res = await fetch('/status');
    const data = await res.json();
    if (!data.connected) setStatus('robot niet verbonden', 'err');
    showCamera(data.camera);
    showMarkerCommand(data.camera);
    showPhotos(data.photos);
    showFollowing(data);
    showRoute(data.route);
  } catch (err) {
    setStatus('server onbereikbaar', 'err');
  }
}
poll();
setInterval(poll, 1000);
</script>
</body>
</html>
"""


ARUCO_PAGE = """
<!doctype html>
<html lang="nl">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
  <meta name="theme-color" content="#111418">
  <title>Go2 draairegels</title>
  <style>
""" + PAGE_CSS + """
    table { width: 100%; border-collapse: collapse; font-size: 13px; }
    th, td { padding: 8px 6px; text-align: left; border-bottom: 1px solid var(--line); }
    th { color: var(--muted); font-size: 11px; text-transform: uppercase; letter-spacing: .06em; }
    td.marker { color: #ff7bff; font-weight: 700; font-variant-numeric: tabular-nums; }
    td.bereik { color: var(--muted); font-variant-numeric: tabular-nums; }
    td.actie { width: 44px; }
    td.actie button { min-height: 34px; padding: 4px; }
    .form { display: grid; grid-template-columns: repeat(2, 1fr); gap: 8px; }
    .form label { font-size: 11px; color: var(--muted); display: block; margin-bottom: 4px; }
    .form input, .form select {
      font: inherit;
      color: var(--text);
      background: var(--panel-2);
      border: 1px solid var(--line);
      border-radius: 10px;
      padding: 10px;
      min-height: 46px;
      width: 100%;
      text-align: center;
    }
    .form input { user-select: text; -webkit-user-select: text; }
    .wide { grid-column: 1 / -1; }
    a.terug { color: var(--accent); text-decoration: none; font-size: 13px; }
    .leeg { color: var(--muted); font-size: 13px; }
  </style>
</head>
<body>

<header>
  <h1>&#127919; Draairegels</h1>
  <div class="status"><span class="dot" id="dot"></span><span id="msg">klaar</span></div>
</header>

<section>
  <h2>Regels</h2>
  <table>
    <thead>
      <tr><th>marker</th><th>draai</th><th>duur</th><th>afstand</th><th></th></tr>
    </thead>
    <tbody id="rules"></tbody>
  </table>
  <p class="leeg" id="leeg">Nog geen regels ingesteld.</p>
</section>

<section>
  <h2>Regel toevoegen of aanpassen</h2>
  <div class="form">
    <div>
      <label for="id">marker</label>
      <input id="id" type="number" inputmode="numeric" min="0" max="999" step="1" value="30">
    </div>
    <div>
      <label for="direction">richting</label>
      <select id="direction">
        <option value="links">&#11013;&#65039; links</option>
        <option value="rechts">&#10145;&#65039; rechts</option>
      </select>
    </div>
    <div>
      <label for="seconds">duur (s)</label>
      <input id="seconds" type="number" inputmode="decimal" min="0.1" max="{{ max_seconds }}" step="0.1" value="2">
    </div>
    <div>
      <label for="distance">afstand (m)</label>
      <input id="distance" type="number" inputmode="decimal" min="0.1" max="{{ max_distance }}" step="0.1" value="1">
    </div>
    <div>
      <label for="margin">marge (%)</label>
      <input id="margin" type="number" inputmode="numeric" min="0" max="100" step="5" value="20">
    </div>
    <div>
      <label>&nbsp;</label>
      <button id="save"><span class="ico">&#128190;</span><span class="lbl">bewaren</span></button>
    </div>
    <p class="hint wide" id="preview">&nbsp;</p>
  </div>
</section>

<section>
  <h2>Vaste commando's</h2>
  <table>
    <thead><tr><th>marker</th><th>commando</th><th></th></tr></thead>
    <tbody id="commands"></tbody>
  </table>
  <p class="leeg" id="leeg-commands">Nog geen markers gekoppeld.</p>

  <div class="form">
    <div>
      <label for="command">commando</label>
      <select id="command">
        {% for name in commands %}
        <option value="{{ name }}">{{ name }}</option>
        {% endfor %}
      </select>
    </div>
    <div>
      <label for="command-id">marker</label>
      <input id="command-id" type="number" inputmode="numeric" min="0" max="999" step="1" value="22">
    </div>
    <div class="wide">
      <button id="save-command"><span class="ico">&#128190;</span><span class="lbl">commando bewaren</span></button>
    </div>
  </div>

</section>

<footer><a class="terug" href="/">&#8592; terug naar de bediening</a></footer>

<script>
const dot = document.getElementById('dot');
const msg = document.getElementById('msg');
const velden = ['id', 'direction', 'seconds', 'distance', 'margin'];

function setStatus(text, state) {
  msg.textContent = text;
  dot.className = 'dot' + (state ? ' ' + state : '');
}

function waarde(naam) { return document.getElementById(naam).value; }

// Laat zien tussen welke afstanden de regel zal afgaan
function toonBereik() {
  const distance = parseFloat(waarde('distance'));
  const margin = parseFloat(waarde('margin'));
  const preview = document.getElementById('preview');

  if (isNaN(distance) || isNaN(margin)) { preview.textContent = ' '; return; }

  const rand = distance * margin / 100;
  preview.textContent = 'gaat af tussen ' + (distance - rand).toFixed(2) +
                        ' m en ' + (distance + rand).toFixed(2) + ' m';
}
velden.forEach(naam => document.getElementById(naam).addEventListener('input', toonBereik));
toonBereik();

function rij(rule) {
  const tr = document.createElement('tr');
  const pijl = rule.direction === 'links' ? '&#11013;&#65039;' : '&#10145;&#65039;';
  tr.innerHTML =
    '<td class="marker">' + rule.id + '</td>' +
    '<td>' + pijl + ' ' + rule.direction + '</td>' +
    '<td>' + rule.seconds + ' s</td>' +
    '<td class="bereik">' + rule.low.toFixed(2) + ' - ' + rule.high.toFixed(2) + ' m</td>';

  const cel = document.createElement('td');
  cel.className = 'actie';
  const knop = document.createElement('button');
  knop.innerHTML = '<span class="ico">&#128465;&#65039;</span>';
  knop.addEventListener('click', () => verwijder(rule.id));
  cel.appendChild(knop);
  tr.appendChild(cel);
  return tr;
}

function toonRegels(rules) {
  const body = document.getElementById('rules');
  body.innerHTML = '';
  rules.forEach(rule => body.appendChild(rij(rule)));
  document.getElementById('leeg').style.display = rules.length ? 'none' : 'block';
}

async function laden() {
  try {
    const res = await fetch('/aruco/rules');
    toonRegels(await res.json());
  } catch (err) {
    setStatus('server onbereikbaar', 'err');
  }
}

async function bewaren() {
  const vraag = velden.map(naam => naam + '=' + encodeURIComponent(waarde(naam))).join('&');
  try {
    const res = await fetch('/aruco/rules?' + vraag, { method: 'POST' });
    const data = await res.json();
    if (!data.ok) { setStatus(data.error || 'fout', 'err'); return; }
    setStatus('marker ' + data.rule.id + ' bewaard', 'ok');
    toonRegels(data.rules);
  } catch (err) {
    setStatus('geen verbinding', 'err');
  }
}

async function verwijder(markerId) {
  try {
    const res = await fetch('/aruco/rules/delete?id=' + markerId, { method: 'POST' });
    const data = await res.json();
    if (!data.ok) { setStatus(data.error || 'fout', 'err'); return; }
    setStatus('marker ' + markerId + ' verwijderd', 'ok');
    toonRegels(data.rules);
  } catch (err) {
    setStatus('geen verbinding', 'err');
  }
}

// ------------------------------------------------ de vaste commando's

function commandoRij(koppeling) {
  const tr = document.createElement('tr');
  tr.innerHTML =
    '<td class="marker">' + koppeling.id + '</td>' +
    '<td>' + koppeling.command + '</td>';

  const cel = document.createElement('td');
  cel.className = 'actie';
  const knop = document.createElement('button');
  knop.innerHTML = '<span class="ico">&#128465;&#65039;</span>';
  knop.addEventListener('click', () => verwijderCommando(koppeling.command));
  cel.appendChild(knop);
  tr.appendChild(cel);
  return tr;
}

function toonCommandos(koppelingen) {
  const body = document.getElementById('commands');
  body.innerHTML = '';
  koppelingen.forEach(koppeling => body.appendChild(commandoRij(koppeling)));
  document.getElementById('leeg-commands').style.display =
    koppelingen.length ? 'none' : 'block';
}

async function ladenCommandos() {
  try {
    const res = await fetch('/aruco/commands');
    toonCommandos(await res.json());
  } catch (err) {
    setStatus('server onbereikbaar', 'err');
  }
}

async function bewaarCommando() {
  const vraag = 'command=' + encodeURIComponent(waarde('command')) +
                '&id=' + encodeURIComponent(waarde('command-id'));
  try {
    const res = await fetch('/aruco/commands?' + vraag, { method: 'POST' });
    const data = await res.json();
    if (!data.ok) { setStatus(data.error || 'fout', 'err'); return; }
    setStatus(data.marker.command + ' op marker ' + data.marker.id, 'ok');
    toonCommandos(data.markers);
  } catch (err) {
    setStatus('geen verbinding', 'err');
  }
}

async function verwijderCommando(command) {
  try {
    const res = await fetch('/aruco/commands/delete?command=' + encodeURIComponent(command),
                            { method: 'POST' });
    const data = await res.json();
    if (!data.ok) { setStatus(data.error || 'fout', 'err'); return; }
    setStatus(command + ' losgekoppeld', 'ok');
    toonCommandos(data.markers);
  } catch (err) {
    setStatus('geen verbinding', 'err');
  }
}

document.getElementById('save').addEventListener('click', bewaren);
document.getElementById('save-command').addEventListener('click', bewaarCommando);
laden();
ladenCommandos();
</script>
</body>
</html>
"""


app = Flask(__name__)


@app.route("/")
def index():
    return render_template_string(
        HTML_PAGE,
        robot_ip=ROBOT_IP,
        confidence_choices=CONFIDENCE_CHOICES,
        confidence=segmentation.confidence,
        model_choices=segmentation.model_choices(),
        model=segmentation.model_name,
        photo_intervals=photos.interval_choices(),
        photo_interval=photos.interval,
        aruco_commands=command_markers.as_dict(),
    )


@app.route("/status")
def status():
    return jsonify({
        "connected": robot.connected,
        "camera": segmentation.snapshot(),
        "photos": photos.snapshot(),
        "following": path_follower.active,
        "follow_reason": path_follower.reason,
        "follow_changes": path_follower.changes,
        "route": route_follower.snapshot(),
    })


@app.route("/route/paths")
def route_paths():
    """De routes die op de Jetson staan (uit de stayontrails-app)."""
    try:
        return jsonify(freewalk_get("/?action=list_paths"))
    except Exception as exc:
        return jsonify({"ok": False, "error": "stayontrails-app niet bereikbaar: %s" % exc,
                        "paths": []}), 503


@app.route("/route/start", methods=["POST", "GET"])
def route_start():
    slug = request.args.get("slug", "")
    if not slug:
        return jsonify({"ok": False, "error": "parameter 'slug' ontbreekt"}), 400
    result = route_follower.start(slug)
    return jsonify(result), (200 if result.get("ok") else 409)


@app.route("/route/stop", methods=["POST", "GET"])
def route_stop():
    route_follower.stop("op de knop geklikt")
    if robot.connected:
        robot.move(x=0, y=0, z=0)
    return jsonify({"ok": True})


@app.route("/video")
def video():
    """Live camerabeeld van de robot met het masker van het model erop.

    Het is een MJPEG-stream, zo kan de webpagina er gewoon een <img> van maken."""
    if not USE_CAMERA or cv2 is None:
        return jsonify({"ok": False, "error": "camera staat uit"}), 404

    def frames():
        for jpeg in segmentation.stream_frames():
            yield (b"--frame\r\n"
                   b"Content-Type: image/jpeg\r\n"
                   b"Content-Length: " + str(len(jpeg)).encode() + b"\r\n\r\n" +
                   jpeg + b"\r\n")

    return Response(
        frames(),
        mimetype="multipart/x-mixed-replace; boundary=frame",
        headers={"Cache-Control": "no-store", "Pragma": "no-cache"},
    )


@app.route("/heading/", methods=["GET", "POST"])
@app.route("/heading", methods=["GET", "POST"])
def heading():
    """Draai de robot naar een heading, bv. http://<ip>:8080/heading/?heading=101

    Bij precies 90 stapt de robot een seconde rechtdoor. Meer dan 90 draait
    naar rechts, minder naar links: het verschil met 90 is de hoek die de robot
    draait, en ondertussen gaat hij traag vooruit."""
    raw = request.args.get("heading")
    if raw is None:
        return jsonify({"ok": False, "error": "parameter 'heading' ontbreekt"}), 400

    try:
        value = float(raw)
    except ValueError:
        return jsonify({"ok": False, "error": "heading moet een getal zijn: " + raw}), 400

    if not robot.connected:
        return jsonify({"ok": False, "error": "geen verbinding met de robot"}), 503

    error = value - HEADING_FORWARD

    try:
        result = robot.follow_heading(error)
    except Exception as exc:
        return jsonify({"ok": False, "error": str(exc)}), 500

    if result is None:
        return jsonify({"ok": False, "error": "er is al een beweging bezig"}), 409

    return jsonify({
        "ok": True,
        "heading": value,
        "turned": result["degrees"],
        "direction": "rechts" if result["degrees"] > 0 else ("links" if result["degrees"] < 0 else "vooruit"),
        "forward": result["forward"],
        "duration": result["duration"],
    })


@app.route("/confidence/", methods=["GET", "POST"])
@app.route("/confidence", methods=["GET", "POST"])
def confidence():
    """Lees of zet de confidence van de segmentatie, bv. /confidence/?value=0.4

    Zonder parameter geeft dit gewoon de huidige waarde terug."""
    raw = request.args.get("value")
    if raw is None:
        return jsonify({"ok": True, "confidence": segmentation.confidence,
                        "choices": CONFIDENCE_CHOICES})

    try:
        value = float(raw)
    except ValueError:
        return jsonify({"ok": False, "error": "confidence moet een getal zijn: " + raw}), 400

    return jsonify({"ok": True, "confidence": segmentation.set_confidence(value)})


@app.route("/model/", methods=["GET", "POST"])
@app.route("/model", methods=["GET", "POST"])
def model():
    """Lees of kies het model, bv. /model/?name=laerbeekbos.pt

    De keuze gaat over de .pt-bestanden die in MODEL_DIR staan. Het laden
    gebeurt in de achtergrondthread en duurt een tiental seconden; ondertussen
    staat de status op "model laden"."""
    raw = request.args.get("name")
    if raw is None:
        return jsonify({"ok": True, "model": segmentation.model_name,
                        "choices": segmentation.model_choices(),
                        "directory": MODEL_DIR})

    try:
        name = segmentation.set_model(raw)
    except ValueError as exc:
        return jsonify({"ok": False, "error": str(exc),
                        "choices": segmentation.model_choices()}), 400

    return jsonify({"ok": True, "model": name})


@app.route("/photo/", methods=["GET", "POST"])
@app.route("/photo", methods=["GET", "POST"])
def photo():
    """Bewaar meteen een foto van het ruwe camerabeeld, bv. /photo/

    Dat is het beeld zoals het binnenkwam: zonder masker, zonder kader rond een
    marker en zonder het commando in het midden."""
    try:
        name = photos.save_now()
    except Exception as exc:
        return jsonify({"ok": False, "error": str(exc)}), 503

    return jsonify({"ok": True, "photo": name, "count": photos.count,
                    "directory": PHOTO_DIR})


@app.route("/photo/interval/", methods=["GET", "POST"])
@app.route("/photo/interval", methods=["GET", "POST"])
def photo_interval():
    """Lees of zet om de hoeveel seconden er vanzelf een foto bewaard wordt,
    bv. /photo/interval/?value=30. Met 0 gebeurt dat enkel nog op de knop."""
    choices = photos.interval_choices()
    raw = request.args.get("value")
    if raw is None:
        return jsonify({"ok": True, "interval": photos.interval,
                        "count": photos.count, "choices": choices,
                        "directory": PHOTO_DIR})

    try:
        seconds = photos.set_interval(raw)
    except ValueError as exc:
        return jsonify({"ok": False, "error": str(exc), "choices": choices}), 400

    label = next(choice["label"] for choice in choices
                 if choice["seconds"] == seconds)
    return jsonify({"ok": True, "interval": seconds, "label": label,
                    "count": photos.count})


def rules_for_page():
    """De regels met hun bereik erbij, klaar voor de tabel op de pagina."""
    rules = []
    for rule in turn_rules.all():
        low, high = rule_range(rule)
        item = dict(rule)
        item["low"] = round(low, 2)
        item["high"] = round(high, 2)
        rules.append(item)
    return rules


@app.route("/aruco/")
@app.route("/aruco")
def aruco():
    """Pagina om de draairegels van de markers in te stellen."""
    return render_template_string(
        ARUCO_PAGE,
        commands=sorted(COMMANDS),
        max_seconds=MAX_TURN_SECONDS,
        max_distance=MAX_RULE_DISTANCE,
    )


@app.route("/aruco/rules", methods=["GET", "POST"])
def aruco_rules():
    """De draairegels lezen, of er een toevoegen of aanpassen.

    Toevoegen gaat met alle vijf de waarden, bv.
    /aruco/rules?id=30&direction=links&seconds=2&distance=1&margin=20"""
    if request.method == "GET":
        return jsonify(rules_for_page())

    try:
        rule = turn_rules.set(
            request.args.get("id"),
            request.args.get("direction"),
            request.args.get("seconds"),
            request.args.get("distance"),
            request.args.get("margin"),
        )
    except ValueError as exc:
        return jsonify({"ok": False, "error": str(exc)}), 400

    return jsonify({"ok": True, "rule": rule, "rules": rules_for_page()})


@app.route("/aruco/commands", methods=["GET", "POST"])
def aruco_commands():
    """De vaste commando's lezen, of een commando aan een marker koppelen.

    Koppelen gaat met beide waarden, bv. /aruco/commands?command=hello&id=22"""
    if request.method == "GET":
        return jsonify(command_markers.all())

    try:
        marker = command_markers.set(request.args.get("command"), request.args.get("id"))
    except ValueError as exc:
        return jsonify({"ok": False, "error": str(exc)}), 400

    return jsonify({"ok": True, "marker": marker, "markers": command_markers.all()})


@app.route("/aruco/commands/delete", methods=["POST"])
def aruco_commands_delete():
    """Haal de marker van een commando weg, bv. /aruco/commands/delete?command=hello"""
    try:
        removed = command_markers.delete(request.args.get("command"))
    except ValueError as exc:
        return jsonify({"ok": False, "error": str(exc)}), 400

    return jsonify({"ok": True, "removed": removed, "markers": command_markers.all()})


@app.route("/aruco/rules/delete", methods=["POST"])
def aruco_rules_delete():
    """Verwijder de regel van een marker, bv. /aruco/rules/delete?id=30"""
    try:
        removed = turn_rules.delete(request.args.get("id"))
    except ValueError as exc:
        return jsonify({"ok": False, "error": str(exc)}), 400

    return jsonify({"ok": True, "removed": removed, "rules": rules_for_page()})


@app.route("/commands")
def commands():
    """Handig om te zien welke commando's er bestaan."""
    return jsonify(sorted(COMMANDS))


@app.route("/cmd/<command>", methods=["POST", "GET"])
def cmd(command):
    if command not in COMMANDS:
        return jsonify({"ok": False, "error": "onbekend commando: " + command}), 404

    if not robot.connected:
        return jsonify({"ok": False, "error": "geen verbinding met de robot"}), 503

    try:
        robot.run_command(command)
    except Exception as exc:
        return jsonify({"ok": False, "error": str(exc)}), 500

    return jsonify({"ok": True, "command": command})


# ---------------------------------------------------------------------- opstart

def run_asyncio_loop(loop, ready):
    asyncio.set_event_loop(loop)

    async def recv_camera_stream(track):
        while True:
            frame = await track.recv()
            segmentation.put_frame(frame.to_ndarray(format="bgr24"))

    async def setup_sim():
        # Zelfde berichten als de echte Go2, maar via de simulator; die levert
        # ook het camerabeeld en (uit zijn pose) een nep-GPS voor stayontrails
        import simgo2

        conn = simgo2.SimConnection(ROBOT_SIM)
        robot.conn = conn
        robot.loop = loop
        robot.connected = True
        print("Verbonden met de simulator (sessie %s)" % ROBOT_SIM, flush=True)

        if USE_CAMERA:
            conn.start_camera(segmentation.put_frame)
            print("Camerastream van de simulator gestart", flush=True)
        conn.start_gps(FREEWALK_URL)
        print("Nep-GPS van de simulator naar " + FREEWALK_URL, flush=True)

    async def setup():
        if ROBOT_SIM:
            return await setup_sim()

        conn = UnitreeWebRTCConnection(
            WebRTCConnectionMethod.LocalSTA,
            ip=ROBOT_IP
        )
        await conn.connect()

        # Zorg dat de robot in normale modus staat
        response = await conn.datachannel.pub_sub.publish_request_new(
            RTC_TOPIC["MOTION_SWITCHER"],
            {"api_id": 1001}
        )
        data = json.loads(response["data"]["data"])
        if data["name"] != "normal":
            await conn.datachannel.pub_sub.publish_request_new(
                RTC_TOPIC["MOTION_SWITCHER"],
                {"api_id": 1002, "parameter": {"name": "normal"}}
            )
            await asyncio.sleep(5)

        robot.conn = conn
        robot.loop = loop
        robot.connected = True
        print("Verbonden met de Go2", flush=True)

        # Camerabeelden binnenhalen voor de segmentatie
        if USE_CAMERA:
            conn.video.switchVideoChannel(True)
            conn.video.add_track_callback(recv_camera_stream)
            print("Camerastream gestart", flush=True)

    try:
        loop.run_until_complete(setup())
    except Exception as exc:
        print("Verbinden mislukt: " + str(exc), flush=True)
    finally:
        ready.set()

    loop.run_forever()


def main():
    loop = asyncio.new_event_loop()
    ready = threading.Event()

    asyncio_thread = threading.Thread(
        target=run_asyncio_loop, args=(loop, ready), daemon=True)
    asyncio_thread.start()

    # Het YOLO-model laden duurt even, dus dat gebeurt in de achtergrond
    segmentation.start()

    # Bewaart om de zoveel tijd een foto, zodra er een interval gekozen is
    photos.start()

    # Wacht tot de verbindingspoging klaar is, zodat de eerste kliks werken
    ready.wait(timeout=30)

    print("Open op je smartphone: http://<ip-van-deze-pc>:%d/" % WEB_PORT, flush=True)

    try:
        app.run(host=WEB_HOST, port=WEB_PORT, threaded=True, use_reloader=False)
    finally:
        loop.call_soon_threadsafe(loop.stop)


if __name__ == "__main__":
    main()
