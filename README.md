# Unitree Go2 – Stay On Trails

Een Unitree Go2 die een blinde of slechtziende wandelaar begeleidt langs een
GPS-route. Een YOLO-segmentatiemodel herkent het pad in het camerabeeld en
houdt robot of wandelaar in het midden van het pad. ArUco-markers geven extra
instructies, en een RTK-GPS bepaalt waar op de route je bent.

Alles draait op een **NVIDIA Jetson Orin** (map `~/JetsonLukas`). De
webpagina's van Stay On Trails staan op https://stayontrails.ehb.be.

## Overzicht

```
                 ┌──────────────── Jetson Orin ────────────────┐
 Go2 (WebRTC) ──▶│ Unitree/urlcontrol.py  :8080                │
 of simulator    │  YOLO-segmentatie, pad volgen, ArUco,       │
                 │  route via GPS, /segment voor de gsm-camera │
                 │                                             │
                 │ stayontrails/freewalkFlask.py  :5003 (https)│
                 │  routes (stayontrails/paths/*.json),        │
                 │  laatste GPS-positie (/api/gps)             │
                 └──────────────▲──────────────────────────────┘
                                │ USB (adb reverse 8080 / 5003)
                 ┌──────────────┴──────────────┐
 RTK-ontvanger ─▶│ gsm (Android, Chrome)       │◀── followpathCombined.php
 (bluetooth)     │  GNSS Master: mock location │    (stayontrails.ehb.be)
                 └─────────────────────────────┘
```

| Onderdeel | Wat het doet |
|---|---|
| `Unitree/urlcontrol.py` | Bestuurt de Go2 en heeft een eigen webpagina op poort 8080. Volgt het pad met YOLO, leest ArUco-markers, loopt een route af op GPS, en segmenteert beelden van de gsm-camera (`POST /segment`). |
| `Unitree/simgo2.py` | Vervangt de echte Go2 door de Botopia-simulator, en zet de gesimuleerde positie om in nep-GPS. |
| `stayontrails/freewalkFlask.py` | De Stay On Trails-app op de Jetson (https, poort 5003). Houdt de routes en de laatste GPS-positie van de gsm bij. |
| `signaling/models/` | De YOLO-segmentatiemodellen (`laerbeekbos.pt`, `unrealsim.pt`, …). |
| `faceassist/`, `CSI/`, `voices/` | Gezichtsherkenning, de CSI-camera en Piper-stemmen (aparte onderdelen). |

`followpathCombined.php` (de wandelpagina met Cloud- en Edge-modus) staat op
de webserver van stayontrails.ehb.be, niet in deze repository.

## Installatie op de Jetson

```bash
cd ~ && git clone https://github.com/Lukasehb/UnitreeRoboticsProject2026-2027.git JetsonLukas
pip install flask opencv-python numpy ultralytics unitree_webrtc_connect cryptography
```

Gebruik op de Jetson de **PyTorch-versie van NVIDIA voor JetPack** (met CUDA).
Met de gewone `pip`-torch rekent YOLO op de CPU, en dat is ongeveer tien keer
trager.

### Sneller model met TensorRT

Exporteer het model op de Jetson zelf. Een `.engine` werkt enkel op de
Jetson waar hij gemaakt is, en staat daarom in `.gitignore`.

```bash
cd ~/JetsonLukas/signaling/models
yolo export model=laerbeekbos.pt format=engine half=True device=0   # duurt ~5-10 min
```

`urlcontrol.py` kiest automatisch `laerbeekbos.engine` als die naast
`laerbeekbos.pt` staat. Op een Jetson Orin kost een beeld dan ongeveer 30 ms
(waarvan ~16 ms inference), tegen ~300 ms op de CPU.

## Starten

### Met de echte Go2

`urlcontrol.py` draait als service, `unitree-urlcontrol.service`, en verbindt
met de Go2 op `ROBOT_IP` (standaard `10.2.172.215`).

```bash
sudo systemctl restart unitree-urlcontrol.service
journalctl -u unitree-urlcontrol.service -f     # wacht op "Running on ...:8080" (~40 s)
```

### Met de simulator (zonder Go2)

Je hebt [robotSimulatorAPI](https://github.com/mdequanter/robotSimulatorAPI)
nodig, een account op robotics.dinno.be met een API-sleutel, Node.js 22 en
Chromium.

```bash
# terminal 1: de simulator
cd ~/robotSimulatorAPI
ROBOTSIM_KEY=rsk_... node server.mjs --browsers 1 --chrome /pad/naar/chrome

# terminal 2: urlcontrol met de simulator (stop eerst de service)
sudo systemctl stop unitree-urlcontrol.service
cd ~/JetsonLukas/Unitree
BOTOPIA_DIR=~/robotSimulatorAPI/python ROBOT_SIM=local SIM_WORLD=park python3 urlcontrol.py
```

`ROBOT_SIM=local` gebruikt de lokale simulator. Met een sessiecode
(`ROBOT_SIM=CODE`) gebruik je in de plaats een simulatorvenster in de browser
dat geopend is met `?remote=CODE`.

Op een Jetson werkt de snap-versie van Chromium niet. Een Chromium zonder snap
installeer je met `npx playwright install chromium`; geef het pad dan mee met
`--chrome`.

### De gsm aansluiten (USB)

Zet op de gsm USB-foutopsporing aan en verbind hem met de Jetson:

```bash
adb devices                      # de gsm moet als "device" staan
adb reverse tcp:8080 tcp:8080    # urlcontrol.py
adb reverse tcp:5003 tcp:5003    # freewalkFlask.py
```

`adb reverse` vervalt telkens de kabel loskomt. Via `localhost` mag de https-pagina
van stayontrails.ehb.be de Jetson aanspreken zonder certificaat.

## Werkwijzen

### Edge-modus: de gsm-camera, de Jetson rekent

Open op de gsm:
`https://stayontrails.ehb.be/lukas/followpathCombined.php?mode=edge&slug=<route>`

De pagina stuurt elk camerabeeld naar `POST localhost:8080/segment`. De Jetson
geeft de richting en de omtrek van het pad terug, en de pagina tekent het pad
over het camerabeeld. Met `&camera=robot` toont de pagina in de plaats het
beeld van de robot (`/video`) en de richting van `/guidance`.

### De Go2 het pad laten volgen

Druk op `http://localhost:8080/` op **Stand up** en **volg pad**, of gebruik:

```bash
curl http://localhost:8080/cmd/stand_up
curl http://localhost:8080/cmd/follow        # nog eens = stoppen
```

Verliest de robot het pad, dan draait hij ter plaatse naar de kant waar het
pad het laatst lag. Vindt hij het na 6 s niet terug, dan stopt hij. In bochten
vertraagt hij.

### Een route afleggen met RTK-GPS

1. Zet de route als JSON in `stayontrails/paths/<slug>.json` op de Jetson.
2. Start freewalkFlask, zonder `FREEWALK_SIM_GPS`:
   `cd ~/JetsonLukas/stayontrails && python3 freewalkFlask.py`
3. Op de gsm: verbind de RTK-ontvanger (ArduSimple simpleRTK2B) via bluetooth
   in **GNSS Master**, met FLEPOS-correcties (*Send GGA* aan) en **Mock
   Location** aan. Open `https://localhost:5003` en laat die pagina open: ze
   stuurt de positie elke seconde naar `/api/gps`.
4. Start de route: `curl "http://localhost:8080/route/start?slug=<slug>"`

De camera houdt de robot op het pad. De GPS zet het volgende punt actief en
stuurt bij als de looprichting te ver afwijkt.

## Endpoints van urlcontrol.py (poort 8080)

| Endpoint | Doel |
|---|---|
| `GET /` | Bedieningspagina |
| `GET /status` | Verbinding, camera, pad volgen, route |
| `GET /video` | MJPEG-stream van de robotcamera, met het pad erop |
| `GET /guidance` | Richting uit de robotcamera, `{"heading": 92.5}` (90 = rechtdoor, >90 = links) |
| `POST /segment` | JPEG van de gsm → `{heading, path, masks, ms, timings, gpu, …}` |
| `GET /timings` | Rekentijden van YOLO (gsm en robot), ook om de 5 s in de log |
| `GET /cmd/<commando>` | Bv. `stand_up`, `stand_down`, `follow`, `hello` |
| `GET /route/paths`, `/route/start?slug=`, `/route/stop` | Een GPS-route afleggen |
| `GET /model?name=`, `/confidence?value=` | Model en confidence kiezen |
| `GET /heading?heading=` | De robot naar een richting draaien |

`/guidance` en `/segment` laten via CORS enkel `https://stayontrails.ehb.be`
toe (`GUIDANCE_ORIGIN`).

## Omgevingsvariabelen

| Variabele | Standaard | Betekenis |
|---|---|---|
| `ROBOT_IP` | `10.2.172.215` | IP-adres van de Go2 |
| `ROBOT_AES_KEY` | leeg | Sleutel per robot, nodig vanaf Go2-firmware 1.1.15 (`unitree-fetch-aes-key`) |
| `ROBOT_SIM` | leeg | `local`, `http://<ip>:8780` of een sessiecode: de simulator gebruiken |
| `BOTOPIA_DIR` | leeg | Map met `botopia.py` uit robotSimulatorAPI |
| `SIM_WORLD` / `SIM_LANE` | – / `track` | Wereld (`park`) en startplaats in de simulator |
| `SIM_GPS_NOISE_M` | `0` | Ruis op de nep-GPS van de simulator, in meter |
| `USE_CAMERA` | `1` | `0` = zonder robotcamera en YOLO |
| `PAUSE_ROBOT_YOLO_FOR_PHONE` | `1` | Robot-YOLO pauzeren zolang de gsm beelden stuurt |
| `GUIDANCE_ORIGIN` | `https://stayontrails.ehb.be` | Toegelaten site voor `/guidance` en `/segment` |
| `FREEWALK_URL` | `https://127.0.0.1:5003` | Waar urlcontrol de routes en de GPS ophaalt |
| `FREEWALK_SIM_GPS` | `0` | freewalkFlask: GPS van de simulator in plaats van de gsm |

## Problemen oplossen

- **`adb devices` toont `unauthorized`, en er komt geen vraag op de gsm:** kijk
  of `~/.android/adbkey.pub` bestaat. Sommige adb-versies van Ubuntu maken
  enkel `adbkey` aan. Schakel op Samsung ook *Automatische blokkering* uit.
- **De gsm verdwijnt uit `adb devices`:** gebruik een datakabel en kies
  *Bestandsoverdracht*. Doe daarna opnieuw `adb reverse`.
- **"Address already in use" op poort 8080:** de service draait nog. Stop hem met
  `sudo systemctl stop unitree-urlcontrol.service`.
- **YOLO is traag:** `/timings` toont `"device": "cpu"`. Installeer dan de
  PyTorch van NVIDIA en exporteer een `.engine`.
- **De eerste `/segment` duurt een paar seconden:** dat is het laden en opwarmen
  van het model. Daarna gaat het snel.
