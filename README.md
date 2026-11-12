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
pip3 install flask opencv-python ultralytics cryptography
pip3 install -U unitree_webrtc_connect      # nieuwe versie nodig voor recente Go2-firmware
pip3 install "numpy<2"                      # de NVIDIA-PyTorch werkt niet met numpy 2
```

Gebruik op de Jetson de **PyTorch-versie van NVIDIA voor JetPack** (met CUDA).
Met de gewone `pip`-torch rekent YOLO op de CPU, en dat is ongeveer tien keer
trager.

Na elke `pip3 install -U ...` opnieuw nakijken dat numpy onder versie 2 blijft:
`python3 -c "import numpy; print(numpy.__version__)"`. Anders geeft de camera
de fout `Numpy is not available`.

### Het IP-adres van de Go2 instellen

`urlcontrol.py` draait als service (`unitree-urlcontrol.service`) en verbindt
met de Go2 op `ROBOT_IP`. Het netwerk kan de Go2 bij elke herstart een ander
adres geven: zoek het op (bv. met Advanced IP Scanner, of op de gsm bij de
verbonden apparaten van de hotspot) en zet het in een aanvulling op de service:

```bash
sudo mkdir -p /etc/systemd/system/unitree-urlcontrol.service.d
printf '[Service]\nEnvironment=ROBOT_IP=<ip-van-de-go2>\n' | sudo tee /etc/systemd/system/unitree-urlcontrol.service.d/robot-ip.conf
sudo systemctl daemon-reload
systemctl show unitree-urlcontrol.service -p Environment   # controle
```

Heeft de Go2 firmware 1.1.15 of nieuwer en meldt de verbinding
`AesKeyRequiredError`, voeg dan ook `Environment=ROBOT_AES_KEY=<32 hex-tekens>`
toe (ophalen met `unitree-fetch-aes-key --device-type Go2 --email <account> --sn <serienummer>`
via het Unitree-account waaraan de Go2 gekoppeld is). Zet die sleutel nooit in
de repository, en maak het bestand enkel leesbaar voor root:
`sudo chmod 600 .../robot-ip.conf`.

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

### Met de echte Go2, stap voor stap

1. **Netwerk.** De Go2, de Jetson en (als je geen USB gebruikt) de gsm zitten op
   hetzelfde netwerk. Buiten kan dat de hotspot van de gsm zijn, met mobiele
   data aan voor FLEPOS. Kijk het adres van de Jetson na met `hostname -I`, en
   dat van de Go2 zoals hierboven.
2. **Go2 aan**, en wacht 1–2 minuten tot hij op het netwerk zit:
   `ping -c 3 <ip-van-de-go2>`.
3. **urlcontrol.py (re)starten.** De service start vanzelf bij het opstarten
   van de Jetson, maar verbindt enkel bij het starten met de Go2:
   ```bash
   sudo systemctl restart unitree-urlcontrol.service
   journalctl -u unitree-urlcontrol.service -f   # wacht op "Verbonden met de Go2" en "Running on ...:8080" (~40 s)
   curl http://localhost:8080/status             # "connected": true
   ```
   De Go2 aanvaardt maar één WebRTC-verbinding: sluit de Unitree-app en andere
   scripts af. Na een herstart van de service of de Jetson houdt de Go2 de
   oude verbinding soms nog vast; herstart dan ook de Go2 (zie *Problemen
   oplossen*). Herstart de service dus niet onnodig.
4. **StayOnTrails op de Jetson** (start niet vanzelf; laat de terminal open):
   ```bash
   cd ~/JetsonLukas/stayontrails && python3 freewalkFlask.py
   ```
5. **De gsm aansluiten** (USB, zie hieronder) of op hetzelfde netwerk zetten.
6. Verder met een van de *Werkwijzen*.

`urlcontrol.py` laat een Go2 met recente firmware in de bewegingsmodus `mcf`
staan; de log toont `Bewegingsmodus van de Go2: mcf`.

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
`--chrome`. Staat er ook een oudere Node via nvm, start dan met `/usr/bin/node`.

**Een route testen in de simulator** (zonder Go2 en zonder RTK). Gebruik de open
vloer (`SIM_LANE=free`, de standaardwereld) en leg de oorsprong van de nep-GPS
op het eerste punt van de route, zodat de robot daar start. Die vloer is maar
zo'n 14 m breed; een grotere route kan je voor de test verkleinen.

```bash
# terminal 2: StayOnTrails met de GPS van de simulator
cd ~/JetsonLukas/stayontrails && FREEWALK_SIM_GPS=1 python3 freewalkFlask.py

# terminal 3: urlcontrol met de simulator, gestart op het eerste routepunt
cd ~/JetsonLukas/Unitree
BOTOPIA_DIR=~/robotSimulatorAPI/python ROBOT_SIM=local SIM_LANE=free \
  SIM_ORIGIN_LAT=<lat van punt 1> SIM_ORIGIN_LON=<lon van punt 1> python3 urlcontrol.py

curl http://localhost:8080/cmd/stand_up
curl "http://localhost:8080/route/start?slug=<slug>&mode=gps"
```

Meekijken kan met `http://localhost:8780/api/envs/<id>/frame?view=scene&width=960`
(het id staat in `http://localhost:8780/api/envs`), of door de simulator te
starten met `--show`.

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

Zo komt de GPS bij de Jetson:

```
RTK-ontvanger ─bluetooth─▶ gsm (GNSS Master, mock location) ─▶ Chrome op :5003
  ─elke seconde, wifi of USB─▶ freewalkFlask (/api/gps) ─▶ urlcontrol.py ─▶ Go2
```

1. **De route** staat als JSON in `stayontrails/paths/<slug>.json` op de Jetson
   (urlcontrol.py leest de routes daar, niet van stayontrails.ehb.be). Kies
   punten minstens 5 m uit elkaar en een aankomststraal van 1–2 m.
2. **freewalkFlask** draait, zonder `FREEWALK_SIM_GPS` (zie *Starten*).
3. **RTK op de gsm**, onder open lucht:
   - de RTK-ontvanger (ArduSimple simpleRTK2B, verschijnt als `BT+BLE_Bridge_…`)
     verbonden in **GNSS Master**, met FLEPOS-correcties (*Send GGA* aan);
   - wacht tot GNSS Master **RTK Fix** toont;
   - **Mock Location** aan, en GNSS Master gekozen bij Ontwikkelaarsopties →
     *App voor neplocatie selecteren*;
   - zet bij Locatie → Locatieservices *Google-locatienauwkeurigheid*, wifi- en
     bluetooth-scannen uit, anders schuift Android zijn eigen locatie (~15 m)
     tussen die van de ontvanger.
4. **De positie doorsturen.** Open op de gsm `https://<ip-van-de-jetson>:5003`
   (of `https://localhost:5003` via USB), druk op **Start wandelen** en sta de
   locatie toe. Pas dan stuurt de pagina de positie; laat ze open met het
   scherm aan. Controle op de Jetson:
   ```bash
   curl -k https://127.0.0.1:5003/api/gps    # "fix": true, accuracy_m rond 0.02 bij RTK Fix, age_s < 2
   ```
5. **De Go2 klaarzetten:** antenne bovenop de robot, met open lucht erboven,
   robot op het startpunt, vrije ruimte rond de route.
   `curl http://localhost:8080/cmd/stand_up`
6. **Starten**, in een van twee modi:
   ```bash
   curl "http://localhost:8080/route/start?slug=<slug>&mode=gps"    # GPS-modus
   curl "http://localhost:8080/route/start?slug=<slug>"             # padmodus (camera)
   journalctl -u unitree-urlcontrol.service -f                      # meekijken
   curl http://localhost:8080/route/stop                            # stoppen
   ```
   Op `http://<ip-van-de-jetson>:8080/` kan het ook: kies de route, vink
   **GPS-modus** aan en druk op **volg route**.

**Padmodus** (standaard): de camera houdt de robot op het pad, de GPS zet het
volgende punt actief en stuurt grof bij. Werkt op een pad dat het model
herkent (bv. een bospad).

**GPS-modus** (`mode=gps`): geen camera; de robot loopt rechtstreeks naar elk
punt. Hij leert zijn looprichting uit de RTK-posities (~1 m rechtdoor), draait
ter plaatse als het volgende punt meer dan 45° opzij ligt, en stuurt anders al
lopend bij, aan 0,3 m/s. Hij start enkel bij een nauwkeurigheid van 1 m of
beter, negeert losse onnauwkeurige posities, staat stil na 3 s zonder goede
positie en stopt na 15 s. Hij ontwijkt geen obstakels: loop ernaast.

Binnen werkt de GPS-modus niet: zonder open lucht geen RTK Fix (binnen
5–15 m of slechter), en dan weigert hij te starten.

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
| `GET /route/paths`, `/route/start?slug=&mode=`, `/route/stop` | Een GPS-route afleggen (`mode=path` of `gps`) |
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
| `SIM_WORLD` / `SIM_LANE` | – / `track` | Wereld (`park`) en startplaats in de simulator (`free` = open vloer) |
| `SIM_ORIGIN_LAT` / `SIM_ORIGIN_LON` | `50.841746` / `4.322936` | GPS-positie van de oorsprong van de simulator |
| `SIM_GPS_NOISE_M` | `0` | Ruis op de nep-GPS van de simulator, in meter |
| `USE_CAMERA` | `1` | `0` = zonder robotcamera en YOLO |
| `PAUSE_ROBOT_YOLO_FOR_PHONE` | `1` | Robot-YOLO pauzeren zolang de gsm beelden stuurt |
| `GUIDANCE_ORIGIN` | `https://stayontrails.ehb.be` | Toegelaten site voor `/guidance` en `/segment` |
| `FREEWALK_URL` | `https://127.0.0.1:5003` | Waar urlcontrol de routes en de GPS ophaalt |
| `FREEWALK_SIM_GPS` | `0` | freewalkFlask: GPS van de simulator in plaats van de gsm |

## Problemen oplossen

**Verbinding met de Go2** (`"connected": false`, zie
`journalctl -u unitree-urlcontrol.service -n 40 --no-pager | grep -iv werkzeug`):

- **`not exposing a signaling port` of `Could not get SDP from the peer`:**
  verkeerd IP-adres (kijk `systemctl show unitree-urlcontrol.service -p Environment`
  na), of de Go2 staat (nog) niet op het netwerk. Test met `ping` en met
  `curl -s -o /dev/null -w "%{http_code}" http://<ip-van-de-go2>:9991/con_notify`
  (200 = de Go2 is klaar).
- **`Data channel setup timed out` (`ice=checking`):** een andere client houdt
  de enige verbinding bezet, of de Go2 houdt een oude verbinding vast. Sluit de
  Unitree-app en andere scripts af, en herstart proper:
  `sudo systemctl stop unitree-urlcontrol.service`, Go2 uit en aan, 2 minuten
  wachten, `sudo systemctl start unitree-urlcontrol.service`.
- **`AesKeyRequiredError` / `AesKeyRejectedError`:** zie `ROBOT_AES_KEY` bij
  *Het IP-adres van de Go2 instellen*.
- **De camera geeft `Numpy is not available`:** `pip3 install "numpy<2"` en de
  service herstarten.

**GPS en route:**

- **`geen verse GPS van de gsm`:** draait freewalkFlask, en is op de pagina
  `:5003` op **Start wandelen** gedrukt? Staat het scherm aan?
- **`GPS niet nauwkeurig genoeg voor de GPS-modus`:** geen RTK Fix. Ga naar
  buiten onder open lucht en wacht tot GNSS Master *RTK Fix* toont.
- **`accuracy_m` springt tussen ~0,02 m en ~15 m:** Android mengt zijn eigen
  locatie erdoor; zet *Google-locatienauwkeurigheid* uit (zie hierboven).
- **De route stopt meteen met `bestemming bereikt`:** de punten liggen dichter
  bij elkaar dan de aankomststraal.

**Gsm en Jetson:**

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
