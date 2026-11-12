# CLAUDE.md

Unitree Go2 guiding a blind/visually impaired walker along a GPS route
("Stay On Trails"). Runs on a Jetson Orin (`~/JetsonLukas`). Code comments,
README and UI text are in Dutch. Full setup/endpoints/env vars: `README.md`
(read only the section you need).

## Where things are
- `Unitree/urlcontrol.py` — **main app** (~4k lines, Flask :8080): Go2 control,
  YOLO path segmentation, ArUco, GPS routes, `POST /segment` for the phone.
  Never read it whole: grep for the route/function, then read that line range.
- `Unitree/simgo2.py` — Botopia simulator in place of the real Go2.
- `signaling/models/` — YOLO `.pt` weights (binary; just reference by name).
- `faceassist/` — separate face-recognition app (`app.py` ~3k lines, same rule
  as urlcontrol). `nl_launchv2.py` is the newer version of `nl_launch.py`.
- `CSI/`, `voices/` — CSI camera scripts, Piper TTS voices (separate parts).
- Root `*.py` (ollama*, detectPersons, naoCam, viewCSICam) — small standalone tests.

## Don't touch / don't read
- `stayontrails/` — finished codebase, **do not read or change unless
  explicitly asked**. urlcontrol only talks to it over HTTP via `FREEWALK_URL`
  (`/api/gps`, `/?action=load_path&slug=`); route files are `stayontrails/paths/*.json`.
- `Unitree/urlcontrolv1.py` — old version of urlcontrol.py, ignore.
- Weights, videos, photos, venv, `__pycache__` (blocked in `.claude/settings.json`).

## Working style
- Search (grep) before reading; read line ranges of large files, not whole files.
- No test suite; changes are verified on the Jetson/robot or the simulator by the user.
