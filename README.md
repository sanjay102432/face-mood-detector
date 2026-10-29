<div align="center">

# 🎭 face-mood-detector

**Turn any laptop webcam into a real-time AI camera —**
**face • eye • iris • hand • body detection, plus an 8-mood emotion engine.**

[![Python](https://img.shields.io/badge/python-3.9%20--%203.12-blue.svg)](https://www.python.org/downloads/)
[![MediaPipe](https://img.shields.io/badge/MediaPipe-0.10.21-orange.svg)](https://developers.google.com/mediapipe)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.13-red.svg)](https://opencv.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

No VR headset. No expensive RGB camera. No GPU. **Just your webcam** 📷 - it will work

</div>

---

## ✨ What's inside

| App | Command | What you get |
|---|---|---|
| 😎 **Face & Hands Detector** | `python detector.py` | Face mesh outline, **eyes + iris rings**, two-hand skeleton (labeled L/R), full **body-pose** skeleton, live FPS HUD |
| 🧠 **Mood Detector** | `python mood_detector.py` | Detects **HAPPY · SAD · ANGRY · SURPRISED · FEAR · DISGUST · CONFUSED · NEUTRAL** with intensity %, expressive meters, blink counter + mood-colored UI |

Everything runs **100% locally** — no cloud, no API keys, no sign-up.

---

## 🎭 The Mood Engine (the fun part)

At startup it watches your face for **~2 seconds** and calibrates to *your* neutral
face. After that, every expression is measured as a **deviation from your own
baseline**, not from a dataset average — so it adapts to *your* face shape.

```
 webcam frame ─▶ MediaPipe Face Mesh (468 + iris landmarks)
             ─▶ 7 geometric features extracted
             ─▶ minus YOUR neutral-face baseline
             ─▶ signed % meters (bars you can see live)
             ─▶ weighted mood scores + 6-frame hysteresis (no flicker)
             ─▶ banner 🏷 + colored border 🟨 + meters 🎚
```

| Your face does | Meter that moves | Moods it feeds |
|---|---|---|
| eyebrows rise | `BROWS UP` ☝️ | Surprised / Happy / Fear |
| brows knit together | `FROWN` 😠 | Angry / Sad / Disgust |
| mouth corners widen | `SMILE width` 😄 | Happy |
| corners rise above upper lip | `LIP corners up` | Happy |
| lips separate | `MOUTH OPEN` 😮 | Surprised / Disgust |
| eyes wide / squinted | `EYES wide/squint` 👀 | Surprised, Fear / Angry |
| left vs right brow mismatch | `BROW asymmetry` 🤨 | Confused |
| face is animated | `MOVEMENT` 📈 | intensity boost |

When you quit (`Q`), you get a **session report** of where your face spent its time:

```
=== session mood summary ===
  HAPPY       24s   32%  ##########
  ANGRY       18s   24%  #######
  NEUTRAL     15s   20%  ######
  SURPRISED    8s   10%  ###
  blinks this session: 14
```

---

## 📦 Installation (about 3 minutes)

### 0. Check yourself first
- **Python 3.9 – 3.12** → check with `python --version`
- a working **webcam**
- 💡 no GPU required, inference is CPU-only

> ⚠️ **Python 3.13+ won't work**: this project pins `mediapipe==0.10.21`, and
> that build has no wheels for 3.13. Use 3.9–3.12 (or install 3.11 alongside).

### 1. Clone the repo
```bash
git clone https://github.com/sanjay102432/face-mood-detector.git
cd face-mood-detector
```

### 2. (Recommended) virtual environment
```bash
python -m venv .venv

# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate
```

### 3. Install dependencies
```bash
python -m pip install -r requirements.txt
```
That installs the **known-good combo** (pinned in `requirements.txt`):
`mediapipe 0.10.21` + `opencv-python`. ~120 MB, one time only.

### 4. Sanity check (no camera needed)
```bash
python smoke_test.py    # face/hands/pose pipeline
python smoke_mood.py    # mood-math engine (neutral / open-mouth / smile tests)
```
Both should print `OK`. If they do, you're golden ✅

---

## 🚀 Run it

```bash
# the face + eyes + hands + body app
python detector.py

# the mood engine
python mood_detector.py

# you have more than one camera? choose it:
python detector.py --camera 1
python mood_detector.py --camera 1
```

A video window appears (mirrored like a phone selfie).
**In the mood app, first hold a neutral face ~2 s** until you see
*Calibrating 45/45 ✅*. Then perform! 😄😮😠🤨

Photo saved with `S` (git-ignored, stays on your machine).

---

## ⌨️ Controls

*(click the video window once so it has keyboard focus)*

| Key | In `detector.py` | In `mood_detector.py` |
|---|---|---|
| `F` | toggle face mesh + eyes + iris | toggle face lines (brows green, lips pink, eyes red) |
| `H` | toggle hand skeleton | toggle hand skeleton |
| `P` | toggle body-pose skeleton | — (not used) |
| `D` | toggle stats panel | toggle expressive-meters panel |
| `B` | — | **recalibrate to neutral face** |
| `S` | save screenshot | save screenshot (filename has the mood!) |
| `Q` / `ESC` | quit (prints session summary in mood app) | quit |

---

## 🖼 Add screenshots to this page (optional, takes 1 min)

1. Run either app, press `S` to capture the `detect_*.png` / `mood_*.png` files
2. Move the nicest ones into a `docs/` folder (they're regenerated with your face)
3. Commit them, then add under this line in the README:
   `![demo](docs/your-screenshot.png)`

*(kept out of Git by default so your face never auto-shares itself — your call to publish it)*

---

## 🛠 Troubleshooting

| Symptom | Fix |
|---|---|
| `Could not open camera` | Close Zoom/Teams/Snipping that grabbed it, or try `--camera 1` |
| Moods feel "stuck" on one face | Press `B` while making a neutral face |
| Banner jumps between moods too fast | It uses 6-frame hysteresis already; if still jumpy, sit more centered so the head is frontal |
| Low FPS | Press `H` (drops hands = cheaper), then `D` |
| `pip install` fails on Python 3.13 | Use Python 3.9–3.12 (see warning above) |
| Import error `mediapipe has no attribute solutions` | You grabbed MediaPipe 1.x — reinstall: `pip install -r requirements.txt` (pins 0.10.21) |
| First run looks laggy | Windows Defender / caches — the second run is smooth |

---

## 🗺 Ideas to build next (fork me!)

- 🕶 **Virtual try-on** — sunglasses that stick to the face-mesh's eye points
- 💪 **Rep counter** — squats/push-ups via the pose skeleton (already implemented in `detector.py`!)
- ✋ **Air-magic** — pinch fingers to snap a photo, raise two fingers to draw glowing trails
- 📱 **Go mobile** — same logic runs on Android via MediaPipe Tasks + ARCore for real 3D AR
- 🌦 **Mood journal** — append every session summary to `moods.csv` and chart your week

Pull requests welcome — the mood-weights live in one tiny function
(`score_moods`) begging to be tuned by you 😄

---

## 📁 Project structure

```
face-mood-detector/
├── detector.py       # face + eyes + iris + hands + body, live overlays
├── mood_detector.py  # 8-mood emption-like engine, meters, summary
├── smoke_test.py     # headless check #1 (pipelines run)
├── smoke_mood.py     # headless check #2 (mood math classifies right)
├── requirements.txt  # pinned, known-good versions
├── LICENSE           # MIT
└── README.md         # this file
```

---

## 🙌 Credits

Built on the shoulders of giants:
- [Google MediaPipe](https://developers.google.com/mediapipe) — the 468-point face mesh, hand & pose tracking
- [OpenCV](https://opencv.org) — camera I/O and rendering
- Mood engine: hand-rolled geometric heuristics (no dataset, no training, no cloud)

Made with ☕ and a laptop webcam by **Sanjay Bhoye** —
one of those "what can I actually build with TensorFlow/tooling?" weekends that got out of hand 😄

## 📜 License

MIT — free to use, remix, and ship. See [LICENSE](LICENSE).
