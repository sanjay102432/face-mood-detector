"""
MOOD DETECTOR - reads your mood from facial geometry (eyebrows, lips, eyes, jaw).

How it works:
    * MediaPipe Face Mesh (468 + iris landmarks) -> geometric features
    * Hold a NEUTRAL face for ~2 s at startup  (auto-calibration)
    * Deviations from YOUR baseline -> mood scores:
        HAPPY / SAD / ANGRY / SURPRISED / FEAR / DISGUST / CONFUSED / NEUTRAL
      plus per-feature meters (brows, frown, smile, mouth open, eyes, asymmetry)

Run:
    python mood_detector.py
    python mood_detector.py --camera 1

Keys (click the video window first):
    F = toggle face lines      H = toggle hands   D = toggle side panel
    B = recalibrate (make a neutral face again)
    S = save screenshot        Q / ESC = quit
Dependencies: pip install opencv-python mediapipe   (same as detector.py)
"""
import argparse
import datetime
import math
import pathlib
import statistics
import sys
import time

import cv2

try:
    import mediapipe as mp
except ImportError:
    print("[!] mediapipe is not installed.  Fix it with:   pip install mediapipe")
    sys.exit(1)

# ----------------------------------------------------------------- landmarks
mp_face_mesh = mp.solutions.face_mesh
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

# landmark indices used by the mood engine (468-point mesh + iris points)
I_IRIS_L, I_IRIS_R = 468, 473                        # fallback: 33, 263
EYE_PTS = dict(                                      # eye corners / lids
    l_outer=33, l_inner=133, l_upper=159, l_lower=145,
    r_inner=362, r_outer=263, r_upper=386, r_lower=374,
)
BROW = dict(l=105, r=334, l_inner=107, r_inner=336)  # brow tops + inner ends
MOUTH = dict(c_l=61, c_r=291, up=13, low=14)         # corners + inner lips

# drawing extras: brows in green, lips in pink, eyes/iris in red
BROW_LINED = (mp_face_mesh.FACEMESH_LEFT_EYEBROW, mp_face_mesh.FACEMESH_RIGHT_EYEBROW)
LIP_LINES = (mp_face_mesh.FACEMESH_LIPS,)
EYE_CONNECTIONS = (
    mp_face_mesh.FACEMESH_LEFT_EYE,
    mp_face_mesh.FACEMESH_RIGHT_EYE,
    mp_face_mesh.FACEMESH_LEFT_IRIS,
    mp_face_mesh.FACEMESH_RIGHT_IRIS,
)

WHITE = (250, 250, 250)
GRAY = (170, 170, 170)
GREEN = (80, 220, 80)
RED = (60, 60, 255)
CYAN = (255, 200, 0)
PINK = (200, 90, 230)

MOOD_COLORS = {                       # BGR
    "NEUTRAL": GRAY, "HAPPY": (30, 215, 235), "SAD": (255, 130, 10),
    "ANGRY": (60, 60, 255), "SURPRISED": (30, 130, 255),
    "FEAR": (230, 100, 255), "DISGUST": (60, 180, 60), "CONFUSED": (255, 255, 80),
}
WINDOW_TITLE = "MOOD DETECTOR  (F=lines  H=hands  D=panel  B=recalibrate  S=shot  Q=quit)"
OUT_DIR = pathlib.Path(__file__).resolve().parent
FONT = cv2.FONT_HERSHEY_SIMPLEX

FEAT_KEYS = ["brow", "inner", "smile", "lift", "open", "eyes", "asym"]


def px(lm, w, h):
    return lm.x * w, lm.y * h


def d2(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def clamp100(v):
    return max(-99, min(99, int(v)))


def tanh100(v, k):
    """squash a raw deviation into -99..99 %"""
    return clamp100(math.tanh(v * k) * 100)


# ------------------------------------------------------------- feature math
def extract_features(lms, w, h):
    """raw geometry -> stable, scale-free numbers (units: inter-iris distance)"""
    p = {k: px(lms[i], w, h) for k, i in
         ((k, i) for k, i in
          [("irl", I_IRIS_L), ("irr", I_IRIS_R),
           ("e_l_o", EYE_PTS["l_outer"]), ("e_l_i", EYE_PTS["l_inner"]),
           ("e_l_u", EYE_PTS["l_upper"]), ("e_l_d", EYE_PTS["l_lower"]),
           ("e_r_o", EYE_PTS["r_outer"]), ("e_r_i", EYE_PTS["r_inner"]),
           ("e_r_u", EYE_PTS["r_upper"]), ("e_r_d", EYE_PTS["r_lower"]),
           ("b_l", BROW["l"]), ("b_r", BROW["r"]),
           ("b_l_i", BROW["l_inner"]), ("b_r_i", BROW["r_inner"]),
           ("m_c_l", MOUTH["c_l"]), ("m_c_r", MOUTH["c_r"]),
           ("m_up", MOUTH["up"]), ("m_low", MOUTH["low"])])}

    if len(lms) > 473:                                   # prefer iris as scale ref
        scale = max(d2(p["irl"], p["irr"]), 1e-6)
    else:
        scale = max(d2(p["e_l_o"], p["e_r_o"]), 1e-6)

    eye_c_l = ((p["e_l_o"][0] + p["e_l_i"][0]) / 2, (p["e_l_o"][1] + p["e_l_i"][1]) / 2)
    eye_c_r = ((p["e_r_o"][0] + p["e_r_i"][0]) / 2, (p["e_r_o"][1] + p["e_r_i"][1]) / 2)

    brow_l = d2(p["b_l"], eye_c_l) / scale
    brow_r = d2(p["b_r"], eye_c_r) / scale
    inner = d2(p["b_l_i"], p["b_r_i"]) / scale
    smile = d2(p["m_c_l"], p["m_c_r"]) / scale
    lift = (p["m_up"][1] - 0.5 * (p["m_c_l"][1] + p["m_c_r"][1])) / scale
    mouth_open = d2(p["m_up"], p["m_low"]) / scale
    eye_h_l = max(d2(p["e_l_u"], p["e_l_d"]), 1e-6)
    eye_w_l = max(d2(p["e_l_o"], p["e_l_i"]), 1e-6)
    eye_h_r = max(d2(p["e_r_u"], p["e_r_d"]), 1e-6)
    eye_w_r = max(d2(p["e_r_o"], p["e_r_i"]), 1e-6)
    eyes = 0.5 * (eye_h_l / eye_w_l + eye_h_r / eye_w_r)   # open-ness ratio

    return dict(brow=0.5 * (brow_l + brow_r), inner=inner, smile=smile,
                lift=lift, open=mouth_open, eyes=eyes, asym=abs(brow_l - brow_r),
                ear=eyes)                              # ear kept raw for blink


def dev_to_pcts(raw, base):
    """raw - baseline  ->  signed percent meters per feature"""
    return dict(
        brow=tanh100((raw["brow"] / max(base["brow"], 1e-6) - 1.0), 9.0),
        furrow=tanh100(max((base["inner"] - raw["inner"]) / max(base["inner"], 1e-6), -0.6), 6.0),
        smile=tanh100((raw["smile"] / max(base["smile"], 1e-6) - 1.0), 7.0),
        lift=tanh100(raw["lift"] - base["lift"], 40.0),
        open=tanh100(raw["open"] - base["open"], 15.0),
        eyes=tanh100(raw["eyes"] - base["eyes"], 30.0),
        asym=tanh100(raw["asym"] - base["asym"], 25.0),
    )


def score_moods(pcts):
    """weighted meter percentages -> mood score dict (0..~1.4)"""
    S, L = max(pcts["smile"], 0), max(pcts["lift"], 0)
    U, F = max(pcts["brow"], 0), max(pcts["furrow"], 0)
    O, W = max(pcts["open"], 0), max(pcts["eyes"], 0)
    N, DL = max(-pcts["eyes"], 0), max(-pcts["lift"], 0)
    A = max(pcts["asym"], 0)

    scores = {
        "HAPPY": 0.0100 * S + 0.0090 * L + 0.0025 * min(U, 40),
        "SURPRISED": 0.0110 * O + 0.0100 * U + 0.0090 * W,
        "SAD": 0.0115 * DL + 0.0060 * F,
        "ANGRY": 0.0090 * F + 0.0060 * N + 0.0040 * DL,
        "FEAR": 0.0075 * U + 0.0060 * W + 0.0040 * O + 0.0030 * F,
        "DISGUST": 0.0070 * F + 0.0050 * N + 0.0060 * O + 0.0030 * DL,
        "CONFUSED": 0.0095 * A if (A > 30 and (S + F + O + U + W) < 90) else 0.0,
    }
    return scores


# ----------------------------------------------------------------- drawing
def hud(frame, lines):
    if not lines:
        return
    overlay = frame.copy()
    box_w = max(len(t) for t in lines) * 14 + 30
    cv2.rectangle(overlay, (8, 8), (box_w, 14 + 25 * len(lines)), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)
    for i, text in enumerate(lines):
        cv2.putText(frame, text, (18, 38 + 25 * i), FONT, 0.55, WHITE, 1, cv2.LINE_AA)


def banner(frame, w, label, intensity, held_s, mood_color, extra=""):
    text = f"{label}  {intensity}%"
    (tw, _), _ = cv2.getTextSize(text, FONT, 1.0, 2)
    x = max(20, (w - tw) // 2 - 18)
    overlay = frame.copy()
    cv2.rectangle(overlay, (x, 8), (x + tw + 36, 52), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)
    cv2.rectangle(frame, (x, 8), (x + tw + 36, 52), mood_color, 2)
    cv2.putText(frame, text, (x + 18, 42), FONT, 1.0, mood_color, 2, cv2.LINE_AA)
    sub = f"held {held_s:.1f}s   {extra}" if extra else f"held {held_s:.1f}s"
    cv2.putText(frame, sub, (x + 18, 72), FONT, 0.45, WHITE, 1, cv2.LINE_AA)


def meter(frame, x, y, name, pct, pos_color):
    half, hgt = 70, 12
    mid = x + half
    cv2.putText(frame, name, (x, y - 4), FONT, 0.45, WHITE, 1, cv2.LINE_AA)
    cv2.rectangle(frame, (x, y), (x + 2 * half, y + hgt), (90, 90, 90), 1)
    cv2.line(frame, (mid, y), (mid, y + hgt), (150, 150, 150), 1)
    if abs(pct) >= 3:
        width = int(pct / 100 * half)
        color = pos_color if pct > 0 else (140, 140, 140)
        cv2.rectangle(frame, (mid, y + 2), (mid + width, y + hgt - 2), color, -1)
    cv2.putText(frame, f"{pct:+d}", (x + 2 * half + 8, y + hgt - 2), FONT, 0.45, WHITE, 1, cv2.LINE_AA)


# ----------------------------------------------------------------- main app
def main():
    ap = argparse.ArgumentParser(description="Camera mood detector from facial geometry")
    ap.add_argument("--camera", type=int, default=0, help="camera index (0 = default)")
    args = ap.parse_args()

    show_lines = True
    show_hands = True
    show_panel = True

    backend = cv2.CAP_DSHOW if sys.platform.startswith("win") else 0
    cap = cv2.VideoCapture(args.camera, backend)
    if not cap.isOpened():
        cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(f"[!] Could not open camera {args.camera}. Close other camera apps and retry,")
        print("    or try:   python mood_detector.py --camera 1")
        sys.exit(1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    cv2.namedWindow(WINDOW_TITLE, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_TITLE, 1000, 740)
    print("Camera ready. HOLD A NEUTRAL FACE for ~2 s to calibrate. Q to quit.")

    fps = 0.0
    prev_time = None
    flash_until = 0.0
    pcts_smooth = {k: 0.0 for k in
                   ("brow", "furrow", "smile", "lift", "open", "eyes", "asym")}
    scores_smooth = {k: 0.0 for k in
                     ("HAPPY", "SURPRISED", "SAD", "ANGRY", "FEAR", "DISGUST", "CONFUSED")}
    prev_pcts = None
    activity = 0.0

    baseline = None
    baseline_vecs = []
    target_frames = 45
    mood_label, mood_since, mood_prev_logged = "CALIBRATING", None, None
    blinks = 0
    spent = {}
    votes = [None, 0]
    eye_state = dict(opened=True, closed_since=None, wink_since=None)
    last_face_time = None
    prev_logged = time.time()

    try:
        with mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True,
                                   min_detection_confidence=0.5,
                                   min_tracking_confidence=0.5) as face_mesh, \
             mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.5,
                            min_tracking_confidence=0.5) as hands_detector:

            while True:
                ok, frame = cap.read()
                if not ok:
                    print("[!] Camera stopped returning frames. Re-plug it and restart.")
                    break
                frame = cv2.flip(frame, 1)
                h, w = frame.shape[:2]
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                rgb.flags.writeable = False

                mesh_result = face_mesh.process(rgb)
                hand_result = hands_detector.process(rgb) if show_hands else None

                faces = len(mesh_result.multi_face_landmarks) \
                    if (mesh_result and mesh_result.multi_face_landmarks) else 0
                face_seen = faces > 0
                hand_count = len(hand_result.multi_hand_landmarks) \
                    if (show_hands and hand_result and hand_result.multi_hand_landmarks) else 0
                if face_seen:
                    last_face_time = time.time()

                # ---------------- calibration state machine ----------------
                if face_seen and baseline is None:
                    lms = mesh_result.multi_face_landmarks[0].landmark
                    baseline_vecs.append(extract_features(lms, w, h))
                    if len(baseline_vecs) >= target_frames:
                        baseline = {k: statistics.median(v[k] for v in baseline_vecs)
                                    for k in FEAT_KEYS}
                        print("[OK] Calibrated. Now show moods!  (B = recalibrate)")
                elif not face_seen and baseline is None:
                    baseline_vecs.clear()          # restart calibration if face lost

                # ---------------- mood analysis ----------------
                mood_color = GRAY
                if face_seen and baseline is not None:
                    lms = mesh_result.multi_face_landmarks[0].landmark
                    raw = extract_features(lms, w, h)
                    pcts = dev_to_pcts(raw, baseline)
                    scores = score_moods(pcts)

                    for k in pcts:                       # smooth meters
                        pcts_smooth[k] = 0.70 * pcts_smooth[k] + 0.30 * pcts[k]
                    for k in scores:                     # smooth scores
                        scores_smooth[k] = 0.72 * scores_smooth[k] + 0.28 * scores[k]

                    if prev_pcts is not None:            # 'movement' meter
                        step = sum(abs(pcts[k] - prev_pcts[k]) for k in pcts) / len(pcts)
                        activity = 0.90 * activity + 0.10 * min(step, 20.0)
                    prev_pcts = pcts

                    top = max(scores_smooth, key=scores_smooth.get)
                    intensity = clamp100(scores_smooth[top] * 75)
                    if scores_smooth[top] < 0.30:
                        mood = "NEUTRAL"
                    else:
                        mood = top

                    # ---------- blinks / wink (raw eye open-ness) ----------
                    now = time.time()
                    both_open = raw["eyes"]
                    if both_open < 0.14:
                        if eye_state["opened"]:
                            eye_state["closed_since"] = now
                        eye_state["opened"] = False
                    else:
                        if not eye_state["opened"] and eye_state["closed_since"] \
                           and now - eye_state["closed_since"] < 0.5:
                            blinks += 1
                        eye_state["opened"] = True
                        eye_state["closed_since"] = None

                    # ---------------- face overlay lines ----------------
                    if show_lines:
                        mp_draw.draw_landmarks(frame, mesh_result.multi_face_landmarks[0],
                                               mp_face_mesh.FACEMESH_CONTOURS, None,
                                               mp_draw.DrawingSpec(color=(120, 190, 250), thickness=1))
                        for conn in BROW_LINED:
                            mp_draw.draw_landmarks(frame, mesh_result.multi_face_landmarks[0], conn,
                                                   None, mp_draw.DrawingSpec(color=GREEN, thickness=2))
                        for conn in LIP_LINES:
                            mp_draw.draw_landmarks(frame, mesh_result.multi_face_landmarks[0], conn,
                                                   None, mp_draw.DrawingSpec(color=PINK, thickness=2))
                        for conn in EYE_CONNECTIONS:
                            mp_draw.draw_landmarks(frame, mesh_result.multi_face_landmarks[0], conn,
                                                   None, mp_draw.DrawingSpec(color=RED, thickness=2))

                    # ---------------- banner + mood border ----------------
                    # hysteresis: a new mood must lead several frames to win
                    # the banner, so it doesn't flicker every half-second
                    if mood == mood_label:
                        votes = [None, 0]
                    else:
                        if votes[0] == mood:
                            votes[1] += 1
                        else:
                            votes = [mood, 1]
                        if votes[1] >= 6:
                            if mood_since is not None and mood_label != "CALIBRATING":
                                spent[mood_label] = spent.get(mood_label, 0.0) + now - mood_since
                            mood_label, mood_since = mood, now
                            if mood_label != "NEUTRAL":
                                print(f"[mood] {mood_label}  {intensity}%")
                                mood_prev_logged = mood_label
                            votes = [None, 0]
                    held = max(0.0, now - mood_since) if mood_since else 0.0
                    mood_color = MOOD_COLORS.get(mood_label, GRAY)
                    extra = "eyes closed" if (not eye_state["opened"] and
                                              now - (eye_state["closed_since"] or now) > 0.5) else ""
                    banner(frame, w, mood_label, intensity, held, mood_color, extra)
                    border_width = 10 if intensity > 60 else 4
                    cv2.rectangle(frame, (2, 2), (w - 3, h - 3), mood_color, border_width)

                    # ---------------- side meters ----------------
                    if show_panel:
                        ox = w - 320
                        oy = 96
                        overlay = frame.copy()
                        cv2.rectangle(overlay, (ox - 104, oy - 22), (ox - 8 + 260, oy + 20 + 8 * 30), (15, 15, 15), -1)
                        cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)
                        rows = [
                            ("BROWS UP", pcts_smooth, "brow", GREEN),
                            ("FROWN (furrow)", pcts_smooth, "furrow", RED),
                            ("SMILE width", pcts_smooth, "smile", GREEN),
                            ("LIP corners up", pcts_smooth, "lift", GREEN),
                            ("MOUTH OPEN", pcts_smooth, "open", CYAN),
                            ("EYES wide / squint", pcts_smooth, "eyes", CYAN),
                            ("BROW asymmetry", pcts_smooth, "asym", CYAN),
                        ]
                        cv2.putText(frame, "EXPRESSIVE FEATURES (vs your neutral face)", (ox - 100, oy - 4),
                                    FONT, 0.44, WHITE, 1, cv2.LINE_AA)
                        for i, (name, _, key, col) in enumerate(rows):
                            meter(frame, ox - 100, oy + 24 + i * 30, name, int(pcts_smooth[key]), col)
                        meter(frame, ox - 100, oy + 24 + len(rows) * 30, "MOVEMENT (activity)",
                              int(clamp100(activity * 8)), GRAY)

                        top3 = sorted(pcts_smooth.items(), key=lambda kv: -abs(kv[1]))[:4]
                        contrib = "  ".join(f"{k.upper()}:{int(v)}" for k, v in top3 if abs(v) > 8)
                        if contrib:
                            cv2.putText(frame, f"movers: {contrib}", (ox - 100, oy + 22 + (len(rows) + 1) * 30),
                                        FONT, 0.45, WHITE, 1, cv2.LINE_AA)
                else:
                    if baseline is None:
                        done = min(len(baseline_vecs), target_frames)
                        msg = f"Calibrating {done}/{target_frames} - hold a NEUTRAL face..."
                    else:
                        msg = "NO FACE - step in front of the camera"
                    overlay = frame.copy()
                    cv2.rectangle(overlay, (0, 8), (w, 60), (0, 0, 0), -1)
                    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)
                    cv2.putText(frame, msg, (16, 42), FONT, 0.8, WHITE, 2, cv2.LINE_AA)

                hands_text = f"Hands {hand_count}" if show_hands else "Hands off"
                closed_after = False

                # ---------------- stats ----------------
                now = time.time()
                if show_panel or True:
                    hud(frame, [
                        "MOOD DETECTOR",
                        f"FPS {fps:4.1f}   Faces {faces}  {hands_text}   Blinks {blinks}",
                        "F lines | H hands | D panel | B recalib | S shot | Q quit",
                    ])

                cv2.imshow(WINDOW_TITLE, frame)

                # ---------------- keyboard ----------------
                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord("q")):
                    break
                if key == ord("f"):
                    show_lines = not show_lines
                if key == ord("h"):
                    show_hands = not show_hands
                if key == ord("d"):
                    show_panel = not show_panel
                if key == ord("b"):
                    baseline, baseline_vecs = None, []
                    mood_label, mood_since = "CALIBRATING", None
                    scores_smooth = {k: 0.0 for k in scores_smooth}
                    print("[i] Recalibrating - hold a NEUTRAL face.")
                if key == ord("s"):
                    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                    path = OUT_DIR / f"mood_{stamp}_{mood_label}.png"
                    cv2.imwrite(str(path), frame)
                    print(f"[OK] Screenshot saved: {path.name}")
                    flash_until = time.time() + 0.35
                if time.time() < flash_until:
                    cv2.rectangle(frame, (0, 0), (w - 1, h - 1), GREEN, 12)

                closed_after = cv2.getWindowProperty(WINDOW_TITLE, cv2.WND_PROP_VISIBLE) < 1

                if prev_time is not None and now > prev_time:
                    fps = 0.9 * fps + 0.1 * (1.0 / (now - prev_time))
                prev_time = now
                if closed_after:
                    break
    finally:
        cap.release()
        cv2.destroyAllWindows()
        if mood_since is not None and mood_label not in ("CALIBRATING",):
            spent[mood_label] = spent.get(mood_label, 0.0) + time.time() - mood_since
        if spent:
            total = max(sum(spent.values()), 1e-6)
            print("\n=== session mood summary ===")
            for k, v in sorted(spent.items(), key=lambda kv: -kv[1]):
                bar = "#" * max(1, int(v / total * 30))
                print(f"  {k:<10} {int(v):>4}s  {int(v / total * 100):>3}%  {bar}")
            print(f"  blinks this session: {blinks}")
        print("Bye!")


if __name__ == "__main__":
    main()
