"""
Face / Eyes / Hands / Body Pose  —  live detector using your laptop camera.

One-time setup:
    pip install opencv-python mediapipe

Run:
    python detector.py
    python detector.py --camera 1        (if you have several cameras)

Hotkeys (first CLICK on the video window, then press):
    F = toggle Face + Eyes + Iris
    H = toggle Hand skeleton
    P = toggle Body pose skeleton
    D = toggle stats panel
    S = save a screenshot (PNG saved next to this file)
    Q or ESC = quit
"""
import argparse
import datetime
import pathlib
import sys
import time

import cv2

try:
    import mediapipe as mp
except ImportError:
    print("[!] mediapipe is not installed.  Fix it with:   pip install mediapipe")
    sys.exit(1)

# ----------------------------------------------------------------- setup
mp_face_mesh = mp.solutions.face_mesh
mp_hands = mp.solutions.hands
mp_pose = mp.solutions.pose
mp_draw = mp.solutions.drawing_utils
mp_styles = mp.solutions.drawing_styles          # ready-made drawing styles

# eye outline + iris rings, drawn on top of the face mesh
EYE_CONNECTIONS = (
    mp_face_mesh.FACEMESH_LEFT_EYE,
    mp_face_mesh.FACEMESH_RIGHT_EYE,
    mp_face_mesh.FACEMESH_LEFT_IRIS,
    mp_face_mesh.FACEMESH_RIGHT_IRIS,
)

HAND_LAND_STYLE = mp_styles.get_default_hand_landmarks_style()
HAND_CONN_STYLE = mp_styles.get_default_hand_connections_style()
POSE_STYLE = mp_styles.get_default_pose_landmarks_style()

GREEN = (80, 220, 80)
RED = (60, 60, 255)
WHITE = (250, 250, 250)
WINDOW_TITLE = "AR Camera Detector  (F=face  H=hands  P=pose  S=shot  Q=quit)"

OUT_DIR = pathlib.Path(__file__).resolve().parent


def pixel(landmark, w, h):
    """MediaPipe normalized landmark -> pixel x, y."""
    return int(landmark.x * w), int(landmark.y * h)


def hud(frame, lines):
    """Semi-transparent stats panel in the top-left corner."""
    if not lines:
        return
    overlay = frame.copy()
    box_w = max(len(t) for t in lines) * 14 + 30      # rough text width
    cv2.rectangle(overlay, (8, 8), (box_w, 14 + 25 * len(lines)), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)
    for i, text in enumerate(lines):
        cv2.putText(frame, text, (18, 38 + 25 * i),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, WHITE, 1, cv2.LINE_AA)


def main():
    parser = argparse.ArgumentParser(description="Live face / eyes / hands / pose detector")
    parser.add_argument("--camera", type=int, default=0, help="camera index (0 = default)")
    args = parser.parse_args()

    show_face = show_hands = show_pose = show_stats = True
    fps = 0.0
    prev_time = None
    flash_until = 0.0

    # open the camera (DSHOW backend starts fast on Windows, default backend as fallback)
    backend = cv2.CAP_DSHOW if sys.platform.startswith("win") else 0
    cap = cv2.VideoCapture(args.camera, backend)
    if not cap.isOpened():
        cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(f"[!] Could not open camera {args.camera}.")
        print("    Close other apps using it (Zoom/Teams/Browser) and retry,")
        print("    or try another index:   python detector.py --camera 1")
        sys.exit(1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    cv2.namedWindow(WINDOW_TITLE, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_TITLE, 960, 720)
    print("Camera ready. Press F / H / P / D to toggle features, S for screenshot, Q to quit.")

    try:
        with mp_face_mesh.FaceMesh(max_num_faces=1,
                                   refine_landmarks=True,     # enables iris rings
                                   min_detection_confidence=0.5,
                                   min_tracking_confidence=0.5) as face_mesh, \
             mp_hands.Hands(max_num_hands=2,
                            min_detection_confidence=0.5,
                            min_tracking_confidence=0.5) as hands_detector, \
             mp_pose.Pose(min_detection_confidence=0.5,
                          min_tracking_confidence=0.5) as pose_detector:

            while True:
                ok, frame = cap.read()
                if not ok:
                    print("[!] Camera stopped returning frames. Re-plug it and restart.")
                    break

                frame = cv2.flip(frame, 1)                  # mirror (phone-selfie style)
                h, w = frame.shape[:2]
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                rgb.flags.writeable = False                 # faster inference

                mesh_result = face_mesh.process(rgb) if show_face else None
                hand_result = hands_detector.process(rgb) if show_hands else None
                pose_result = pose_detector.process(rgb) if show_pose else None

                # ---------------- face + eyes + iris ----------------
                faces = len(mesh_result.multi_face_landmarks) \
                    if (show_face and mesh_result and mesh_result.multi_face_landmarks) else 0
                if faces:
                    for face_landmarks in mesh_result.multi_face_landmarks:
                        # light outline of the whole face
                        mp_draw.draw_landmarks(frame, face_landmarks,
                                               mp_face_mesh.FACEMESH_CONTOURS, None,
                                               mp_draw.DrawingSpec(color=(120, 190, 250), thickness=1))
                        # eyes + iris on top, thicker
                        for conn in EYE_CONNECTIONS:
                            mp_draw.draw_landmarks(frame, face_landmarks, conn, None,
                                                   mp_draw.DrawingSpec(color=RED, thickness=2))

                # ---------------- hands with skeleton ----------------
                hand_count = 0
                hand_labels = []
                if show_hands and hand_result and hand_result.multi_hand_landmarks:
                    hand_count = len(hand_result.multi_hand_landmarks)
                    for idx, hand_lms in enumerate(hand_result.multi_hand_landmarks):
                        mp_draw.draw_landmarks(frame, hand_lms, mp_hands.HAND_CONNECTIONS,
                                               HAND_LAND_STYLE, HAND_CONN_STYLE)
                        label = "?"
                        if hand_result.multi_handedness and idx < len(hand_result.multi_handedness):
                            raw = hand_result.multi_handedness[idx].classification[0].label
                            label = "L" if raw == "Left" else "R"
                        hand_labels.append(label)
                        # write the label above the wrist
                        wx, wy = pixel(hand_lms.landmark[0], w, h)  # landmark 0 = wrist
                        cv2.putText(frame, f"{label}-hand", (wx - 25, wy - 20),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, GREEN, 2, cv2.LINE_AA)

                # ---------------- body pose skeleton ----------------
                body_found = bool(show_pose and pose_result and pose_result.pose_landmarks)
                if body_found:
                    mp_draw.draw_landmarks(frame, pose_result.pose_landmarks,
                                           mp_pose.POSE_CONNECTIONS, POSE_STYLE)

                # ---------------- screenshot flash ----------------
                if time.time() < flash_until:
                    cv2.rectangle(frame, (0, 0), (w - 1, h - 1), GREEN, 12)

                # ---------------- stats panel ----------------
                if show_stats:
                    hud(frame, [
                        "AR CAMERA DETECTOR",
                        f"FPS {fps:4.1f}   Faces {faces}   Hands {hand_count}"
                        f" ({','.join(hand_labels) if hand_labels else '-'})   Body {'ON' if body_found else '--'}",
                        f"face {'ON' if show_face else 'off'} | hands {'ON' if show_hands else 'off'}"
                        f" | pose {'ON' if show_pose else 'off'} | S shot | Q quit",
                    ])

                cv2.imshow(WINDOW_TITLE, frame)

                # ---------------- keyboard ----------------
                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord("q")):
                    break
                if key == ord("f"):
                    show_face = not show_face
                if key == ord("h"):
                    show_hands = not show_hands
                if key == ord("p"):
                    show_pose = not show_pose
                if key == ord("d"):
                    show_stats = not show_stats
                if key == ord("s"):
                    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                    path = OUT_DIR / f"detect_{stamp}.png"
                    cv2.imwrite(str(path), frame)
                    print(f"[OK] Screenshot saved: {path}")
                    flash_until = time.time() + 0.35
                closed_after = cv2.getWindowProperty(WINDOW_TITLE, cv2.WND_PROP_VISIBLE) < 1

                # ---------------- fps smoothing ----------------
                now = time.time()
                if prev_time is not None and now > prev_time:
                    fps = 0.9 * fps + 0.1 * (1.0 / (now - prev_time))
                prev_time = now

                if closed_after:
                    break
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("Bye!")


if __name__ == "__main__":
    main()
