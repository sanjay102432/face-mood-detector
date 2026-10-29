"""Quick sanity check: runs the face/hands/pose pipelines on a fake image.
No camera needed. If this prints OK, the real detector will work."""
import numpy as np
import cv2
import mediapipe as mp

img = np.full((480, 640, 3), (160, 180, 200), np.uint8)   # fake frame
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img.flags.writeable = False

with mp.solutions.face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True) as fm, \
     mp.solutions.hands.Hands() as hnd, \
     mp.solutions.pose.Pose() as ps:
    r1 = fm.process(img)
    r2 = hnd.process(img)
    r3 = ps.process(img)

print("OK - face mesh / hands / pose all process fine on this Python.")
