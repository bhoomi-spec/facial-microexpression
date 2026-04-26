import cv2
import torch
import numpy as np
import mediapipe as mp
from collections import deque

from models.full_model import FullModel

# ---------------- MODEL ----------------
model = FullModel()
model.load_state_dict(torch.load("model.pth", map_location="cpu"))
model.eval()

emotion_map = {
    0: "Surprised",
    1: "Sad",
    2: "Neutral",
    3: "Happy",
    4: "Fear",
    5: "Disgust",
    6: "Anger"
}

# ---------------- MEDIAPIPE ----------------
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils

face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6
)

mesh_style = mp_drawing.DrawingSpec(
    color=(0, 255, 0),
    thickness=1,
    circle_radius=1
)

# ---------------- CAMERA ----------------
cap = cv2.VideoCapture(0)

cv2.namedWindow("AI Mirror", cv2.WINDOW_NORMAL)
cv2.setWindowProperty(
    "AI Mirror",
    cv2.WND_PROP_FULLSCREEN,
    cv2.WINDOW_FULLSCREEN
)

# ---------------- SMOOTHING ----------------
history = deque(maxlen=25)
stable_label = "Thinking..."
stable_conf = 0.0

# ---------------- MOTION ----------------
prev_gray = None

# ---------------- LOOP ----------------
while True:
    ret, frame = cap.read()

    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape

    # =====================================
    # MOTION DETECTION
    # =====================================
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    if prev_gray is None:
        motion = np.zeros((h, w), dtype=np.uint8)
    else:
        diff = cv2.absdiff(prev_gray, gray)

        # increased threshold
        _, motion = cv2.threshold(diff, 18, 255, cv2.THRESH_BINARY)

        motion = cv2.GaussianBlur(motion, (5, 5), 0)
        motion = cv2.dilate(motion, None, iterations=2)

    prev_gray = gray.copy()

    # =====================================
    # FACE MESH + EMOTION
    # =====================================
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:

            xs = []
            ys = []

            for lm in face_landmarks.landmark:
                xs.append(int(lm.x * w))
                ys.append(int(lm.y * h))

            x1 = max(min(xs) - 20, 0)
            y1 = max(min(ys) - 20, 0)
            x2 = min(max(xs) + 20, w)
            y2 = min(max(ys) + 20, h)

            face = frame[y1:y2, x1:x2]

            if face.size > 0:
                face = cv2.resize(face, (128, 128))
                img = face.astype(np.float32) / 255.0
                img = np.transpose(img, (2, 0, 1))
                img = np.expand_dims(img, axis=0)

                img = torch.tensor(img, dtype=torch.float32)

                with torch.no_grad():
                    pred = model(img)
                    probs = torch.softmax(pred, dim=1)

                    conf, label = torch.max(probs, dim=1)

                    conf = conf.item()
                    label = label.item()

                # stronger confidence filter
                if conf > 0.55:
                    history.append(label)

                if len(history) > 0:
                    final = max(set(history), key=history.count)
                    stable_label = emotion_map[final]
                    stable_conf = conf

            # CLEAN FACE MESH
            overlay = frame.copy()

            mp_drawing.draw_landmarks(
                overlay,
                face_landmarks,
                mp_face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=mesh_style,
                connection_drawing_spec=mesh_style
            )

            frame = cv2.addWeighted(overlay, 0.65, frame, 0.35, 0)

    # =====================================
    # TEXT
    # =====================================
    cv2.putText(
        frame,
        f"{stable_label} {stable_conf*100:.1f}%",
        (40, 70),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.3,
        (0, 255, 0),
        3
    )

    # =====================================
    # FLOATING MOTION CAM
    # =====================================
    motion_small = cv2.resize(motion, (220, 160))
    motion_small = cv2.cvtColor(motion_small, cv2.COLOR_GRAY2BGR)

    frame[20:180, w-240:w-20] = motion_small

    cv2.rectangle(frame, (w-240, 20), (w-20, 180), (0, 255, 255), 2)

    cv2.putText(
        frame,
        "Motion",
        (w-210, 15),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 255),
        2
    )

    # =====================================
    # DISPLAY
    # =====================================
    cv2.imshow("AI Mirror", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()