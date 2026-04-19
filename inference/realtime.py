import cv2
import numpy as np
import torch
from models.full_model import FullModel
import mediapipe as mp

# -------- MODEL --------
model = FullModel()
model.load_state_dict(torch.load("model.pth", map_location="cpu"))
model.eval()

SEQ_LEN = 5
sequence = []
predictions = []

emotion_map = {
    0: "Happy",
    1: "Disgust",
    2: "Surprise"
}

# -------- MEDIAPIPE --------
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils

face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True
)

# -------- CAMERA --------
cap = cv2.VideoCapture(0)

# FULLSCREEN WINDOW
cv2.namedWindow("AI Mirror", cv2.WINDOW_NORMAL)
cv2.setWindowProperty("AI Mirror", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

mesh_style = mp_drawing.DrawingSpec(
    color=(255, 255, 255),
    thickness=1,
    circle_radius=1
)

prev_gray = None
final_label = 0

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # MIRROR EFFECT
        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape

        # ================= MOTION =================
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if prev_gray is not None:
            diff = cv2.absdiff(prev_gray, gray)
            _, thresh = cv2.threshold(diff, 8, 255, cv2.THRESH_BINARY)
            thresh = cv2.medianBlur(thresh, 7)
            thresh = cv2.dilate(thresh, None, iterations=1)
        else:
            thresh = np.zeros((h, w), dtype=np.uint8)

        prev_gray = gray

        # ================= FACE MESH =================
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb)

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:

                x_list, y_list = [], []

                for lm in face_landmarks.landmark:
                    x_list.append(int(lm.x * w))
                    y_list.append(int(lm.y * h))

                x_min, x_max = min(x_list), max(x_list)
                y_min, y_max = min(y_list), max(y_list)

                if x_max > x_min and y_max > y_min:

                    face_crop = frame[y_min:y_max, x_min:x_max]
                    face_crop = cv2.resize(face_crop, (128, 128))

                    img = face_crop / 255.0
                    img = np.transpose(img, (2, 0, 1))

                    sequence.append(img)

                    if len(sequence) == SEQ_LEN:

                        seq_np = np.array(sequence)
                        seq_np = np.expand_dims(seq_np, axis=0)
                        seq = torch.tensor(seq_np, dtype=torch.float32)

                        with torch.no_grad():
                            pred = model(seq)

                            probs = torch.softmax(pred, dim=1)
                            confidence, label = torch.max(probs, dim=1)

                            label = label.item()
                            confidence = confidence.item()

                        # CONFIDENCE FILTER
                        if confidence > 0.55:
                            predictions.append(label)

                            if len(predictions) > 3:
                                predictions.pop(0)

                            final_label = max(set(predictions), key=predictions.count)

                        emotion_text = emotion_map.get(final_label, str(final_label))

                        # DISPLAY TEXT
                        cv2.putText(frame,
                                    f"{emotion_text} ({confidence*100:.1f}%)",
                                    (50, 80),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    1.5,
                                    (0, 255, 0),
                                    3)

                        sequence.pop(0)

                # DRAW FACE MESH
                overlay = frame.copy()

                mp_drawing.draw_landmarks(
                    overlay,
                    face_landmarks,
                    mp_face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=mesh_style,
                    connection_drawing_spec=mesh_style
                )

                frame = cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)

        # ================= FLOATING MOTION =================
        motion_small = cv2.resize(thresh, (200, 150))
        motion_small = cv2.cvtColor(motion_small, cv2.COLOR_GRAY2BGR)

        frame[20:170, w-220:w-20] = motion_small

        cv2.putText(frame, "Motion",
                    (w-200, 15),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 255),
                    2)

        # ================= DISPLAY =================
        cv2.imshow("AI Mirror", frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break

except KeyboardInterrupt:
    print("Stopped by user")

cap.release()
cv2.destroyAllWindows()