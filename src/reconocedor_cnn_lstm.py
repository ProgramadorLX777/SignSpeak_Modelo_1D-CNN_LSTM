import cv2
import mediapipe as mp
import numpy as np
import torch
import joblib
import time
from collections import deque
import torch.nn as nn

MODEL_PATH = "models/modelo_cnn_lstm_bimanual.pth"
LABELS_PATH = "models/labels_bimano.pkl"

SEQ_LEN = 50
FEATURES = 126

UMBRAL_CONF = 0.70
VENTANA_ESTABILIDAD = 5
TIEMPO_DESAPARECER = 2.0
UMBRAL_MOVIMIENTO = 0.01  # 🔥 detector de inicio de seña


mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
id_to_label = joblib.load(LABELS_PATH)

# -------------------------
# MODELO (igual que en el entrenador mejorado)
# -------------------------
class CNN1D_LSTM(nn.Module):
    def __init__(self, features, hidden_lstm=128, num_classes=2):
        super().__init__()

        self.conv1 = nn.Conv1d(features, 64, 3, padding=1)
        self.bn1 = nn.BatchNorm1d(64)

        self.conv2 = nn.Conv1d(64, 128, 3, padding=1)
        self.bn2 = nn.BatchNorm1d(128)

        self.conv3 = nn.Conv1d(128, 128, 3, padding=1)
        self.bn3 = nn.BatchNorm1d(128)

        self.relu = nn.ReLU()

        self.lstm = nn.LSTM(
            128,
            hidden_lstm,
            batch_first=True,
            bidirectional=True
        )

        self.dropout = nn.Dropout(0.35)
        self.fc = nn.Linear(hidden_lstm * 2, num_classes)

    def forward(self, x):
        x = x.permute(0, 2, 1)

        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.relu(self.bn3(self.conv3(x)))

        x = x.permute(0, 2, 1)
        x, _ = self.lstm(x)

        x = x[:, -1, :]
        x = self.dropout(x)
        return self.fc(x)

model = CNN1D_LSTM(FEATURES, 128, len(id_to_label)).to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

# -------------------------
# FUNCIONES
# -------------------------
def dibujar_manos_colores(frame, hand_landmarks):
    for connection in mp_hands.HAND_CONNECTIONS:
        x1 = int(hand_landmarks.landmark[connection[0]].x * frame.shape[1])
        y1 = int(hand_landmarks.landmark[connection[0]].y * frame.shape[0])
        x2 = int(hand_landmarks.landmark[connection[1]].x * frame.shape[1])
        y2 = int(hand_landmarks.landmark[connection[1]].y * frame.shape[0])
        cv2.line(frame, (x1, y1), (x2, y2), (255, 255, 0), 2)

    for idx, lm in enumerate(hand_landmarks.landmark):
        x = int(lm.x * frame.shape[1])
        y = int(lm.y * frame.shape[0])
        if idx in [4,8,12,16,20]:
            color = (0,255,0)
        elif idx in [3,7,11,15,19]:
            color = (255,0,0)
        elif idx == 0:
            color = (0,0,255)
        else:
            color = (0,165,255)
        cv2.circle(frame, (x,y), 5, color, -1)

def extraer_landmarks_ordenados(result):
    mano_left = np.zeros((21,3))
    mano_right = np.zeros((21,3))

    if not result.multi_hand_landmarks:
        return mano_left.flatten(), mano_right.flatten(), False

    for hl in result.multi_hand_landmarks:
        dibujar_manos_colores(frame_draw, hl)

    if getattr(result, "multi_hand_landmarks", None) and getattr(result, "multi_handedness", None):
        for lm, hand_h in zip(result.multi_hand_landmarks, result.multi_handedness):
            pts = np.array([[p.x, p.y, p.z] for p in lm.landmark])
            label = hand_h.classification[0].label
            if label == "Left":
                mano_left = pts
            else:
                mano_right = pts
        manos_detectadas = (not np.allclose(mano_left, 0)) and (not np.allclose(mano_right, 0))
        return mano_left.flatten(), mano_right.flatten(), manos_detectadas
    else:
        landmarks_list = [np.array([[p.x, p.y, p.z] for p in lm.landmark]) for lm in result.multi_hand_landmarks]

        if len(landmarks_list) == 2:
            xs = [pts[:,0].mean() for pts in landmarks_list]
            if xs[0] <= xs[1]:
                return landmarks_list[0].flatten(), landmarks_list[1].flatten(), True
            else:
                return landmarks_list[1].flatten(), landmarks_list[0].flatten(), True

        if len(landmarks_list) == 1:
            xs = landmarks_list[0][:,0].mean()
            if xs < 0.5:
                return landmarks_list[0].flatten(), mano_right.flatten(), False
            else:
                return mano_left.flatten(), landmarks_list[0].flatten(), False

        return mano_left.flatten(), mano_right.flatten(), False

# -------------------------
# RECONOCIMIENTO
# -------------------------
window = deque(maxlen=SEQ_LEN)
prob_history = deque(maxlen=10)   # 🔥 suavizado de probabilidad
cap = cv2.VideoCapture(0)

ultimo_timestamp = time.time()
ultimo_resultado = ""
historial = []
estado_actual = ""

print("Reconociendo (CNN1D+BiLSTM) ...")

prev_vec = None  # 🔥 para medir movimiento entre frames

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_draw = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame_draw, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    left_vec, right_vec, manos_detectadas = extraer_landmarks_ordenados(result)
    vec = np.concatenate([left_vec, right_vec])
    '''window.append(vec)'''
    
    # -------------------------
    # 🔥 DETECTOR DE MOVIMIENTO
    # -------------------------
    movimiento = 0.0
    if prev_vec is not None:
        movimiento = np.mean(np.abs(vec - prev_vec))
    prev_vec = vec

    if movimiento > UMBRAL_MOVIMIENTO:
        window.append(vec)
    '''else:
        window.clear()
        prob_history.clear()
        historial = []'''

    ahora = time.time()

    if not manos_detectadas:
        ultimo_resultado = "NO DETECTADO"
        historial = []
        ultimo_timestamp = ahora
    elif len(window) > 5 and len(window) < SEQ_LEN:
        estado_actual = "CAPTURANDO..."


    if len(window) == SEQ_LEN:
        seq = np.array(window, dtype=np.float32)
        seq = (seq - seq.mean()) / (seq.std() + 1e-6)

        tensor = torch.tensor(seq).unsqueeze(0).to(device)

        with torch.no_grad():
            out = model(tensor)
            probs = torch.softmax(out, dim=1)[0]

        prob_history.append(probs.cpu().numpy())

        avg_probs = np.mean(prob_history, axis=0)
        pred_id = int(np.argmax(avg_probs))
        conf = avg_probs[pred_id]

        pred_label = id_to_label[pred_id]

        if conf >= UMBRAL_CONF:
            historial.append(pred_label)
            estado_actual = "DETECTANDO..."

            if historial.count(pred_label) >= VENTANA_ESTABILIDAD:
                ultimo_resultado = pred_label
                ultimo_timestamp = ahora
                historial = []
                estado_actual = ""

        else:
            historial = []
            estado_actual = ""

    if ahora - ultimo_timestamp > TIEMPO_DESAPARECER:
        ultimo_resultado = ""

    cv2.putText(frame_draw, ultimo_resultado, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
    
    if estado_actual:
        cv2.putText(frame_draw, estado_actual, (10, 65),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)


    cv2.imshow("Reconocedor CNN1D+LSTM", frame_draw)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
