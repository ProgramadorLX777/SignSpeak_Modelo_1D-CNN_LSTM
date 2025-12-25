import os
import numpy as np
import torch
import torch.nn as nn
import joblib
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

# =============================
# CONFIGURACIÓN
# =============================
DATA_PATH = "data_cnn_lstm_bimano"
MODEL_PATH = "models/modelo_cnn_lstm_bimanual.pth"
LABELS_PATH = "models/labels_bimano.pkl"

SEQ_LEN = 50
FEATURES = 126

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =============================
# CARGAR LABELS
# =============================
id_to_label = joblib.load(LABELS_PATH)
label_to_id = {v: k for k, v in id_to_label.items()}
num_classes = len(id_to_label)

print("Clases:", id_to_label)

# =============================
# MODELO (MISMO QUE ENTRENAMIENTO)
# =============================
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

model = CNN1D_LSTM(FEATURES, 128, num_classes).to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

# =============================
# CARGAR DATASET
# =============================
X = []
y_true = []

for label in os.listdir(DATA_PATH):
    class_dir = os.path.join(DATA_PATH, label)
    if not os.path.isdir(class_dir):
        continue

    for file in os.listdir(class_dir):
        if file.endswith(".npy"):
            seq = np.load(os.path.join(class_dir, file))

            if seq.shape == (SEQ_LEN, FEATURES):
                X.append(seq)
                y_true.append(label_to_id[label])

X = np.array(X, dtype=np.float32)
y_true = np.array(y_true)

print("Datos cargados:", X.shape)

# =============================
# NORMALIZACIÓN (MISMA QUE EN INFERENCIA)
# =============================
X = (X - X.mean(axis=(1,2), keepdims=True)) / (X.std(axis=(1,2), keepdims=True) + 1e-6)

# =============================
# PREDICCIÓN
# =============================
with torch.no_grad():
    inputs = torch.tensor(X).to(device)
    outputs = model(inputs)
    probs = torch.softmax(outputs, dim=1)
    y_pred = torch.argmax(probs, dim=1).cpu().numpy()

# =============================
# MATRIZ DE CONFUSIÓN
# =============================
cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(8,6))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=[id_to_label[i] for i in range(num_classes)],
    yticklabels=[id_to_label[i] for i in range(num_classes)]
)
plt.xlabel("Predicción")
plt.ylabel("Real")
plt.title("Matriz de Confusión")
plt.show()

# =============================
# MÉTRICAS
# =============================
print("\n📊 REPORTE DE CLASIFICACIÓN\n")
print(classification_report(
    y_true,
    y_pred,
    target_names=[id_to_label[i] for i in range(num_classes)]
))
