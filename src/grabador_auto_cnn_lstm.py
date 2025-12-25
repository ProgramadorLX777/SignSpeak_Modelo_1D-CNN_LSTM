import cv2
import mediapipe as mp
import numpy as np
import os
import time
import sys

# Mediapipe
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Carpeta donde se guardan los datos
DATA_PATH = "data_cnn_lstm_bimano"
os.makedirs(DATA_PATH, exist_ok=True)

NUM_SECUENCIAS = 10     # ← SIEMPRE 10 secuencias por clase
NUM_FRAMES = 50        # ← 50 frames por secuencia

def extraer_manos(results):
    """
    Devuelve SIEMPRE 2 manos (2 x 21 x 3)
    Si falta una → se llena con ceros.
    """
    mano_izq = np.zeros((21, 3))
    mano_der = np.zeros((21, 3))

    if results.multi_handedness and results.multi_hand_landmarks:
        for idx, hand_info in enumerate(results.multi_handedness):

            label = hand_info.classification[0].label  # "Left" o "Right"
            lm = results.multi_hand_landmarks[idx]

            puntos = np.array([[p.x, p.y, p.z] for p in lm.landmark])

            if label == "Left":
                mano_izq = puntos
            else:
                mano_der = puntos

    return mano_izq, mano_der


def grabar_secuencia(nombre_clase, indice):
    cap = cv2.VideoCapture(0)

    out_dir = os.path.join(DATA_PATH, nombre_clase)
    os.makedirs(out_dir, exist_ok=True)

    frames = []

    with mp_hands.Hands(
        model_complexity=1,
        max_num_hands=2,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.6
    ) as hands:

        print(f"\nPreparando secuencia: {indice+1}/{NUM_SECUENCIAS}...")

        # -----------------------------
        # 🔥 NUEVO: Mostramos la ventana ANTES de la cuenta regresiva
        # -----------------------------
        for _ in range(10):  
            ret, frame = cap.read()
            if not ret:
                continue
            frame = cv2.flip(frame, 1)
            cv2.putText(frame, "Preparando grabacion...", (30, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 2)
            cv2.imshow("Grabando secuencias...", frame)
            cv2.waitKey(30)

        # -----------------------------
        # 🔥 NUEVO: Cuenta regresiva de 3 segundos visible
        # -----------------------------
        for t in [3, 2, 1]:
            ret, frame = cap.read()
            if ret:
                frame = cv2.flip(frame, 1)
                cv2.putText(frame, str(t), (250, 250),
                            cv2.FONT_HERSHEY_SIMPLEX, 4, (0,0,255), 6)
                cv2.imshow("Grabando secuencias...", frame)
                cv2.waitKey(1000)

        print("GRABANDO SECUENCIAS...\n")

        # -----------------------------
        # GRABACIÓN DE LOS 50 FRAMES
        # -----------------------------
        while len(frames) < NUM_FRAMES:
            ret, frame = cap.read()
            if not ret:
                continue

            # Vista con espejo SOLO para mostrar, pero procesamiento sin invertir
            frame_vis = cv2.flip(frame, 1)

            img_rgb = cv2.cvtColor(frame_vis, cv2.COLOR_BGR2RGB)
            results = hands.process(img_rgb)

            mano_izq, mano_der = extraer_manos(results)

            # Vector: 42 puntos × 3 coordenadas → (126,)
            frame_vec = np.concatenate([mano_izq.reshape(21,3), mano_der.reshape(21,3)], axis=0).flatten()
            frames.append(frame_vec)
            
            # 🔥 DIBUJO DE LANDMARKS (SOLO VISUAL)
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        frame_vis,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,
                        mp_drawing.DrawingSpec(color=(0,255,0), thickness=2, circle_radius=3),
                        mp_drawing.DrawingSpec(color=(255,0,0), thickness=2)
                    )

            cv2.putText(
                frame_vis,
                f"Sec {indice+1}/{NUM_SECUENCIAS} - Frame {len(frames)}/{NUM_FRAMES}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2
            )
            cv2.imshow("Grabando Secuencias...", frame_vis)

            key = cv2.waitKey(1) & 0xFF 
            
            if key == 27:  # ESC
                print("⛔ Grabación cancelada por el usuario.")
                cap.release()
                cv2.destroyAllWindows()
                sys.exit(0)

            '''if cv2.waitKey(1) & 0xFF == ord('q'):
                break'''

    cap.release()
    cv2.destroyAllWindows()

    frames = np.array(frames)  # (50, 126)
    np.save(os.path.join(out_dir, f"seq_{indice}.npy"), frames)
    print("✔ Secuencia guardada correctamente.")


# =======================================
#               MAIN
# =======================================
if __name__ == "__main__":
    clase = input("Nombre de la clase: ")

    out_dir = os.path.join(DATA_PATH, clase)
    os.makedirs(out_dir, exist_ok=True)

    # Contar cuántas secuencias existen
    existentes = [f for f in os.listdir(out_dir) if f.endswith(".npy")]
    inicio = len(existentes)

    print(f"\nLa clase '{clase}' ya tiene {inicio} secuencias guardadas.")
    print(f"Se grabarán {NUM_SECUENCIAS} secuencias nuevas.\n")

    for i in range(NUM_SECUENCIAS):
        grabar_secuencia(clase, inicio + i)

    print("\n✔ Todas las secuencias nuevas fueron agregadas.")

