import numpy as np
import pandas as pd
import os
import time
import matplotlib.pyplot as plt

today = time.strftime("%Y%m%d")
DATA_DIR = f"game_data_{today}"

frames = np.load(os.path.join(DATA_DIR, "pong_obs.npy"))

# Cerca PRIMA il CSV features, ALTRIMENTI usa quello base
features_path = os.path.join(DATA_DIR, "pong_data_features.csv")
if os.path.exists(features_path):
    df = pd.read_csv(features_path)
else:
    df = pd.read_csv(os.path.join(DATA_DIR, "pong_log.csv"))

# Limiti campo per la RACCHETTA SINISTRA (modifica se serve)
y_min, y_max = 34, 194
x_min, x_max = 10, 25   # Bordo sinistro (modifica se serve per il tuo Pong)

PADDLE_COLOR = np.array([213, 130, 74])  # Arancione racchetta sinistra

paddle_ys = []

plt.ion()
fig, ax = plt.subplots()

for i, frame in enumerate(frames):
    cropped = frame[y_min:y_max, x_min:x_max, :]
    mask = np.all(cropped == PADDLE_COLOR, axis=2)
    ys, xs = np.where(mask)

    if len(ys) > 0:
        paddle_y_min = np.min(ys) + y_min
        paddle_y_max = np.max(ys) + y_min
        paddle_y = int(np.mean(ys)) + y_min
    else:
        paddle_y_min = paddle_y_max = paddle_y = -1

    paddle_ys.append(paddle_y)

    # Visualizzazione: barra verticale rossa sulla racchetta sinistra
    ax.clear()
    ax.imshow(frame)
    if paddle_y != -1:
        paddle_x = int((x_min + x_max) // 2)
        ax.plot([paddle_x, paddle_x], [paddle_y_min, paddle_y_max], color='red', linewidth=6, label='Racchetta sinistra')
        ax.legend()
    ax.set_title(f"Frame {i} - Paddle sinistra y={paddle_y}")
    plt.pause(0.04)

plt.ioff()
plt.show()

# AGGIUNGI SOLO la colonna della paddle sinistra!
df['left_paddle_y'] = paddle_ys
df.to_csv(features_path, index=False)
print(f"\nFile con posizione racchetta sinistra AGGIORNATO in {features_path}!")
