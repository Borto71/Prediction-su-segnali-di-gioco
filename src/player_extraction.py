import numpy as np
import pandas as pd
import os
import time
import matplotlib.pyplot as plt

today = time.strftime("%Y%m%d")
DATA_DIR = f"game_data_{today}"

frames = np.load(os.path.join(DATA_DIR, "pong_obs.npy"))

# CERCA PRIMA UN CSV features, ALTRIMENTI USA QUELLO BASE
features_path = os.path.join(DATA_DIR, "pong_data_features.csv")
if os.path.exists(features_path):
    df = pd.read_csv(features_path)
else:
    df = pd.read_csv(os.path.join(DATA_DIR, "pong_log.csv"))

# Limiti campo 
y_min, y_max = 34, 194
x_min, x_max = 127, 158   # Solo bordo destro

PADDLE_COLOR = np.array([92, 186, 92])  # Verde racchetta destra

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

    # Visualizzazione: barra verticale rossa sulla racchetta
    ax.clear()
    ax.imshow(frame)
    if paddle_y != -1:
        paddle_x = int((x_min + x_max) // 2)
        ax.plot([paddle_x, paddle_x], [paddle_y_min, paddle_y_max], color='red', linewidth=6, label='Racchetta destra')
        ax.legend()
    ax.set_title(f"Frame {i} - Paddle destra y={paddle_y}")
    plt.pause(0.04)

plt.ioff()
plt.show()

# AGGIUNGI SOLO la colonna della paddle destra!
df['right_paddle_y'] = paddle_ys
df.to_csv(features_path, index=False)
print(f"\nFile con posizione racchetta destra AGGIORNATO in {features_path}!")
