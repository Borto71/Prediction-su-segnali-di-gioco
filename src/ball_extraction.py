import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import time

today = time.strftime("%Y%m%d")
DATA_DIR = f"game_data_{today}"

game_id = "20250723"  

frames = np.load(os.path.join(DATA_DIR, f"pong_obs.npy"))
df = pd.read_csv(os.path.join(DATA_DIR, f"pong_log.csv"))

# --- Limiti del campo di gioco (modifica se necessario) ---
y_min, y_max = 34, 194   # verticale (in genere Pong classico)
x_min, x_max = 10, 150   # orizzontale (salta i bordi e paddle)

BALL_COLOR = np.array([236, 236, 236])  # Bianco "Pong"
ball_xs, ball_ys = [], []

plt.ion()
fig, ax = plt.subplots()

for i, frame in enumerate(frames):
    # Croppa il campo di gioco
    cropped = frame[y_min:y_max, x_min:x_max, :]
    mask = np.all(cropped == BALL_COLOR, axis=2)
    ys, xs = np.where(mask)

    if len(xs) > 0:
        # Riporta la posizione alle coordinate originali del frame!
        ball_x = int(np.mean(xs)) + x_min
        ball_y = int(np.mean(ys)) + y_min
    else:
        ball_x = -1
        ball_y = -1

    ball_xs.append(ball_x)
    ball_ys.append(ball_y)

    # Visualizza il frame con il punto rosso sulla pallina

    ax.clear()
    ax.imshow(frame)
    if ball_x != -1 and ball_y != -1:
        ax.scatter([ball_x], [ball_y], c='red', s=40, label='Pallina')
    ax.set_title(f"Frame {i} - Pallina ({ball_x},{ball_y})")
    ax.legend()
    plt.pause(0.04)

plt.ioff()
plt.show()

df['ball_x'] = ball_xs
df['ball_y'] = ball_ys

out_csv = os.path.join(DATA_DIR, "pong_data_features.csv")
df.to_csv(out_csv, index=False)
print(f"\nFile con posizione pallina salvato in {out_csv}!")
