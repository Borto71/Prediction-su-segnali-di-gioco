import numpy as np
import pandas as pd
import os
import sys
import time



if len(sys.argv) < 2:
    print("Errore: devi specificare il nome della cartella (es. game_data_20251006)")
    sys.exit(1)

DATA_DIR = sys.argv[1]
REPLAY_DIR = os.path.join(DATA_DIR, "replay")
log_path = os.path.join(DATA_DIR, "pong_log.csv")
features_path = os.path.join(DATA_DIR, "pong_data_features.csv")

if not os.path.exists(DATA_DIR):
    print(f"Cartella {DATA_DIR} non trovata.")
    sys.exit(1)

# --- Colori e limiti campo ---
BALL_COLOR = np.array([236, 236, 236])       # Bianco "Pong"
RIGHT_PADDLE_COLOR = np.array([92, 186, 92]) # Verde racchetta destra
LEFT_PADDLE_COLOR = np.array([213, 130, 74]) # Arancione racchetta sinistra

y_min, y_max = 34, 194
x_min_ball, x_max_ball = 10, 150
x_min_right, x_max_right = 127, 158
x_min_left, x_max_left = 10, 25

# --- Funzioni estrazione ---
def extract_ball(frame):
    cropped = frame[y_min:y_max, x_min_ball:x_max_ball, :]
    mask = np.all(cropped == BALL_COLOR, axis=2)
    ys, xs = np.where(mask)
    if len(xs) > 0:
        ball_x = int(np.mean(xs)) + x_min_ball
        ball_y = int(np.mean(ys)) + y_min
    else:
        ball_x, ball_y = -1, -1
    return ball_x, ball_y

def extract_right_paddle(frame):
    cropped = frame[y_min:y_max, x_min_right:x_max_right, :]
    mask = np.all(cropped == RIGHT_PADDLE_COLOR, axis=2)
    ys, xs = np.where(mask)
    if len(ys) > 0:
        paddle_y = int(np.mean(ys)) + y_min
    else:
        paddle_y = -1
    return paddle_y

def extract_left_paddle(frame):
    cropped = frame[y_min:y_max, x_min_left:x_max_left, :]
    mask = np.all(cropped == LEFT_PADDLE_COLOR, axis=2)
    ys, xs = np.where(mask)
    if len(ys) > 0:
        paddle_y = int(np.mean(ys)) + y_min
    else:
        paddle_y = -1
    return paddle_y

# --- MAIN ---
if not os.path.exists(log_path):
    print(f"error: file not found: {log_path}")
    sys.exit(1)

print(f"Processing file: {log_path}")

df_log = pd.read_csv(log_path).reset_index(drop=True)
df_log['ball_x'] = np.nan
df_log['ball_y'] = np.nan
df_log['right_paddle_y'] = np.nan
df_log['left_paddle_y'] = np.nan

partite = df_log['partita'].unique()


for partita_id in partite:
    print(f"Extracting game features: {int(partita_id)}...")
    npy_file = os.path.join(REPLAY_DIR, f"game{int(partita_id)}.npy")

    if not os.path.exists(npy_file):
        print(f"  File {npy_file} missing, skipped.")
        continue

    frames = np.load(npy_file)
    mask = df_log['partita'] == partita_id
    idx_log = df_log.index[mask]

    step_min = min(len(frames), len(idx_log))
    if len(frames) != len(idx_log):
        print(f"{len(frames)} frame ma {len(idx_log)} righe di log, taglio a {step_min} step.")

    ball_xs, ball_ys, right_paddle_ys, left_paddle_ys = [], [], [], []
    for i in range(step_min):
        frame = frames[i]
        bx, by = extract_ball(frame)
        rx = extract_right_paddle(frame)
        lx = extract_left_paddle(frame)
        ball_xs.append(bx)
        ball_ys.append(by)
        right_paddle_ys.append(rx)
        left_paddle_ys.append(lx)

    df_log.loc[idx_log[:step_min], 'ball_x'] = ball_xs
    df_log.loc[idx_log[:step_min], 'ball_y'] = ball_ys
    df_log.loc[idx_log[:step_min], 'right_paddle_y'] = right_paddle_ys
    df_log.loc[idx_log[:step_min], 'left_paddle_y'] = left_paddle_ys

df_log = df_log.sort_values(['partita', 'step']).reset_index(drop=True)
df_log.to_csv(features_path, index=False)
print(f"Extracted data csv created in: {features_path}")
