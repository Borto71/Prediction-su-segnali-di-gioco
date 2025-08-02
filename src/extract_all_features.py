import numpy as np
import pandas as pd
import os
import re

import time

import matplotlib.pyplot as plt  # Se vuoi debug grafico, altrimenti puoi togliere

# ---- CONFIG ----
DATA_DIR = input("Inserisci il percorso della cartella dati (ad esempio: 'game_data_20240802'): ")
REPLAY_DIR = os.path.join(DATA_DIR, "replay")
log_path = os.path.join(DATA_DIR, "pong_log.csv")
features_path = os.path.join(DATA_DIR, "pong_data_features.csv")

# --- Colori e limiti campo (stessi dei tuoi script) ---
BALL_COLOR = np.array([236, 236, 236])     # Bianco "Pong"
RIGHT_PADDLE_COLOR = np.array([92, 186, 92])   # Verde racchetta destra
LEFT_PADDLE_COLOR = np.array([213, 130, 74])  # Arancione racchetta sinistra

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
    raise FileNotFoundError(f"Log file {log_path} non trovato!")

df_log = pd.read_csv(log_path)
df_log = df_log.reset_index(drop=True)

# Crea colonne vuote per features
df_log['ball_x'] = np.nan
df_log['ball_y'] = np.nan
df_log['right_paddle_y'] = np.nan
df_log['left_paddle_y'] = np.nan

# --- Ciclo sulle partite presenti nel log ---
partite = df_log['partita'].unique()
print(f"Partite trovate nel log: {partite}")

for partita_id in partite:
    print(f"Estrazione features per partita {int(partita_id)}...")
    # Trova il relativo file npy
    npy_file = os.path.join(REPLAY_DIR, f"partita_{int(partita_id)}.npy")
    if not os.path.exists(npy_file):
        print(f"  [!] Frame .npy non trovato per partita {partita_id}, skip!")
        continue

    frames = np.load(npy_file)
    # Seleziona solo il subset del log relativo a questa partita
    mask = df_log['partita'] == partita_id
    idx_log = df_log.index[mask]

    if len(idx_log) != len(frames):
        print(f"  [!] ATTENZIONE: {len(frames)} frame ma {len(idx_log)} righe di log! Le feature potrebbero non essere allineate.")
        step_min = min(len(frames), len(idx_log))
    else:
        step_min = len(frames)

    # Estrai feature per ogni frame
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

    # Aggiorna solo le righe del log della partita corrente, e solo per gli step allineati
    df_log.loc[idx_log[:step_min], 'ball_x'] = ball_xs
    df_log.loc[idx_log[:step_min], 'ball_y'] = ball_ys
    df_log.loc[idx_log[:step_min], 'right_paddle_y'] = right_paddle_ys
    df_log.loc[idx_log[:step_min], 'left_paddle_y'] = left_paddle_ys

# Ordina e salva
df_log = df_log.sort_values(['partita', 'step']).reset_index(drop=True)
df_log.to_csv(features_path, index=False)
print(f"File feature master creato in: {features_path}")
