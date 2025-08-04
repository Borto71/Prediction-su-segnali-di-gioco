# normalize_data.py
import os
import pandas as pd
import numpy as np

# 1. CHIEDI SOLO IL NOME DELLA CARTELLA
cartella = input("Inserisci il nome della cartella dati (es: game_data_20250801): ").strip()
# 2. COSTRUISCI IL PERCORSO DEL FILE CSV PREPROCESSATO
csv_in = os.path.join(cartella, "pong_data_features_preprocessed.csv")
if not os.path.exists(csv_in):
    raise FileNotFoundError(f"File non trovato: {csv_in}")

print(f"Carico dati da: {csv_in}")
df = pd.read_csv(csv_in)

# 3. NORMALIZZA LE FEATURE NUMERICHE PRINCIPALI
to_normalize = [
    'ball_x', 'ball_y', 'right_paddle_y', 'left_paddle_y',
    'ball_vx', 'ball_vy', 'right_paddle_vy', 'left_paddle_vy',
    'dist_right', 'dist_left', 'offset_right', 'ball_angle', 'ball_dir',
    'relative_vy_right', 'frames_to_right', 'sin_angle', 'cos_angle'
]
print("Normalizzo queste colonne:", to_normalize)

stats = {}
for col in to_normalize:
    if col in df.columns:
        mean = df[col].mean()
        std = df[col].std()
        minv = df[col].min()
        maxv = df[col].max()
        df[f"{col}_std"] = (df[col] - mean) / (std + 1e-8)
        df[f"{col}_minmax"] = (df[col] - minv) / (maxv - minv + 1e-8)
        stats[col] = dict(mean=mean, std=std, min=minv, max=maxv)

for col in to_normalize:
    if f"{col}_std" in df.columns and f"{col}_minmax" in df.columns:
        print(f"{col}_std: {df[f'{col}_std'].min():.2f} / {df[f'{col}_std'].max():.2f}")
        print(f"{col}_minmax: {df[f'{col}_minmax'].min():.2f} / {df[f'{col}_minmax'].max():.2f}")

# 4. SALVA SU FILE CON NOME STANDARD
csv_out = os.path.join(cartella, "pong_data_features_preprocessed_normalized.csv")
df.to_csv(csv_out, index=False)
print(f"\nDati normalizzati salvati in: {csv_out}")

# (Opzionale) Salva parametri normalizzazione
import json
stats_path = os.path.join(cartella, "normstats.json")
with open(stats_path, "w") as f:
    json.dump(stats, f, indent=2)
print(f"Salvati parametri normalizzazione in: {stats_path}")
