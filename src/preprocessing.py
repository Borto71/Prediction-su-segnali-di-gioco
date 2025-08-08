import os
import pandas as pd
import numpy as np
import sys

# -------------------------
# 1. Controllo argomenti
# -------------------------
if len(sys.argv) < 2:
    raise ValueError("Uso: python preprocessing.py <nome_cartella>")

cartella = sys.argv[1].strip()

# Percorsi input/output
csv_in = os.path.join(cartella, "pong_data_features.csv")
csv_out = os.path.join(cartella, "pong_data_features_preprocessed.csv")

# -------------------------
# 2. Controllo file di input
# -------------------------
if not os.path.exists(csv_in):
    raise FileNotFoundError(f"File non trovato: {csv_in}")

print(f"[INFO] Carico dati da: {csv_in}")

# -------------------------
# 3. Caricamento CSV
# -------------------------
df = pd.read_csv(csv_in)

# -------------------------
# 4. Rimuovi placeholder (-1.0)
# -------------------------
placeholder_cols = ['ball_x', 'ball_y', 'right_paddle_y', 'left_paddle_y']
for col in placeholder_cols:
    df = df[df[col] != -1.0]

df = df.reset_index(drop=True)
print(f"[INFO] Righe dopo filtro placeholder: {len(df)}")

# -------------------------
# 5. Funzione feature engineering
# -------------------------
def engineer_features(group):
    group = group.copy()
    group['action_next'] = group['action'].shift(-1)

    # Velocità
    group['ball_vx'] = group['ball_x'].diff().fillna(0)
    group['ball_vy'] = group['ball_y'].diff().fillna(0)
    group['right_paddle_vy'] = group['right_paddle_y'].diff().fillna(0)
    group['left_paddle_vy'] = group['left_paddle_y'].diff().fillna(0)

    # Distanze e offset
    group['dist_right'] = (group['ball_y'] - group['right_paddle_y']).abs()
    group['dist_left']  = (group['ball_y'] - group['left_paddle_y']).abs()
    group['offset_right'] = group['ball_y'] - group['right_paddle_y']

    # Angolo e direzione
    group['ball_angle'] = np.degrees(np.arctan2(group['ball_vy'], group['ball_vx']))
    group['ball_dir']   = np.sign(group['ball_vx'])

    # Rimuovi ultima riga (action_next = NaN)
    return group.iloc[:-1].reset_index(drop=True)

# -------------------------
# 6. Applica feature engineering
# -------------------------
if 'partita' in df.columns:
    df_processed = df.groupby('partita', group_keys=False).apply(engineer_features)
else:
    df_processed = engineer_features(df)

# -------------------------
# 7. Salva CSV preprocessato
# -------------------------
df_processed.to_csv(csv_out, index=False)
print(f"[INFO] File preprocessato salvato in: {csv_out}")
