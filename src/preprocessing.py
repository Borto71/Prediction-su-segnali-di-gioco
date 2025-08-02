

import os
import pandas as pd
import numpy as np

cartella = input("Inserisci il nome della cartella dati (es: game_data_20250801): ").strip()
csv_in = os.path.join(cartella, "pong_data_features.csv")
csv_out = os.path.join(cartella, "pong_data_features_preprocessed.csv")

if not os.path.exists(csv_in):
    raise FileNotFoundError(f"File non trovato: {csv_in}")

print(f"Carico dati da: {csv_in}")
df = pd.read_csv(csv_in)

# 1. Filtra righe con valori placeholder (-1.0)
placeholder_cols = ['ball_x', 'ball_y', 'right_paddle_y', 'left_paddle_y']
for col in placeholder_cols:
    df = df[df[col] != -1.0]
df = df.reset_index(drop=True)

print(f"Righe dopo filtro placeholder: {len(df)}")

# 2. Feature engineering di base
def engineer_features(group):
    group = group.copy()
    group['action_next'] = group['action'].shift(-1)
    group['ball_vx'] = group['ball_x'].diff().fillna(0)
    group['ball_vy'] = group['ball_y'].diff().fillna(0)
    group['right_paddle_vy'] = group['right_paddle_y'].diff().fillna(0)
    group['left_paddle_vy'] = group['left_paddle_y'].diff().fillna(0)
    group['dist_right'] = (group['ball_y'] - group['right_paddle_y']).abs()
    group['dist_left']  = (group['ball_y'] - group['left_paddle_y']).abs()
    group['offset_right'] = group['ball_y'] - group['right_paddle_y']
    group['ball_angle'] = np.degrees(np.arctan2(group['ball_vy'], group['ball_vx']))
    group['ball_dir']   = np.sign(group['ball_vx'])
    return group.iloc[:-1].reset_index(drop=True)

def advanced_features(group):
    group = group.copy()
    TOL_X = 2
    right_paddle_x = 158
    group['impact_right'] = (
        (group['ball_x'] >= right_paddle_x - TOL_X) &
        (group['ball_x'] <= right_paddle_x + TOL_X)
    ).astype(int)
    group['frames_to_right'] = -1
    impact_indices = group.index[group['impact_right'] == 1].tolist()
    for i in range(len(group)):
        future_impacts = [idx for idx in impact_indices if idx >= i]
        if future_impacts:
            group.at[i, 'frames_to_right'] = future_impacts[0] - i
    group['relative_vy_right'] = group['ball_vy'] - group['right_paddle_vy']
    group['align_dir_right'] = 0
    moving = (group['ball_vy'] != 0) & (group['right_paddle_vy'] != 0)
    same_sign = np.sign(group['ball_vy']) == np.sign(group['right_paddle_vy'])
    group.loc[moving & same_sign, 'align_dir_right'] = 1
    group.loc[moving & ~same_sign, 'align_dir_right'] = -1
    group['aligns_right']   = (group['align_dir_right'] == 1).astype(int)
    group['opposes_right']  = (group['align_dir_right'] == -1).astype(int)
    group['sin_angle'] = np.sin(np.radians(group['ball_angle']))
    group['cos_angle'] = np.cos(np.radians(group['ball_angle']))
    return group

print("Feature engineering di base...")
df = df.groupby('partita', group_keys=False).apply(engineer_features).reset_index(drop=True)

print("Aggiunta feature avanzate...")
df = df.groupby('partita', group_keys=False).apply(advanced_features).reset_index(drop=True)

features_to_check = [
    'ball_x', 'ball_y', 'right_paddle_y', 'left_paddle_y',
    'ball_vx', 'ball_vy', 'right_paddle_vy', 'left_paddle_vy',
    'dist_right', 'dist_left', 'offset_right', 'ball_angle', 'ball_dir',
    'relative_vy_right', 'frames_to_right', 'sin_angle', 'cos_angle',
    'aligns_right', 'opposes_right'
]
print("\n[CHECK] Range delle feature calcolate:")
for col in features_to_check:
    if col in df.columns:
        print(f"{col}: min={df[col].min()}, max={df[col].max()}, unique={df[col].nunique()}")

# 3. Salva CSV preprocessato in automatico
df.to_csv(csv_out, index=False)
print(f"\nDati preprocessati salvati in: {csv_out}")
