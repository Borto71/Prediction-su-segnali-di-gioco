import pandas as pd
import numpy as np
import sys

if len(sys.argv) > 1:
    csv_path = sys.argv[1]
else:
    csv_path = input("Inserisci il percorso del file CSV da preprocessare: ").strip()

print(f"Carico dati da: {csv_path}")
df = pd.read_csv(csv_path)

# Filtra righe con valori placeholder (-1.0)
placeholder_cols = ['ball_x', 'ball_y', 'right_paddle_y', 'left_paddle_y']
for col in placeholder_cols:
    df = df[df[col] != -1.0]
df = df.reset_index(drop=True)

print(f"Righe dopo filtro placeholder: {len(df)}")

def engineer_features(group):
    group = group.copy()
    # Aggiungi l'azione del frame successivo
    group['action_next'] = group['action'].shift(-1)
    # Calcola velocità come differenza tra frame
    group['ball_vx'] = group['ball_x'].diff().fillna(0)
    group['ball_vy'] = group['ball_y'].diff().fillna(0)
    group['right_paddle_vy'] = group['right_paddle_y'].diff().fillna(0)
    group['left_paddle_vy'] = group['left_paddle_y'].diff().fillna(0)
    # Feature nuove:
    group['dist_right'] = (group['ball_y'] - group['right_paddle_y']).abs()
    group['dist_left'] = (group['ball_y'] - group['left_paddle_y']).abs()
    group['ball_angle'] = np.degrees(np.arctan2(group['ball_vy'], group['ball_vx']))
    group['ball_dir'] = np.sign(group['ball_vx'])
    # Rimuovi ultima riga che ora ha NaN in action_next
    return group.iloc[:-1].reset_index(drop=True)

print("Feature engineering in corso...")
df = df.groupby('partita', group_keys=False).apply(engineer_features).reset_index(drop=True)

# Mostra sample di dati
print(df.head())

# Controllo range feature
print("\n[CHECK] Range delle feature calcolate:")
features_to_check = [
    'ball_x', 'ball_y', 'right_paddle_y', 'left_paddle_y',
    'ball_vx', 'ball_vy', 'right_paddle_vy', 'left_paddle_vy',
    'dist_right', 'dist_left', 'ball_angle', 'ball_dir'
]
for col in features_to_check:
    if col in df.columns:
        print(f"{col}: min={df[col].min()}, max={df[col].max()}, unique={df[col].nunique()}")

# Salva CSV preprocessato
out_path = csv_path.replace(".csv", "_preprocessed.csv")
df.to_csv(out_path, index=False)
print(f"\nDati preprocessati salvati in: {out_path}")
