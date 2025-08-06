import os
import pandas as pd
import numpy as np
import sys

# Controlla che venga passato almeno un argomento da linea di comando
if len(sys.argv) < 2:
    raise ValueError("Uso: python preprocessing.py <nome_cartella>")
    
# Prende il nome della cartella dai parametri in input
cartella = sys.argv[1].strip()

# Definisce i percorsi del file CSV di input e output
csv_in = os.path.join(cartella, "pong_data_features.csv")
csv_out = os.path.join(cartella, "pong_data_features_preprocessed.csv")

# Verifica che il file di input esista, altrimenti solleva errore
if not os.path.exists(csv_in):
    raise FileNotFoundError(f"File non trovato: {csv_in}")

print(f"Carico dati da: {csv_in}")

# Legge i dati CSV in un DataFrame pandas
df = pd.read_csv(csv_in)

# 1. FILTRO: Rimuove tutte le righe che contengono valori placeholder -1.0 nelle colonne specificate
placeholder_cols = ['ball_x', 'ball_y', 'right_paddle_y', 'left_paddle_y']
for col in placeholder_cols:
    df = df[df[col] != -1.0]
df = df.reset_index(drop=True) # resetta gli indici dopo il filtro

print(f"Righe dopo filtro placeholder: {len(df)}")

# 2. Feature engineering di base
def engineer_features(group):
    group = group.copy()  # copia per evitare warning su assegnazioni a slice
    group['action_next'] = group['action'].shift(-1)  # azione nel frame successivo
    # calcola velocità del pallone e delle racchette come differenze tra frame consecutivi
    group['ball_vx'] = group['ball_x'].diff().fillna(0)
    group['ball_vy'] = group['ball_y'].diff().fillna(0)
    group['right_paddle_vy'] = group['right_paddle_y'].diff().fillna(0)
    group['left_paddle_vy'] = group['left_paddle_y'].diff().fillna(0)
    # distanze assolute verticali tra pallone e racchette
    group['dist_right'] = (group['ball_y'] - group['right_paddle_y']).abs()
    group['dist_left']  = (group['ball_y'] - group['left_paddle_y']).abs()
    # offset verticale (posizione relativa) tra pallone e racchetta destra
    group['offset_right'] = group['ball_y'] - group['right_paddle_y']
    # angolo di movimento del pallone in gradi (arctan2(vy, vx))
    group['ball_angle'] = np.degrees(np.arctan2(group['ball_vy'], group['ball_vx']))
    # direzione orizzontale del pallone: -1 (verso sinistra), 0 o 1 (verso destra)
    group['ball_dir']   = np.sign(group['ball_vx'])
    # Rimuove l'ultima riga perché 'action_next' è NaN (shift -1)
    return group.iloc[:-1].reset_index(drop=True)

def engineer_features(group):
    group = group.copy()  # copia per evitare warning su assegnazioni a slice
    group['action_next'] = group['action'].shift(-1)  # azione nel frame successivo
    # calcola velocità del pallone e delle racchette come differenze tra frame consecutivi
    group['ball_vx'] = group['ball_x'].diff().fillna(0)
    group['ball_vy'] = group['ball_y'].diff().fillna(0)
    group['right_paddle_vy'] = group['right_paddle_y'].diff().fillna(0)
    group['left_paddle_vy'] = group['left_paddle_y'].diff().fillna(0)
    # distanze assolute verticali tra pallone e racchette
    group['dist_right'] = (group['ball_y'] - group['right_paddle_y']).abs()
    group['dist_left']  = (group['ball_y'] - group['left_paddle_y']).abs()
    # offset verticale (posizione relativa) tra pallone e racchetta destra
    group['offset_right'] = group['ball_y'] - group['right_paddle_y']
    # angolo di movimento del pallone in gradi (arctan2(vy, vx))
    group['ball_angle'] = np.degrees(np.arctan2(group['ball_vy'], group['ball_vx']))
    # direzione orizzontale del pallone: -1 (verso sinistra), 0 o 1 (verso destra)
    group['ball_dir']   = np.sign(group['ball_vx'])
    # Rimuove l'ultima riga perché 'action_next' è NaN (shift -1)
    return group.iloc[:-1].reset_index(drop=True)