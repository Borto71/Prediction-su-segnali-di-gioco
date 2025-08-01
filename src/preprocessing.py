import pandas as pd
import sys
import os

if len(sys.argv) < 2:
    print("Usage: python preprocessing.py <input_csv>")
    sys.exit(1)

data_path = sys.argv[1]
output_path = data_path.replace('.csv', '_preprocessed.csv')

df = pd.read_csv(data_path)

# Funzione che calcola le colonne nuove su ogni partita
def engineer_features(gr):
    gr = gr.copy()
    gr['action_next'] = gr['action'].shift(-1)
    gr['ball_vx'] = gr['ball_x'].diff()
    gr['ball_vy'] = gr['ball_y'].diff()
    gr['right_paddle_vy'] = gr['right_paddle_y'].diff()
    gr['left_paddle_vy'] = gr['left_paddle_y'].diff()
    return gr

if 'partita' in df.columns:
    df = df.groupby('partita', group_keys=False).apply(engineer_features)
else:
    df = engineer_features(df)

# Pulisci i NaN dovuti a diff() e shift()
df = df.dropna().reset_index(drop=True)

# Se vuoi selezionare alcune colonne per il modello, puoi modificare qui
input_features = [
    'ball_x', 'ball_y',
    'right_paddle_y', 'left_paddle_y',
    'ball_vx', 'ball_vy',
    'right_paddle_vy', 'left_paddle_vy'
]
# Salva il nuovo file
df.to_csv(output_path, index=False)
print(f"Salvato il file preprocessato: {output_path}")

# Visualizza un campione per conferma
if set(input_features + ['action_next']).issubset(df.columns):
    print(df[input_features + ['action_next']].sample(10))
else:
    print(df.sample(10))
