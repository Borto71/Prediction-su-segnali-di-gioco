import pandas as pd

# Carica i dati
df = pd.read_csv('game_data_20250724/pong_data_features.csv')

# Colonna 'action_next'
df['action_next'] = df['action'].shift(-1)

# Feature engineering: velocità pallina
df['ball_vx'] = df['ball_x'].diff()
df['ball_vy'] = df['ball_y'].diff()

# Feature engineering: velocità paddle
df['right_paddle_vy'] = df['right_paddle_y'].diff()
df['left_paddle_vy'] = df['left_paddle_y'].diff()

# Pulisci i NaN dovuti a diff() e shift()
df = df.dropna().reset_index(drop=True)

# Scegli le feature di input
input_features = [
    'ball_x', 'ball_y',
    'right_paddle_y', 'left_paddle_y',
    'ball_vx', 'ball_vy',
    'right_paddle_vy', 'left_paddle_vy'
]

# Salva il nuovo file
df.to_csv('game_data_20250724/pong_data_features_preprocessed.csv', index=False)

# Visualizza un campione per conferma
print(df[input_features + ['action_next']].sample(10))
