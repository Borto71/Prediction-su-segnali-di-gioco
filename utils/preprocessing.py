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

def _balance_single_group(group, majority_label, max_ratio, safety_margin, seed):
    group = group.reset_index(drop=True)
    if 'action_next' not in group.columns:
        return group

    zero_mask = group['action_next'] == majority_label
    non_zero_mask = ~zero_mask
    zero_indices = np.flatnonzero(zero_mask.to_numpy())
    if zero_indices.size == 0 or non_zero_mask.sum() == 0:
        return group

    # preserva gli step vicini a cambi di azione o ai bordi della sequenza
    transition_mask = zero_mask & (
        non_zero_mask.shift(1, fill_value=False)
        | non_zero_mask.shift(-1, fill_value=False)
    )
    if 'action' in group.columns:
        transition_mask |= zero_mask & (
            group['action'].shift(1, fill_value=group['action'].iloc[0]) != group['action']
        )
        transition_mask |= zero_mask & (
            group['action'].shift(-1, fill_value=group['action'].iloc[-1]) != group['action']
        )

    indices_to_keep = np.flatnonzero(transition_mask.to_numpy())

    # includi un margine per preservare il contesto alle estremità
    margin = min(safety_margin, len(group))
    indices_to_keep = np.unique(
        np.concatenate([
            indices_to_keep,
            np.arange(margin),  # inizio sequenza
            np.arange(len(group) - margin, len(group))  # fine sequenza
        ])
    )
    indices_to_keep = indices_to_keep[indices_to_keep >= 0]
    indices_to_keep = indices_to_keep[indices_to_keep < len(group)]

    rng = np.random.default_rng(seed)
    target_zero = int(max_ratio * non_zero_mask.sum())
    target_zero = max(target_zero, min(safety_margin, zero_indices.size))
    target_zero = min(target_zero, zero_indices.size)

    if indices_to_keep.size >= target_zero:
        selected_zero_idx = np.sort(indices_to_keep[:target_zero])
    else:
        remaining_zero_idx = np.setdiff1d(zero_indices, indices_to_keep, assume_unique=False)
        need = target_zero - indices_to_keep.size
        if need > 0 and remaining_zero_idx.size > 0:
            sampled = rng.choice(remaining_zero_idx, size=min(need, remaining_zero_idx.size), replace=False)
            selected_zero_idx = np.sort(np.concatenate([indices_to_keep, sampled]))
        else:
            selected_zero_idx = np.sort(indices_to_keep)

    non_zero_array = non_zero_mask.to_numpy()
    keep_mask = np.zeros(len(group), dtype=bool)
    keep_mask[non_zero_array] = True
    keep_mask[selected_zero_idx] = True

    balanced = group.loc[keep_mask].reset_index(drop=True)
    return balanced

def balance_majority_class(df, majority_label=0, max_ratio=1.2, safety_margin=12, random_state=42):
    if 'action_next' not in df.columns:
        return df

    if 'partita' in df.columns:
        grouped = list(df.groupby('partita', sort=False))
        balanced_groups = []
        for offset, (_, group) in enumerate(grouped):
            balanced_groups.append(
                _balance_single_group(group, majority_label, max_ratio, safety_margin, random_state + offset)
            )
        return pd.concat(balanced_groups, ignore_index=True)

    return _balance_single_group(df, majority_label, max_ratio, safety_margin, random_state)

# Converti esplicitamente le azioni in interi per evitare problemi di confronto
if 'action' in df_processed.columns:
    df_processed['action'] = df_processed['action'].astype(int)
if 'action_next' in df_processed.columns:
    df_processed['action_next'] = df_processed['action_next'].astype(int)

before_counts = df_processed['action_next'].value_counts().sort_index().to_dict() if 'action_next' in df_processed.columns else {}
if before_counts:
    print(f"[INFO] Distribuzione classi prima del bilanciamento: {before_counts}")

df_processed = balance_majority_class(df_processed, majority_label=0, max_ratio=1.1, safety_margin=12, random_state=42)

after_counts = df_processed['action_next'].value_counts().sort_index().to_dict() if 'action_next' in df_processed.columns else {}
if after_counts:
    print(f"[INFO] Distribuzione classi dopo il bilanciamento: {after_counts}")

if {'partita', 'step'}.issubset(df_processed.columns):
    df_processed = df_processed.sort_values(['partita', 'step']).reset_index(drop=True)

# -------------------------
# 7. Salva CSV preprocessato
# -------------------------
df_processed.to_csv(csv_out, index=False)
print(f"[INFO] File preprocessato salvato in: {csv_out}")
