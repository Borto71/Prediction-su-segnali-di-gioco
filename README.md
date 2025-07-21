# 🏓 Pong Atari - Quickstart Guide

Questa guida ti permette di **installare**, **eseguire**, e **salvare dati** da Pong Atari con Gymnasium e ALE, generando sia un file CSV con i dati che una GIF della partita. Funziona sia su **Ubuntu** che su **WSL (Windows Subsystem for Linux)**.

---

## 📦 Requisiti

* Python 3.8+ (consigliato Python 3.10 o superiore)
* pip
* (Per WSL) Server X11 su Windows, es: [VcXsrv](https://sourceforge.net/projects/vcxsrv/)
* AutoROM per scaricare le ROM Atari

---

## 🚀 Setup passo-passo

### 1. Crea e attiva un ambiente virtuale

```bash
python3 -m venv venv
source venv/bin/activate
```

### 2. Installa le dipendenze principali

```bash
pip install gymnasium[atari,accept-rom-license]
pip install ale-py matplotlib pandas imageio
```

### 3. Scarica le ROM Atari

```bash
AutoROM --accept-license
```

### 4. (Solo per WSL) Avvia il server X11 su Windows

Esempio: avvia **VcXsrv** su Windows, poi in WSL esegui:

```bash
export DISPLAY=:0
```

---

## 📝 Esempio di script - Logging CSV

Salva il seguente script come `test_pong_log.py`:

```python
import gymnasium as gym
import ale_py
import pandas as pd

gym.register_envs(ale_py)
env = gym.make('ALE/Pong-v5', render_mode="rgb_array")
obs, info = env.reset()
done = False
steps = 0
data = []

while not done and steps < 500:
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    score = info.get('score', None)
    data.append({'step': steps, 'action': action, 'reward': reward, 'score': score})
    done = terminated or truncated
    steps += 1

env.close()

df = pd.DataFrame(data)
df.to_csv('pong_log.csv', index=False)
print("Dati salvati in pong_log.csv!")
```

Lanciando questo script otterrai un file chiamato `pong_log.csv`.

---

## 🖼️ Esempio di script - Salva una GIF

Salva il seguente script come `test_pong_GIF.py`:

```python
import gymnasium as gym
import ale_py
import imageio

gym.register_envs(ale_py)
env = gym.make('ALE/Pong-v5', render_mode="rgb_array")

obs, info = env.reset()
done = False
steps = 0
frames = []

while not done and steps < 500:
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    frames.append(obs)
    done = terminated or truncated
    steps += 1

env.close()

imageio.mimsave('pong_run.gif', frames, duration=0.04)
print("GIF salvata come pong_run.gif!")
```

Troverai la GIF **pong\_run.gif** nella stessa cartella.

---

## 📈 Output attesi

* **pong\_log.csv** – contiene step, azioni, reward, score (pronto per analisi o training)
* **pong\_run.gif** – GIF animata della partita

---

## ❓ FAQ & Problemi comuni

* **Schermo nero nella finestra:**
  Usa `render_mode="rgb_array"` e visualizza/salva i frame (come negli script sopra).
  Su WSL, il rendering X11 può non funzionare perfettamente.

* **AutoROM non trova le ROM:**
  Assicurati che siano nella cartella `~/.ale/roms/`. Puoi copiare manualmente con:

  ```bash
  mkdir -p ~/.ale/roms/
  cp /percorso/ROM/*.bin ~/.ale/roms/
  ```

* **ImportError: No module named ...**
  Verifica che il venv sia attivo e le dipendenze installate.

---

A cura di Mattia Bortolaso, Emanuele Girardello, Jiashuo Cheng e Francesco Malfer
