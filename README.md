# 📊 Dataset Ricco per Pong Atari (RLDS - HuggingFace) — Guida al Download e Conversione

Questa guida spiega **come ottenere un grande dataset di partite Pong Atari** da HuggingFace (dati usati in AI/RL), e come **convertirlo in CSV**.

**Niente grafica necessaria, tutto da terminale!**

---

## 📦 Cosa ti serve

* **Python 3.8+**
* **pip**
* I pacchetti Python:

  * `datasets` (per scaricare da HuggingFace)
  * `pandas` (per gestire i dati e salvare in CSV)

Installa tutto con:

```bash
pip install datasets pandas
```

---

## 🚀 Come ottenere il dataset Pong

Il dataset **RLDS Atari** di HuggingFace contiene moltissime partite di vari giochi Atari, incluso Pong. Non serve installare grafica, X11 o AutoROM!

### 1. Scarica il dataset con Python

Copia questo codice in un file Python, ad esempio `extract_pong_rlds.py`:

```python
from datasets import load_dataset
import pandas as pd

# Scarica il dataset RLDS Atari (è grande, servono almeno 5-10GB liberi!)
dataset = load_dataset("rlds/atari", split="train")

# Filtra solo i dati di Pong (ci mette qualche minuto)
pong_data = dataset.filter(lambda ex: ex['game'] == "pong")
print(f"Numero di step Pong trovati: {len(pong_data)}")

# Se vuoi solo un sottoinsieme per test (es: 100.000 step):
pong_data = pong_data.select(range(0, min(100000, len(pong_data))))

# Esporta in CSV solo le colonne principali (aggiungi/rimuovi a piacere)
df = pd.DataFrame({
    "observation": pong_data['observation'],    # Attenzione: è un array grande!
    "action": pong_data['action'],
    "reward": pong_data['reward'],
    "discount": pong_data['discount'],
    "step_type": pong_data['step_type'],
    "is_first": pong_data['is_first'],
    "is_last": pong_data['is_last'],
    "is_terminal": pong_data['is_terminal'],
})
df.to_csv("pong_rlds_sample.csv", index=False)
print("Salvato pong_rlds_sample.csv con", len(df), "righe")
```

---

## ℹ️ Note utili

* **AutoROM NON serve** per questa operazione!
* **Nessuna grafica** richiesta: tutto via terminale/script.
* Il campo `"observation"` contiene i pixel del gioco come array: può rendere il CSV molto pesante.

  * Se vuoi solo azioni, reward, step, togli "observation" dalla lista.
* Puoi modificare il numero di righe esportate cambiando `min(100000, len(pong_data))`.
* Il dataset completo contiene **milioni di step**: valuta le risorse del tuo PC!

---

## 📈 Output

* **pong\_rlds\_sample.csv**: dati completi e ricchi, usabili in Excel, pandas, Jupyter, ML, RL.

---

## ⚡ Tips e problemi frequenti

* **RAM o spazio insufficiente:** riduci il numero di step esportati
* **"ImportError"**: Ricorda di installare i pacchetti e usare la stessa versione di Python
* **Analisi dei dati**: stampa sempre `pong_data.features` o `df.head()` per vedere la struttura

---

## 💡 A cosa serve

* Analisi di policy RL, prediction, benchmark, grafici reward/azioni
* Addestramento modelli, autoencoder, reti neurali
* Visualizzazione dati di gameplay reale e agenti RL

---

Dataset originale:
[https://huggingface.co/datasets/rlds/atari](https://huggingface.co/datasets/rlds/atari)

Documentazione RLDS:
[https://github.com/google-research/rlds](https://github.com/google-research/rlds)

---

*A cura di Mattia Bortolaso, Emanuele Girardello, Jiashuo Cheng e Francesco Malfer
