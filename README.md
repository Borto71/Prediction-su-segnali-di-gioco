# Pong Arena: Multi-Model RL & Tournament Pipeline

This repository hosts an advanced research pipeline for **Atari Pong**, evolving from basic Transformer-based imitation learning to a **competitive multi-model ecosystem**. The project enables the training of diverse neural architectures using both offline data and **Reinforcement Learning (RL)**, culminating in an automated **Tournament Mode** where agents compete head-to-head.

---

## 🚀 Project Evolution

Originally designed to predict actions via a single Transformer architecture, this project now supports:
* **Model Zoo:** Comparison between Transformers, LSTMs, and Deep MLPs.
* **Hybrid Training:** Offline pre-training (Behavioral Cloning) followed by **Reinforcement Learning** (PPO/DQN) fine-tuning.
* **Tournament Arena:** A dedicated simulation environment for model-vs-model matchmaking.
* **Advanced Analytics:** Comparative benchmarking of win rates, reaction times, and strategy stability.

---

## 🛠 Pipeline Stages

### 1. Data Acquisition & Extraction
The foundation relies on high-quality gameplay data to bootstrap model learning.
* **Game Recording (`play_pong.py`):** Captures frames and actions (`UP`, `DOWN`, `NOOP`) from the OpenAI Gym environment.
* **Feature Extraction (`extract_all_features.py`):** Isolates coordinates for the **right paddle**, **left paddle**, and the **ball** using color segmentation.
* **Preprocessing & Normalization:** Standardizes numerical features using Z-score and Min-Max scaling via `preprocessing.py` and `normalize_data.py`.

### 2. The Model Zoo
We evaluate different "brains" to determine which architecture handles the temporal dynamics of Pong most effectively:
* **Pong Transformer:** Uses Multi-Head Self-Attention to learn temporal dependencies across sequences.
* **Recurrent Agents (LSTM/GRU):** Designed for sequential state tracking and long-term memory.
* **Baseline MLP:** A high-speed feed-forward network used as a performance benchmark.

### 3. Training Strategy
The pipeline supports a dual-stage training approach:
1.  **Phase I: Imitation Learning:** Models learn by mimicking human or scripted play using the preprocessed feature datasets.
2.  **Phase II: Reinforcement Learning:** Models are placed back into the Gym environment to optimize their policy through trial and error, improving their ability to handle unseen game states.

---

## 🏆 Tournament Mode
The centerpiece of the project is the **Arena**, where trained models face off:
* **Head-to-Head Matches:** Load two different model checkpoints (e.g., `Transformer_v2` vs `LSTM_v1`).
* **Automated Scoring:** Tracks wins, losses, and average volley duration.
* **Leaderboard:** Ranks architectures based on their competitive performance across multiple seeds.

---

## 📂 Repository Structure

```text
.
├── GUI.py                  # Interface for managing games and training
├── play_pong.py            # Game capture and logging
├── extract_all_features.py  # Coordinate extraction logic
├── preprocessing.py        # Data cleaning and feature engineering
├── normalize_data.py       # Feature normalization
├── models/                 # Model definitions
│   ├── transformer.py      # Transformer Encoder architecture
│   ├── recurrent.py        # LSTM/GRU implementations
│   └── mlp.py              # Baseline MLP
├── training/               
│   ├── main.py             # Offline training pipeline
│   └── rl_trainer.py       # Reinforcement Learning (PPO) logic
├── arena.py                # Tournament and Model-vs-Model logic
├── checkpoints/            # Saved model weights
└── logs/                   # Training metrics and tournament results