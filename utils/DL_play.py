#!/usr/bin/env python3
"""
Real-time inference loop for the Pong Transformer agent.

It mirrors the behaviour of `utils/play_pong.py`, but replaces the human
controller with the trained Transformer model for automated play.

Usage example:

    python utils/DL_play.py --checkpoint checkpoints/pong_transformer_best.pt
"""

from __future__ import annotations

import argparse
import sys
import time
from collections import deque
from pathlib import Path
from typing import Deque, Dict, List, Optional, Tuple

import ale_py  # noqa: F401 (needed for Gym registration)
import gymnasium as gym
import imageio
import numpy as np
import pandas as pd
import torch
import tkinter as tk
from PIL import Image, ImageTk

# --- Repository imports -----------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.transformer import PongTransformer  # noqa: E402


# --- Constants --------------------------------------------------------------
ACTION_MEANING = {0: "NOOP", 2: "UP", 3: "DOWN"}
LABEL_TO_ACTION = [0, 2, 3]
ACTION_TO_LABEL = {action: idx for idx, action in enumerate(LABEL_TO_ACTION)}

BALL_COLOR = np.array([236, 236, 236])
RIGHT_PADDLE_COLOR = np.array([92, 186, 92])
LEFT_PADDLE_COLOR = np.array([213, 130, 74])

Y_MIN, Y_MAX = 34, 194
X_MIN_BALL, X_MAX_BALL = 10, 150
X_MIN_RIGHT, X_MAX_RIGHT = 127, 158
X_MIN_LEFT, X_MAX_LEFT = 10, 25


# --- Utility functions ------------------------------------------------------
def detect_positions(frame: np.ndarray) -> Dict[str, float]:
    """Extract ball and paddle coordinates from the RGB frame."""
    cropped_ball = frame[Y_MIN:Y_MAX, X_MIN_BALL:X_MAX_BALL, :]
    ball_mask = np.all(cropped_ball == BALL_COLOR, axis=2)
    if ball_mask.any():
        ys, xs = np.where(ball_mask)
        ball_x = float(xs.mean() + X_MIN_BALL)
        ball_y = float(ys.mean() + Y_MIN)
    else:
        ball_x, ball_y = -1.0, -1.0

    cropped_right = frame[Y_MIN:Y_MAX, X_MIN_RIGHT:X_MAX_RIGHT, :]
    right_mask = np.all(cropped_right == RIGHT_PADDLE_COLOR, axis=2)
    if right_mask.any():
        ys = np.where(right_mask)[0]
        right_paddle_y = float(ys.mean() + Y_MIN)
    else:
        right_paddle_y = -1.0

    cropped_left = frame[Y_MIN:Y_MAX, X_MIN_LEFT:X_MAX_LEFT, :]
    left_mask = np.all(cropped_left == LEFT_PADDLE_COLOR, axis=2)
    if left_mask.any():
        ys = np.where(left_mask)[0]
        left_paddle_y = float(ys.mean() + Y_MIN)
    else:
        left_paddle_y = -1.0

    return {
        "ball_x": ball_x,
        "ball_y": ball_y,
        "right_paddle_y": right_paddle_y,
        "left_paddle_y": left_paddle_y,
    }


# --- Model agent ------------------------------------------------------------
class TransformerAgent:
    """Wraps the Transformer policy with feature extraction and normalization."""

    def __init__(
        self,
        model: PongTransformer,
        feature_cols: List[str],
        seq_len: int,
        norm_stats: Dict[str, Dict[str, float]],
        device: torch.device,
    ) -> None:
        self.model = model.to(device).eval()
        self.feature_cols = feature_cols
        self.seq_len = seq_len
        self.norm_stats = norm_stats or {}
        self.device = device

        self.buffer: Deque[np.ndarray] = deque(maxlen=seq_len)
        self.prev_positions: Optional[Dict[str, float]] = None

    def reset(self) -> None:
        self.buffer.clear()
        self.prev_positions = None

    def _normalise(self, features: Dict[str, float]) -> np.ndarray:
        vector: List[float] = []
        for col in self.feature_cols:
            value = float(features.get(col, 0.0))
            stats = self.norm_stats.get(col, {})
            mean = float(stats.get("mean", 0.0))
            std = float(stats.get("std", 1.0))
            std = std if abs(std) > 1e-8 else 1.0
            vector.append((value - mean) / std)
        return np.asarray(vector, dtype=np.float32)

    def update_state(self, frame: np.ndarray) -> Dict[str, float]:
        positions = detect_positions(frame)

        if self.prev_positions is not None:
            for key in ("ball_x", "ball_y", "right_paddle_y", "left_paddle_y"):
                if positions[key] < 0 and self.prev_positions[key] >= 0:
                    positions[key] = self.prev_positions[key]

        if self.prev_positions is None:
            ball_vx = ball_vy = right_paddle_vy = 0.0
        else:
            ball_vx = positions["ball_x"] - self.prev_positions["ball_x"]
            ball_vy = positions["ball_y"] - self.prev_positions["ball_y"]
            right_paddle_vy = positions["right_paddle_y"] - self.prev_positions["right_paddle_y"]

        features = {
            "ball_x": positions["ball_x"],
            "ball_y": positions["ball_y"],
            "right_paddle_y": positions["right_paddle_y"],
            "left_paddle_y": positions["left_paddle_y"],
            "ball_vx": ball_vx,
            "ball_vy": ball_vy,
            "right_paddle_vy": right_paddle_vy,
            "dist_right": abs(positions["ball_y"] - positions["right_paddle_y"]),
            "offset_right": positions["ball_y"] - positions["right_paddle_y"],
        }

        normalised = self._normalise(features)
        self.buffer.append(normalised)
        self.prev_positions = positions

        return features

    def _build_sequence(self) -> Optional[np.ndarray]:
        if not self.buffer:
            return None

        sequence = list(self.buffer)
        if len(sequence) < self.seq_len:
            pad = np.repeat(sequence[0][None, :], self.seq_len - len(sequence), axis=0)
            stacked = np.concatenate([pad, np.stack(sequence, axis=0)], axis=0)
        else:
            stacked = np.stack(sequence, axis=0)
        return stacked

    @torch.no_grad()
    def select_action(self) -> Tuple[int, Dict[str, object]]:
        seq = self._build_sequence()
        if seq is None:
            return 0, {"label_index": 0, "probs": None, "reason": "empty_buffer"}

        tensor = torch.from_numpy(seq).unsqueeze(0).to(self.device)
        logits = self.model(tensor)
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
        label_idx = int(np.argmax(probs))
        env_action = LABEL_TO_ACTION[label_idx]
        return env_action, {
            "label_index": label_idx,
            "probs": probs.tolist(),
            "reason": "inference",
        }


# --- Logging ----------------------------------------------------------------
class GameLogger:
    """Handles saving frames, CSV log and replay artefacts."""

    def __init__(self, data_root: Path, enable: bool = True) -> None:
        self.enabled = enable
        if not self.enabled:
            return

        today = time.strftime("%Y%m%d")
        self.data_dir = Path(data_root) / f"game_data_{today}"
        self.replay_dir = self.data_dir / "replay"
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.replay_dir.mkdir(parents=True, exist_ok=True)

        existing = []
        for file in self.replay_dir.glob("partita_*.npy"):
            try:
                existing.append(int(file.stem.split("_")[1]))
            except (IndexError, ValueError):
                continue
        self.partita_id = max(existing, default=0) + 1

        self.csv_path = self.data_dir / "pong_log.csv"
        self.gif_path = self.replay_dir / f"partita_{self.partita_id}.gif"
        self.npy_path = self.replay_dir / f"partita_{self.partita_id}.npy"

        self.records: List[Dict[str, object]] = []
        self.frames: List[np.ndarray] = []

    def add_frame(self, frame: np.ndarray) -> None:
        if self.enabled:
            self.frames.append(np.asarray(frame))

    def record_step(
        self,
        step: int,
        action: int,
        reward: float,
        score_left: int,
        score_right: int,
        probs: Optional[List[float]] = None,
    ) -> None:
        if not self.enabled:
            return

        row: Dict[str, object] = {
            "step": step,
            "action": ACTION_TO_LABEL.get(action, action),
            "reward": reward,
            "score_left": score_left,
            "score_right": score_right,
            "partita": self.partita_id,
        }
        if probs is not None:
            for idx, prob in enumerate(probs):
                row[f"prob_{idx}"] = float(prob)
        self.records.append(row)

    def save(self) -> None:
        if not self.enabled or not self.records:
            return

        df = pd.DataFrame(self.records)
        if self.csv_path.exists():
            df.to_csv(self.csv_path, mode="a", header=False, index=False)
        else:
            df.to_csv(self.csv_path, index=False)
        print(f"[DL_play] Dati salvati in {self.csv_path}")

        if self.frames:
            imageio.mimsave(self.gif_path, self.frames, duration=0.04)
            np.save(self.npy_path, np.array(self.frames, dtype=np.uint8))
            print(f"[DL_play] Replay salvato in {self.gif_path} e {self.npy_path}")


# --- GUI --------------------------------------------------------------------
class ModelPongWindow:
    """Tkinter window that runs the environment loop using the model agent."""

    def __init__(
        self,
        env: gym.Env,
        agent: TransformerAgent,
        initial_obs: np.ndarray,
        logger: Optional[GameLogger],
        frame_skip: int,
        update_ms: int,
        scale: int,
    ) -> None:
        self.env = env
        self.agent = agent
        self.logger = logger
        self.frame_skip = frame_skip
        self.update_ms = update_ms
        self.scale = scale

        self.obs = initial_obs
        self.done = False
        self.force_exit = False
        self.frame_counter = 0
        self.step = 0
        self.score_left = 0
        self.score_right = 0
        self.last_action = 0
        self.last_probs: Optional[List[float]] = None

        self.agent.reset()

        self.root = tk.Tk()
        self.root.title("Pong Transformer - Real-Time Play")
        self.image = self._obs_to_photoimage(self.obs)
        self.canvas = tk.Label(self.root, image=self.image)
        self.canvas.pack()

        self.status_var = tk.StringVar()
        self.status_label = tk.Label(self.root, textvariable=self.status_var, font=("Helvetica", 12))
        self.status_label.pack()
        self._update_status()

        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
        self.root.bind("<Escape>", lambda _: self.on_close())

    def _obs_to_photoimage(self, obs: np.ndarray) -> ImageTk.PhotoImage:
        image = Image.fromarray(obs)
        scaled = image.resize((obs.shape[1] * self.scale, obs.shape[0] * self.scale))
        return ImageTk.PhotoImage(scaled)

    def _update_canvas(self) -> None:
        self.image = self._obs_to_photoimage(self.obs)
        self.canvas.configure(image=self.image)
        self.canvas.image = self.image  # avoid GC

    def _update_status(self) -> None:
        action_name = ACTION_MEANING.get(self.last_action, str(self.last_action))
        probs_txt = ""
        if self.last_probs:
            probs_txt = " | probs: " + ", ".join(f"{p:.2f}" for p in self.last_probs)
        self.status_var.set(
            f"Step {self.step} | Action: {action_name} | Score R:{self.score_right} - L:{self.score_left}{probs_txt}"
        )

    def on_close(self) -> None:
        if not self.force_exit:
            self.force_exit = True
            self.done = True
            if self.logger:
                self.logger.save()
                self.logger = None
            if self.env:
                self.env.close()
                self.env = None
            self.root.destroy()

    def game_loop(self) -> None:
        if self.force_exit:
            return

        if not self.done:
            self.frame_counter += 1
            if self.frame_counter >= self.frame_skip:
                self.agent.update_state(self.obs)
                action, info = self.agent.select_action()
                next_obs, reward, terminated, truncated, _ = self.env.step(action)
                self.logger.add_frame(next_obs) if self.logger else None

                if reward == -1:
                    self.score_left += 1
                elif reward == 1:
                    self.score_right += 1

                if self.logger:
                    self.logger.record_step(
                        step=self.step,
                        action=action,
                        reward=reward,
                        score_left=self.score_left,
                        score_right=self.score_right,
                        probs=info.get("probs"),
                    )

                self.obs = next_obs
                self.last_action = action
                self.last_probs = info.get("probs")
                self.step += 1
                self.frame_counter = 0
                self.done = terminated or truncated

                self._update_canvas()
                self._update_status()

            if not self.done:
                self.root.after(self.update_ms, self.game_loop)
            else:
                self._finalise_game()
        else:
            self._finalise_game()

    def _finalise_game(self) -> None:
        if self.logger:
            self.logger.save()
            self.logger = None
        if self.env:
            self.env.close()
            self.env = None
        if not self.force_exit:
            self.force_exit = True
            self.root.destroy()

    def run(self) -> None:
        self.root.after(0, self.game_loop)
        self.root.mainloop()


# --- Model helpers ----------------------------------------------------------
def load_checkpoint(
    checkpoint_path: Path,
    device: torch.device,
) -> Tuple[PongTransformer, List[str], int, Dict[str, Dict[str, float]]]:
    checkpoint = torch.load(checkpoint_path, map_location=device)

    metadata = checkpoint.get("metadata", {})
    feature_cols = metadata.get("feature_cols") or [
        "ball_x",
        "ball_y",
        "right_paddle_y",
        "left_paddle_y",
        "ball_vx",
        "ball_vy",
        "right_paddle_vy",
        "dist_right",
        "offset_right",
    ]
    seq_len = int(metadata.get("seq_len", 10))
    num_classes = int(metadata.get("num_classes", len(LABEL_TO_ACTION)))
    norm_stats = metadata.get("normalization_stats") or {
        col: {"mean": 0.0, "std": 1.0} for col in feature_cols
    }

    model = PongTransformer(
        input_dim=len(feature_cols),
        seq_len=seq_len,
        num_classes=num_classes,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device).eval()
    return model, feature_cols, seq_len, norm_stats


def create_environment() -> gym.Env:
    gym.register_envs(ale_py)
    return gym.make("ALE/Pong-v5", render_mode="rgb_array", mode=1)


# --- CLI --------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Real-time Pong play with the Transformer agent.")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/pong_transformer_best.pt",
        help="Path to the trained checkpoint.",
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default="data",
        help="Root directory where game_data_*/ folders are stored.",
    )
    parser.add_argument(
        "--device",
        type=str,
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="Device to run inference on.",
    )
    parser.add_argument(
        "--frame-skip",
        type=int,
        default=2,
        help="Number of GUI frames to skip between environment steps.",
    )
    parser.add_argument(
        "--update-ms",
        type=int,
        default=30,
        help="Tkinter loop interval in milliseconds.",
    )
    parser.add_argument(
        "--scale",
        type=int,
        default=2,
        help="GUI scaling factor for the rendered frame.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional random seed for the environment.",
    )
    parser.add_argument(
        "--no-record",
        action="store_true",
        help="Disable saving replay data (CSV/GIF/frames).",
    )
    return parser.parse_args()


def resolve_path(path_str: str, base: Path) -> Path:
    path = Path(path_str)
    if not path.is_absolute():
        path = base / path
    return path


def main() -> None:
    args = parse_args()

    checkpoint_path = resolve_path(args.checkpoint, REPO_ROOT)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint non trovato: {checkpoint_path}")

    device_type = args.device
    if device_type == "auto":
        device_type = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_type)

    model, feature_cols, seq_len, norm_stats = load_checkpoint(checkpoint_path, device)
    agent = TransformerAgent(
        model=model,
        feature_cols=feature_cols,
        seq_len=seq_len,
        norm_stats=norm_stats,
        device=device,
    )

    env = create_environment()
    obs, _ = env.reset(seed=args.seed)

    data_root = resolve_path(args.data_root, REPO_ROOT)
    logger = GameLogger(data_root=data_root, enable=not args.no_record)

    window = ModelPongWindow(
        env=env,
        agent=agent,
        initial_obs=obs,
        logger=logger if logger.enabled else None,
        frame_skip=args.frame_skip,
        update_ms=args.update_ms,
        scale=args.scale,
    )
    window.run()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[DL_play] Interrotto dall'utente.")
