#!/usr/bin/env python3
"""
Policy-gradient fine-tuning for the Pong Transformer.

The script plays repeated episodes against the ALE Pong environment, updates
the Transformer with REINFORCE + entropy regularisation, and stores the model
that achieves the best score difference.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import deque
from pathlib import Path
from typing import Deque, Dict, List, Optional, Tuple

import ale_py  # noqa: F401 (Gym registration side-effect)
import gymnasium as gym
import numpy as np
import torch
from torch.distributions import Categorical
import tkinter as tk
from PIL import Image, ImageTk

# Ensure repository root on sys.path so relative imports work when executed
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.DL_play import (  # noqa: E402
    ACTION_MEANING,
    ACTION_TO_LABEL,
    LABEL_TO_ACTION,
    GameLogger,
    detect_positions,
    load_checkpoint,
    Y_MIN,
    Y_MAX,
)


def create_environment() -> gym.Env:
    """Instantiate the Pong environment with RGB observations."""
    gym.register_envs(ale_py)
    return gym.make("ALE/Pong-v5", render_mode="rgb_array", mode=1)


class FeaturePipeline:
    """
    Maintains the temporal feature queue required by the Transformer.

    Frames are converted to the same engineered features used during
    supervised training, normalised with the statistics bundled in the
    checkpoint metadata, and padded to build sequences of length `seq_len`.
    """

    def __init__(
        self,
        feature_cols: List[str],
        seq_len: int,
        norm_stats: Dict[str, Dict[str, float]],
    ) -> None:
        self.feature_cols = list(feature_cols)
        self.seq_len = seq_len
        self.norm_stats = norm_stats
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
            if abs(std) < 1e-8:
                std = 1.0
            vector.append((value - mean) / std)
        return np.asarray(vector, dtype=np.float32)

    def update(self, frame: np.ndarray) -> np.ndarray:
        positions = detect_positions(frame)

        if self.prev_positions is not None:
            for key in ("ball_x", "ball_y", "right_paddle_y", "left_paddle_y"):
                if positions[key] < 0 and self.prev_positions[key] >= 0:
                    positions[key] = self.prev_positions[key]
        else:
            for key in positions:
                if positions[key] < 0:
                    positions[key] = 0.0

        if self.prev_positions is None:
            ball_vx = 0.0
            ball_vy = 0.0
            right_paddle_vy = 0.0
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
        return normalised

    def build_sequence(self, device: torch.device) -> torch.Tensor:
        if not self.buffer:
            raise RuntimeError("Feature buffer vuoto: chiamare update() almeno una volta prima di build_sequence().")

        sequence = list(self.buffer)
        if len(sequence) < self.seq_len:
            pad = np.repeat(sequence[0][None, :], self.seq_len - len(sequence), axis=0)
            stacked = np.concatenate([pad, np.stack(sequence, axis=0)], axis=0)
        else:
            stacked = np.stack(sequence[-self.seq_len :], axis=0)

        tensor = torch.from_numpy(stacked.astype(np.float32)).unsqueeze(0)
        return tensor.to(device)


class LiveDisplay:
    """Simple Tk window to visualise the episode while training."""

    def __init__(self, title: str = "Pong RL - Training", scale: int = 2) -> None:
        self.scale = max(1, scale)
        self.closed = False
        self.root = tk.Tk()
        self.root.title(title)
        self.root.resizable(False, False)

        self.image_label = tk.Label(self.root)
        self.image_label.pack()

        self.status_var = tk.StringVar()
        self.status_label = tk.Label(
            self.root,
            textvariable=self.status_var,
            font=("Helvetica", 12),
            anchor="w",
            justify="left",
            width=80,
        )
        self.status_label.pack()

        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
        self.photo: Optional[ImageTk.PhotoImage] = None
        self._frame_width: Optional[int] = None
        self._frame_height: Optional[int] = None

    def on_close(self) -> None:
        if not self.closed:
            self.closed = True
            self.root.destroy()

    def render(
        self,
        frame: np.ndarray,
        step: int,
        reward: float,
        score_right: int,
        score_left: int,
        action: Optional[int],
        probs: Optional[np.ndarray] = None,
    ) -> None:
        if self.closed:
            return

        try:
            if self._frame_width is None or self._frame_height is None:
                self._frame_width = frame.shape[1] * self.scale
                self._frame_height = frame.shape[0] * self.scale
                self.root.geometry(f"{self._frame_width}x{self._frame_height + 60}")

            image = Image.fromarray(frame)
            image = image.resize((self._frame_width, self._frame_height))
            self.photo = ImageTk.PhotoImage(image)
            self.image_label.configure(image=self.photo)
            self.image_label.image = self.photo

            action_name = ACTION_MEANING.get(action, "-") if action is not None else "-"
            reward_str = f"{reward:+.0f}"
            probs_str = ""
            if probs is not None:
                probs_fmt = ", ".join(f"{p:.2f}" for p in probs.tolist())
                probs_str = f" | probs: [{probs_fmt}]"

            self.status_var.set(
                f"Step {step} | action: {action_name} | reward: {reward_str} | "
                f"score R:{score_right} - L:{score_left}{probs_str}"
            )

            self.root.update_idletasks()
            self.root.update()
        except tk.TclError:
            self.closed = True

    def close(self) -> None:
        if not self.closed:
            self.closed = True
            self.root.destroy()


def compute_returns(rewards: List[float], gamma: float) -> List[float]:
    returns: List[float] = []
    running = 0.0
    for reward in reversed(rewards):
        running = reward + gamma * running
        returns.append(running)
    returns.reverse()
    return returns


def run_episode(
    env: gym.Env,
    model: torch.nn.Module,
    feature_pipe: FeaturePipeline,
    device: torch.device,
    gamma: float,
    entropy_coef: float,
    normalize_returns: bool,
    logger: Optional[GameLogger],
    max_steps: Optional[int],
    seed: Optional[int],
    display: Optional[LiveDisplay],
    dist_reward_coef: float,
    edge_penalty: float,
    edge_threshold: float,
    stuck_threshold: int,
    forced_down_steps: int,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    obs, _ = env.reset(seed=seed)
    feature_pipe.reset()
    feature_pipe.update(obs)

    if logger is not None:
        logger.add_frame(obs)

    if display is not None and not display.closed:
        display.render(
            frame=obs,
            step=0,
            reward=0.0,
            score_right=0,
            score_left=0,
            action=None,
            probs=None,
        )

    log_probs: List[torch.Tensor] = []
    entropies: List[torch.Tensor] = []
    policy_rewards: List[float] = []
    score_left = 0
    score_right = 0
    stuck_counter = 0
    forced_down_remaining = 0
    forced_events = 0

    done = False
    steps = 0
    while not done:
        seq = feature_pipe.build_sequence(device)
        logits = model(seq)
        dist = Categorical(logits=logits.squeeze(0))

        current_positions = feature_pipe.prev_positions or {}
        paddle_y = current_positions.get("right_paddle_y", -1.0)
        if paddle_y >= 0 and paddle_y <= Y_MIN + 1:
            stuck_counter += 1
        else:
            stuck_counter = 0

        if stuck_counter >= stuck_threshold and forced_down_remaining <= 0:
            forced_down_remaining = max(1, forced_down_steps)
            forced_events += 1
            stuck_counter = 0

        if forced_down_remaining > 0:
            forced_idx = torch.tensor(ACTION_TO_LABEL[3], dtype=torch.int64, device=device)
            log_prob = dist.log_prob(forced_idx)
            entropy = dist.entropy()
            action_idx_value = forced_idx.item()
            env_action = LABEL_TO_ACTION[action_idx_value]
            forced_down_remaining -= 1
        else:
            action_idx = dist.sample()
            env_action = LABEL_TO_ACTION[int(action_idx.item())]
            log_prob = dist.log_prob(action_idx)
            entropy = dist.entropy()
            action_idx_value = int(action_idx.item())

        probs_np = dist.probs.detach().cpu().numpy()

        next_obs, reward, terminated, truncated, _ = env.step(env_action)
        done = terminated or truncated

        raw_reward = float(reward)
        log_probs.append(log_prob)
        entropies.append(entropy)

        if raw_reward == -1:
            score_left += 1
        elif raw_reward == 1:
            score_right += 1

        if logger is not None:
            logger.add_frame(next_obs)
            logger.record_step(
                step=steps,
                action=env_action,
                reward=raw_reward,
                score_left=score_left,
                score_right=score_right,
                probs=probs_np.tolist(),
            )

        feature_pipe.update(next_obs)
        positions = feature_pipe.prev_positions or {}

        shaped_reward = raw_reward
        ball_y = positions.get("ball_y", -1.0)
        paddle_y = positions.get("right_paddle_y", -1.0)
        if ball_y >= 0 and paddle_y >= 0:
            shaped_reward -= dist_reward_coef * abs(ball_y - paddle_y)
            if edge_penalty > 0:
                if paddle_y <= Y_MIN + edge_threshold or paddle_y >= Y_MAX - edge_threshold:
                    shaped_reward -= edge_penalty

        policy_rewards.append(float(shaped_reward))

        if display is not None and not display.closed:
            display.render(
                frame=next_obs,
                step=steps + 1,
                reward=raw_reward,
                score_right=score_right,
                score_left=score_left,
                action=env_action,
                probs=probs_np,
            )

        obs = next_obs
        steps += 1

        if max_steps is not None and steps >= max_steps:
            break

    returns = compute_returns(policy_rewards, gamma)
    returns_tensor = torch.tensor(returns, dtype=torch.float32, device=device)
    if normalize_returns and returns_tensor.numel() > 1:
        returns_tensor = (returns_tensor - returns_tensor.mean()) / (returns_tensor.std(unbiased=False) + 1e-8)

    if not log_probs:
        loss = torch.zeros((), dtype=torch.float32, device=device, requires_grad=True)
    else:
        log_probs_tensor = torch.stack(log_probs)
        entropies_tensor = torch.stack(entropies)

        loss = -(returns_tensor * log_probs_tensor).sum()
        if entropy_coef > 0:
            loss -= entropy_coef * entropies_tensor.sum()

    score_diff = float(score_right - score_left)
    info = {
        "score_diff": score_diff,
        "policy_reward": float(sum(policy_rewards)),
        "steps": float(steps),
        "score_left": float(score_left),
        "score_right": float(score_right),
        "forced_events": float(forced_events),
    }
    return loss, info


def save_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    metadata: Dict[str, object],
    episode: int,
    best_diff: float,
) -> None:
    payload = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "episode": episode,
        "best_score_diff": best_diff,
        "metadata": metadata,
        "origin": "rl_train",
    }
    torch.save(payload, path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune the Pong Transformer with policy-gradient RL.")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/pong_transformer_best.pt", help="Checkpoint da cui partire.")
    parser.add_argument("--output-dir", type=str, default="checkpoints", help="Directory di salvataggio.")
    parser.add_argument("--save-prefix", type=str, default="pong_transformer_rl", help="Prefisso dei checkpoint RL.")
    parser.add_argument("--episodes", type=int, default=50, help="Numero di episodi di training.")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor.")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate Adam.")
    parser.add_argument("--entropy-coef", type=float, default=0.01, help="Peso della regularizzazione entropica.")
    parser.add_argument("--normalize-returns", action="store_true", help="Normalizza i returns episodio per ridurre la varianza.")
    parser.add_argument("--grad-clip", type=float, default=1.0, help="Clip L2 dei gradienti (0 per disattivare).")
    parser.add_argument("--seed", type=int, default=None, help="Seed base per l'ambiente.")
    parser.add_argument("--device", type=str, choices=("auto", "cpu", "cuda"), default="auto", help="Dispositivo per il training.")
    parser.add_argument("--save-every", type=int, default=10, help="Salva un checkpoint 'last' ogni N episodi.")
    parser.add_argument("--record-every", type=int, default=0, help="Registra replay + log ogni N episodi (0 per disattivare).")
    parser.add_argument("--max-steps", type=int, default=None, help="Limite massimo di step per episodio.")
    parser.add_argument("--log-jsonl", type=str, default="logs/rl_train.jsonl", help="File JSONL per logging metriche.")
    parser.add_argument("--data-root", type=str, default="data", help="Directory base per salvataggio replay quando si abilita --record-every.")
    parser.add_argument("--show", action="store_true", help="Mostra la partita in tempo reale durante il training.")
    parser.add_argument("--display-scale", type=int, default=2, help="Fattore di scala per la finestra di visualizzazione.")
    parser.add_argument("--dist-reward-coef", type=float, default=1e-3, help="Penalizza la distanza verticale dalla palla.")
    parser.add_argument("--edge-penalty", type=float, default=0.05, help="Penalità aggiuntiva quando la racchetta resta vicina ai bordi.")
    parser.add_argument("--edge-threshold", type=float, default=8.0, help="Soglia (in pixel) dal bordo per applicare la edge penalty.")
    parser.add_argument("--stuck-threshold", type=int, default=15, help="Numero di step consecutivi vicino al bordo per attivare il fix anti-stuck.")
    parser.add_argument("--forced-down-steps", type=int, default=12, help="Numero di step 'DOWN' forzati quando scatta l'anti-stuck.")
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
    model.to(device).train()

    output_num_classes = getattr(model.classifier[-1], "out_features", len(LABEL_TO_ACTION))
    metadata = {
        "feature_cols": feature_cols,
        "seq_len": seq_len,
        "num_classes": int(output_num_classes),
        "normalization_stats": norm_stats,
        "rl_origin": str(checkpoint_path),
    }

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    env = create_environment()
    display: Optional[LiveDisplay] = None
    if args.show:
        try:
            display = LiveDisplay(scale=args.display_scale)
        except tk.TclError as exc:
            print(f"[WARN] Impossibile creare la finestra Tkinter ({exc}). Procedo senza visualizzazione.")
            display = None

    feature_pipe = FeaturePipeline(feature_cols=feature_cols, seq_len=seq_len, norm_stats=norm_stats)

    output_dir = resolve_path(args.output_dir, REPO_ROOT)
    output_dir.mkdir(parents=True, exist_ok=True)
    best_path = output_dir / f"{args.save_prefix}_best.pt"
    last_path = output_dir / f"{args.save_prefix}_last.pt"

    log_path = resolve_path(args.log_jsonl, REPO_ROOT)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    best_diff = -float("inf")

    for episode in range(1, args.episodes + 1):
        record = args.record_every > 0 and episode % args.record_every == 0
        logger: Optional[GameLogger] = None
        if record:
            logger = GameLogger(data_root=resolve_path(args.data_root, REPO_ROOT), enable=True)

        seed = args.seed + episode if args.seed is not None else None
        loss, info = run_episode(
            env=env,
            model=model,
            feature_pipe=feature_pipe,
            device=device,
            gamma=args.gamma,
            entropy_coef=args.entropy_coef,
            normalize_returns=args.normalize_returns,
            logger=logger,
            max_steps=args.max_steps,
            seed=seed,
            display=display,
            dist_reward_coef=args.dist_reward_coef,
            edge_penalty=args.edge_penalty,
            edge_threshold=args.edge_threshold,
            stuck_threshold=args.stuck_threshold,
            forced_down_steps=args.forced_down_steps,
        )

        optimizer.zero_grad()
        loss.backward()
        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()

        if logger is not None:
            logger.save()

        episode_diff = info["score_diff"]
        is_best = episode_diff > best_diff
        if is_best:
            best_diff = episode_diff
            save_checkpoint(best_path, model, optimizer, metadata, episode, best_diff)

        if args.save_every and episode % args.save_every == 0:
            save_checkpoint(last_path, model, optimizer, metadata, episode, best_diff)

        metric = {
            "episode": episode,
            "loss": float(loss.detach().cpu().item()),
            "score_diff": float(episode_diff),
            "policy_reward": info["policy_reward"],
            "score_right": info["score_right"],
            "score_left": info["score_left"],
            "steps": info["steps"],
            "best_diff": float(best_diff),
            "checkpoint": str(best_path if is_best else last_path),
            "forced_events": info["forced_events"],
        }
        with open(log_path, "a", encoding="utf-8") as log_file:
            log_file.write(json.dumps(metric) + "\n")

        status = (
            f"[RL] Episode {episode}/{args.episodes} | diff: {episode_diff:.0f} "
            f"(R:{info['score_right']:.0f} - L:{info['score_left']:.0f}) | "
            f"steps: {info['steps']:.0f} | π-reward: {info['policy_reward']:.2f} | "
            f"forced: {int(info['forced_events'])} | loss: {metric['loss']:.4f}"
        )
        if is_best:
            status += " <- best"
        print(status)

    save_checkpoint(last_path, model, optimizer, metadata, args.episodes, best_diff)
    env.close()
    if display is not None and not display.closed:
        display.close()


if __name__ == "__main__":
    main()
