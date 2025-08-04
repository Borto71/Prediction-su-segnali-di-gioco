import os
import time
import gymnasium as gym
import ale_py
import pandas as pd
import imageio
import numpy as np
import tkinter as tk
from PIL import Image, ImageTk

ACTION_MEANING = {0: "NOOP", 2: "UP", 3: "DOWN"}
KEY_ACTIONS = {"w": 2, "s": 3}

gym.register_envs(ale_py)
env = gym.make('ALE/Pong-v5', render_mode="rgb_array", mode=1)
obs, info = env.reset()
done = False

data = []
frames = []

today = time.strftime("%Y%m%d")
DATA_DIR = f"game_data_{today}"
REPLAY_DIR = os.path.join(DATA_DIR, "replay")
os.makedirs(REPLAY_DIR, exist_ok=True)

# Trova prossimo numero partita
existing = [int(f.split("_")[1].split(".")[0]) for f in os.listdir(REPLAY_DIR) if f.startswith("partita_") and f.endswith(".npy")]
next_partita = max(existing, default=0) + 1

csv_path = os.path.join(DATA_DIR, f'pong_log.csv')
gif_path = os.path.join(REPLAY_DIR, f'partita_{next_partita}.gif')
npy_path = os.path.join(REPLAY_DIR, f'partita_{next_partita}.npy')

class PongWindow:
    def __init__(self, obs):
        self.root = tk.Tk()
        self.root.title(f"Pong Atari - Partita {next_partita}")
        self.img = self.obs_to_photoimage(obs)
        self.label = tk.Label(self.root, image=self.img)
        self.label.pack()
        self.obs = obs
        self.action = 0
        self.pressed_keys = set()
        self.done = False
        self.force_exit = False  # <--- per chiusura istantanea
        self.step = 0

        self.frame_skip = 2       # <- tuning sensibilità (2 = più naturale)
        self.frame_counter = 0

        self.score_left = 0
        self.score_right = 0

        self.root.bind("<KeyPress>", self.on_key_down)
        self.root.bind("<KeyRelease>", self.on_key_up)
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

        self.root.after(0, self.game_loop)
        self.root.mainloop()

    def obs_to_photoimage(self, obs):   
        scale = 1.2
        image = Image.fromarray(obs)
        image = image.resize((int(obs.shape[1]*scale), int(obs.shape[0]*scale)))
        return ImageTk.PhotoImage(image)

    def on_key_down(self, event):
        key = event.keysym.lower()
        if key in KEY_ACTIONS:
            self.pressed_keys.add(key)

    def on_key_up(self, event):
        key = event.keysym.lower()
        self.pressed_keys.discard(key)

    def get_current_action(self):
        if "w" in self.pressed_keys:
            return 2
        elif "s" in self.pressed_keys:
            return 3
        else:
            return 0

    def on_close(self):
        if not self.force_exit:
            self.force_exit = True
            self.done = True
            self.save_results()
            self.root.destroy()

    def game_loop(self):
        global env, done, data, frames
        if self.force_exit:
            return
        if not self.done:
            self.frame_counter += 1
            if self.frame_counter >= self.frame_skip:
                self.action = self.get_current_action()
                obs, reward, terminated, truncated, info = env.step(self.action)
                self.obs = obs
                self.img = self.obs_to_photoimage(obs)
                self.label.configure(image=self.img)
                self.label.image = self.img

                if reward == -1:
                    self.score_left += 1
                elif reward == 1:
                    self.score_right += 1

                action_map = {0: 0, 2: 1, 3: 2}
                data.append({
                    'step': self.step,
                    'action': action_map.get(self.action, self.action),
                    'reward': reward,
                    'score_left': self.score_left,
                    'score_right': self.score_right,
                    'partita': next_partita
                })
                frames.append(obs)

                print(
                    f"Step {self.step} | Action: {self.action} | Reward: {reward} | "
                    f"Score left: {self.score_left} | Score right: {self.score_right}"
                )

                self.done = terminated or truncated
                self.step += 1
                self.frame_counter = 0

            self.root.after(30, self.game_loop)   # ~33fps
        else:
            if not self.force_exit:   # Solo se non chiuso con la X
                print(
                    f"Game Over! Final score: Left={self.score_left} | Right={self.score_right}"
                )
                self.save_results()
                self.root.destroy()

    def save_results(self):
        df = pd.DataFrame(data)
        if os.path.exists(csv_path):
            df.to_csv(csv_path, mode='a', header=False, index=False)
        else:
            df.to_csv(csv_path, index=False)
        print(f"Dati salvati in {csv_path}!")
        imageio.mimsave(gif_path, frames, duration=0.04)
        print(f"GIF salvata come {gif_path}!")
        np.save(npy_path, np.array(frames))
        print(f"Frames salvati in {npy_path}!")

PongWindow(obs)
env.close()
