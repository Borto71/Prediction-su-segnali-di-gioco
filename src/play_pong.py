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
KEY_ACTIONS = {"w": 2, "s": 3}  # w = up, s = down

gym.register_envs(ale_py)
env = gym.make('ALE/Pong-v5', render_mode="rgb_array", mode=1)
obs, info = env.reset()
done = False

data = []
frames = []

today = time.strftime("%Y%m%d")
DATA_DIR = f"game_data_{today}"
os.makedirs(DATA_DIR, exist_ok=True)

csv_path = os.path.join(DATA_DIR, 'pong_log.csv')
gif_path = os.path.join(DATA_DIR, 'pong_run.gif')
npy_path = os.path.join(DATA_DIR, 'pong_obs.npy')

class PongWindow:
    def __init__(self, obs):
        self.root = tk.Tk()
        self.root.title("Pong Atari - Gioca tu!")
        self.img = self.obs_to_photoimage(obs)
        self.label = tk.Label(self.root, image=self.img)
        self.label.pack()
        self.obs = obs
        self.action = 0
        self.done = False
        self.step = 0

        self.frame_skip = 2
        self.frame_counter = 0

        self.score_left = 0
        self.score_right = 0

        self.root.bind("<KeyPress>", self.on_key_down)
        self.root.bind("<KeyRelease>", self.on_key_up)
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

        self.root.after(0, self.game_loop)
        self.root.mainloop()

    def obs_to_photoimage(self, obs):
        image = Image.fromarray(obs)
        image = image.resize((obs.shape[1]*2, obs.shape[0]*2))
        return ImageTk.PhotoImage(image)

    def on_key_down(self, event):
        key = event.keysym.lower()
        if key in KEY_ACTIONS:
            self.action = KEY_ACTIONS[key]

    def on_key_up(self, event):
        self.action = 0  # NOOP quando non premi

    def on_close(self):
        self.save_results()
        self.root.destroy()

    def game_loop(self):
        global env, done, data, frames
        if not self.done:
            self.frame_counter += 1
            if self.frame_counter >= self.frame_skip:
                obs, reward, terminated, truncated, info = env.step(self.action)
                self.obs = obs
                self.img = self.obs_to_photoimage(obs)
                self.label.configure(image=self.img)
                self.label.image = self.img

                if reward == -1:
                    self.score_left += 1
                elif reward == 1:
                    self.score_right += 1

                data.append({
                    'step': self.step,
                    'action': self.action,
                    'reward': reward,
                    'score_left': self.score_left,
                    'score_right': self.score_right
                })
                frames.append(obs)

                print(
                    f"Step {self.step} | Action: {self.action} | Reward: {reward} | "
                    f"Score left: {self.score_left} | Score right: {self.score_right}"
                )

                self.done = terminated or truncated
                self.step += 1
                self.frame_counter = 0

            self.root.after(40, self.game_loop)
        else:
            print(
                f"Game Over! Final score: Left={self.score_left} | Right={self.score_right}"
            )
            self.save_results()
            self.root.destroy()

    def save_results(self):
        df = pd.DataFrame(data)
        df.to_csv(csv_path, index=False)
        print(f"Dati salvati in {csv_path}!")
        imageio.mimsave(gif_path, frames, duration=0.04)
        print(f"GIF salvata come {gif_path}!")
        np.save(npy_path, np.array(frames))
        print(f"Frames salvati in {npy_path}!")

PongWindow(obs)
env.close()
