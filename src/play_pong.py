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
env = gym.make('ALE/Pong-v5', render_mode="rgb_array")
obs, info = env.reset()
done = False
score = 0

data = []
frames = []

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

        self.root.bind("<KeyPress>", self.on_key_down)
        self.root.bind("<KeyRelease>", self.on_key_up)
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)  # <-- AGGIUNTO

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
        global env, done, score, data, frames
        if not self.done:
            obs, reward, terminated, truncated, info = env.step(self.action)
            self.obs = obs
            self.img = self.obs_to_photoimage(obs)
            self.label.configure(image=self.img)
            self.label.image = self.img
            score = info.get('score', score)

            # Logging dati e frame per GIF
            data.append({'step': self.step, 'action': self.action, 'reward': reward, 'score': score})
            frames.append(obs)

            print(f"Step {self.step} | Action: {self.action} | Reward: {reward} | Score: {score}")

            self.done = terminated or truncated
            self.step += 1
            self.root.after(40, self.game_loop)  # ~25 fps
        else:
            print("Game Over! Score:", score)
            self.save_results()
            self.root.destroy()

    def save_results(self):
        df = pd.DataFrame(data)
        df.to_csv('pong_log.csv', index=False)
        print("Dati salvati in pong_log.csv!")
        imageio.mimsave('pong_run.gif', frames, duration=0.04)
        print("GIF salvata come pong_run.gif!")

PongWindow(obs)
env.close()
