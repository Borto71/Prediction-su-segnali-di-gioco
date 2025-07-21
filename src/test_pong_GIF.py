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
