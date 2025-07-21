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

    # Mostra solo quando reward è diverso da zero
    if reward != 0:
        print(f"Step {steps} - Action {action} - Reward {reward} - Score {score}")

    done = terminated or truncated
    steps += 1


env.close()

df = pd.DataFrame(data)
df.to_csv('pong_log.csv', index=False)
print("Dati salvati in pong_log.csv!")
