import gym

from stable_baselines.common.policies import MlpPolicy
from stable_baselines import PPO1
from env import RSEnv

env = RSEnv()

model = PPO1(MlpPolicy, env, verbose=1)
model.learn(total_timesteps=1000, log_interval=10)
model.save("sbppo")

obs = env.reset()
done = False
while not done:
    action, _ = model.predict(obs)
    obs, rewards, done, info = env.step(action)
    env.render()
env.close()