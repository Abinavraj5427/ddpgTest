import gym

from stable_baselines.common.policies import MlpPolicy
from stable_baselines import PPO1
from env.RSEnv import RSEnv
from env.TestRSEnv import TestRSEnv

env = RSEnv()

#model = PPO1(MlpPolicy, env, verbose=1)
model = PPO1.load("sbppov3")
model.set_env(env)
model.learn(total_timesteps=3000000, log_interval=10, reset_num_timesteps=False)
model.save("sbppov4")

env = TestRSEnv()
obs = env.reset()
done = False
while not done:
    action, _ = model.predict(obs)
    obs, rewards, done, info = env.step(action)
    env.render()
env.close()
