import gym

from env.RSEnv import RSEnv
from env.TestRSEnv import TestRSEnv 
from stable_baselines.common.policies import MlpPolicy
from stable_baselines import TRPO

env = RSEnv()

#model = TRPO(MlpPolicy, env, verbose=1)
model = TRPO.load("sbtrpo")
model.set_env(env)
model.learn(total_timesteps=1200000, log_interval=10)
model.save("sbtrpov2")

env = TestRSEnv()
obs = env.reset()
done = False
while not done:
    action, _ = model.predict(obs)
    obs, rewards, done, info = env.step(action)
    env.render()
env.close()
