import gym
import numpy as np

from env import RSEnv
from stable_baselines.ddpg.policies import MlpPolicy
from stable_baselines.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise, AdaptiveParamNoiseSpec
from stable_baselines import DDPG


env = RSEnv()

# the noise objects for DDPG
n_actions = env.action_space.shape[-1]
param_noise = None
action_noise = OrnsteinUhlenbeckActionNoise(mean=np.zeros(n_actions), signma=float(0.5) * np.ones(n_actions))

model = DDPG(MlpPolicy, env, verbose=1, param_noise=param_noise, action_noise=action_noise)
model.learn(total_timesteps=1000, log_interval=10)
model.save("sbddpg")
obs = env.reset()
done = False
while not done:
    action, _ = model.predict(obs)
    obs, rewards, done, info = env.step(action)
    env.render()
env.close()