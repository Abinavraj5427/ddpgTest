import gym
from env import RSEnv 
from stable_baselines.common.policies import MlpPolicy
from stable_baselines import A2C

# Parallel environments
env = RSEnv()

model = A2C(MlpPolicy, env)
model.learn(total_timesteps=1000, log_interval=10)

model.save("sba2c")

obs = env.reset()
done = False
while not done:
    action, _ = model.predict(obs)
    obs, rewards, done, info = env.step(action)
    env.render()
env.close()