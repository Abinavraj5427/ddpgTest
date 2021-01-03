import gym
from env.TestRSEnv import TestRSEnv
from stable_baselines import A2C, PPO1, DDPG, TRPO

#model = A2C.load("sba2c")
#model = PPO1.load("sbppo")
#model = DDPG.load("sbddpg")
#model = PPO1.load("sbppov2")
model = TRPO.load("sbtrpo")
#model = PPO1.load("sbppov3")

returns = 0
env = TestRSEnv()
obs = env.reset()
done = False
while not done:
	action, _ = model.predict(obs)
	obs, rewards, done, info = env.step(action)
	env.render()
	returns+=rewards
env.close()
print(returns)
