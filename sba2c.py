import gym
from env.RSEnv import RSEnv
from env.TestRSEnv import TestRSEnv 
from stable_baselines.common.policies import MlpPolicy
from stable_baselines import A2C
from stable_baselines.common.vec_env import SubprocVecEnv
from stable_baselines.common import set_global_seeds

def make_env(rank, seed=0):
	def _init():
		env = RSEnv()
		env.seed(seed+rank)
		return env
	set_global_seeds(seed)
	return _init

def test():
	# Parallel environments
	n_cpu = 4
	env = SubprocVecEnv([lambda: RSEnv() for i in range(n_cpu)])

	model = A2C(MlpPolicy, env, verbose=1)
	model.learn(total_timesteps=600000, log_interval=10)

	model.save("sba2c")

	env = TestRSEnv()
	obs = env.reset()
	done = False
	while not done:
	    action, _ = model.predict(obs)
	    obs, rewards, done, info = env.step(action)
	    env.render()
	env.close()

if __name__ == '__main__':
	test()
