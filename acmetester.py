import tensorflow as tf
import acme

from env.RSEnv import RSEnv
from env.TestRSEnv import TestRSEnv
from acme import environment_loop
from acme import specs
from acme import wrappers
from acme.agents.tf import d4pg
from acme.tf import networks
from acme.tf import utils as tf2_utils
from acme.utils import loggers
import numpy as np
import sonnet as snt

import gym


p_save = tf.keras.models.load_model('p3_save')

returns = 0
environment = TestRSEnv()
state = np.expand_dims(environment.reset(), axis = 0)
state.astype(np.float32)
done = False
while not done:
  action = p_save.inference(state)
  action = np.squeeze(action, axis=0)
  state, reward, done, _  = environment.step(action)
  state = np.expand_dims(state, axis = 0)
  state.astype(np.float32)
  returns += reward
  environment.render()
print(returns)
environment.close()
