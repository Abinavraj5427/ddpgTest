
import tensorflow as tf

from env.RSCNN import RSCNN
from env.TRSCNN import TRSCNN
from acme import environment_loop
from acme import specs
from acme import wrappers
from acme.agents.tf import d4pg
from acme.tf import networks
from acme.tf import utils as tf2_utils
from acme.utils import loggers
import numpy as np
import sonnet as snt
from sntcnn import LayerNormCNN

import gym

environment = RSCNN()
environment = wrappers.GymWrapper(environment)  # To dm_env interface.

# Make sure the environment outputs single-precision floats.
environment = wrappers.SinglePrecisionWrapper(environment)

# Grab the spec of the environment.
environment_spec = specs.make_environment_spec(environment)

#@title Build agent networks
# BUILDING A D4PG AGENT

# Get total number of action dimensions from action spec.
num_dimensions = np.prod(environment_spec.actions.shape, dtype=int)

# Create the shared observation network; here simply a state-less operation.
observation_network = tf2_utils.batch_concat

# Create the deterministic policy network.
policy_network = snt.Sequential([
    LayerNormCNN((256, 256, 256), activate_final=True),
    networks.NearZeroInitializedLinear(num_dimensions),
    networks.TanhToSpec(environment_spec.actions),
])

# Create the distributional critic network.
critic_network = snt.Sequential([
    # The multiplexer concatenates the observations/actions.
    networks.CriticMultiplexer(),
    LayerNormCNN((512, 512, 256), activate_final=True),
    networks.DiscreteValuedHead(vmin=-150., vmax=150., num_atoms=51),
])

# Create a logger for the agent and environment loop.
agent_logger = loggers.TerminalLogger(label='agent', time_delta=10.)
env_loop_logger = loggers.TerminalLogger(label='env_loop', time_delta=10.)

# Create the D4PG agent.
agent = d4pg.D4PG(
    environment_spec=environment_spec,
    policy_network=policy_network,
    critic_network=critic_network,
    observation_network=observation_network,
    sigma=1.0,
    logger=agent_logger,
    checkpoint=False
)

# Create an loop connecting this agent to the environment created above.
env_loop = environment_loop.EnvironmentLoop(
    environment, agent, logger=env_loop_logger)

# Run a `num_episodes` training episodes.
# Rerun this cell until the agent has learned the given task.
env_loop.run(num_episodes=1000)

tf.save_model(policy_network, "d4pg_policy")
# tf.save_model(critic_network, "d4pg_critic")

environment = TRSCNN()
environment = wrappers.GymWrapper(environment)
environment = wrappers.SinglePrecisionWrapper(environment)

timestep = environment.reset()
while not timestep.last():
  # Simple environment loop.
  action = agent.select_action(timestep.observation)
  timestep = environment.step(action)
  environment.render()

environment.close()
