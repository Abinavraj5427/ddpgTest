import copy
import pyvirtualdisplay
import imageio 
import base64

from acme import environment_loop
from acme.tf import networks
from acme.adders import reverb as adders
from acme.agents.tf import actors as actors
from acme.datasets import reverb as datasets
from acme.wrappers import gym_wrapper
from acme import specs
from acme import wrappers
from acme.agents.tf import d4pg
from acme.agents import agent
from acme.tf import utils as tf2_utils
from acme.utils import loggers

import gym 
import dm_env
import matplotlib.pyplot as plt
import numpy as np
import reverb
import sonnet as snt
import tensorflow as tf

from env.RSEnv import RSEnv
from env.TestRSEnv import TestRSEnv

# Import dm_control if it exists.
try:
  from dm_control import suite
except (ModuleNotFoundError, OSError):
  pass

# Set up a virtual display for rendering OpenAI gym environments.
# display = pyvirtualdisplay.Display(visible=0, size=(1400, 900)).start()


environment = RSEnv()
environment = wrappers.GymWrapper(environment)  # To dm_env interface.

# Make sure the environment outputs single-precision floats.
environment = wrappers.SinglePrecisionWrapper(environment)

environment_spec = specs.make_environment_spec(environment)

print('actions:\n', environment_spec.actions, '\n')
print('observations:\n', environment_spec.observations, '\n')
print('rewards:\n', environment_spec.rewards, '\n')
print('discounts:\n', environment_spec.discounts, '\n')

# Get total number of action dimensions from action spec.
num_dimensions = np.prod(environment_spec.actions.shape, dtype=int)

# Create the deterministic policy network.
policy_network = snt.Sequential([
    networks.LayerNormMLP((256, 256, 256), activate_final=True),
    networks.NearZeroInitializedLinear(num_dimensions),
    networks.TanhToSpec(environment_spec.actions),
])

# Create the distributional critic network.
critic_network = snt.Sequential([
    # The multiplexer concatenates the observations/actions.
    networks.CriticMultiplexer(),
    networks.LayerNormMLP((512, 512, 256), activate_final=True),
    networks.DiscreteValuedHead(vmin=-150., vmax=150., num_atoms=51),
])

replay_buffer = reverb.Table(
    name=adders.DEFAULT_PRIORITY_TABLE,
    max_size=1000000,
    remover=reverb.selectors.Fifo(),
    sampler=reverb.selectors.Uniform(),
    rate_limiter=reverb.rate_limiters.MinSize(min_size_to_sample=1))

replay_table_name = adders.DEFAULT_PRIORITY_TABLE

# Get the server and address so we can give it to the modules such as our actor
# that will interact with the replay buffer.
replay_server = reverb.Server([replay_buffer], port=None)
replay_server_address = 'localhost:%d' % replay_server.port

# Create a 5-step transition adder where in between those steps a discount of
# 0.99 is used (which should be the same discount used for learning).
adder = adders.NStepTransitionAdder(
    priority_fns={replay_table_name: lambda x: 1.},
    client=reverb.Client(replay_server_address),
    n_step=5,
    discount=0.99)

# This connects to the created reverb server; also note that we use a transition
# adder above so we'll tell the dataset function that so that it knows the type
# of data that's coming out.
dataset = datasets.make_reverb_dataset(
    table=replay_table_name,
    server_address=replay_server_address,
    batch_size=256,
    prefetch_size=True)

# Make sure observation network is a Sonnet Module.
observation_network = tf2_utils.batch_concat
observation_network = tf2_utils.to_sonnet_module(observation_network)

# Create the target networks
target_policy_network = copy.deepcopy(policy_network)
target_critic_network = copy.deepcopy(critic_network)
target_observation_network = copy.deepcopy(observation_network)

# Get observation and action specs.
act_spec = environment_spec.actions
obs_spec = environment_spec.observations
emb_spec = tf2_utils.create_variables(observation_network, [obs_spec])

# Create the behavior policy.
behavior_network = snt.Sequential([
    observation_network,
    policy_network,
    networks.ClippedGaussian(0.3), #sigma = 0.3
    networks.ClipToSpec(act_spec),
])

# We must create the variables in the networks before passing them to learner.
# Create variables.
tf2_utils.create_variables(policy_network, [emb_spec])
tf2_utils.create_variables(critic_network, [emb_spec, act_spec])
tf2_utils.create_variables(target_policy_network, [emb_spec])
tf2_utils.create_variables(target_critic_network, [emb_spec, act_spec])
tf2_utils.create_variables(target_observation_network, [obs_spec])

actor = actors.FeedForwardActor(behavior_network, adder=adder)

learner = d4pg.D4PGLearner(policy_network=policy_network,
                           critic_network=critic_network,
                           observation_network=observation_network,
                           target_policy_network=target_policy_network,
                           target_critic_network=target_critic_network,
                           target_observation_network=target_observation_network,
                           dataset=dataset,
                           discount=0.99,
                           clipping=True,
                           target_update_period=100,
                           policy_optimizer=snt.optimizers.Adam(1e-4),
                           critic_optimizer=snt.optimizers.Adam(1e-4),
                           # Log learner updates to console every 10 seconds.
                           logger=loggers.TerminalLogger(time_delta=10.),
                           checkpoint=False)


def _calculate_num_learner_steps(num_observations,min_observations,observations_per_step):
    """Calculates the number of learner steps to do at step=num_observations."""
    n = num_observations - min_observations
    if n < 0:
        # Do not do any learner steps until you have seen min_observations.
        return 0
    if observations_per_step > 1:
        # One batch every 1/obs_per_step observations, otherwise zero.
        return int(n % int(observations_per_step) == 0)
    else:
        # Always return 1/obs_per_step batches every observation.
        return int(1 / observations_per_step)

samples_per_insert = 32.0
observations_per_step = 256 / samples_per_insert # batch size / samples per insert
num_training_episodes =  10 # @param {type: "integer"}
min_actor_steps_before_learning = 1000  # @param {type: "integer"}
num_actor_steps_per_iteration =   100 # @param {type: "integer"}
num_learner_steps_per_iteration = 1  # @param {type: "integer"}

min_steps_taken = 256 # batch size

learner_steps_taken = 0
actor_steps_taken = 0
returns = []
for episode in range(num_training_episodes):
  
    timestep = environment.reset()
    actor.observe_first(timestep)
    episode_return = 0

    # Run Episode
    while not timestep.last():
        # Get an action from the agent and step in the environment.
        action = actor.select_action(timestep.observation)
        next_timestep = environment.step(action)

        # Record the transition.
        actor.observe(action=action, next_timestep=next_timestep)

        # Book-keeping.
        episode_return += next_timestep.reward
        actor_steps_taken += 1
        timestep = next_timestep

        # update
        num_learner_steps = _calculate_num_learner_steps(actor_steps_taken, 1000, observations_per_step)
        for _ in range(num_learner_steps):
            learner.step()
            learner_steps_taken += 1
        
        if num_learner_steps > 0:
            actor.update()

        
        # See if we have some learning to do.
        # if (actor_steps_taken >= min_actor_steps_before_learning and
        #         actor_steps_taken % num_actor_steps_per_iteration == 0):
        #     # Learn.
        #     for learner_step in range(num_learner_steps_per_iteration):
        #         learner.step()
        #     learner_steps_taken += num_learner_steps_per_iteration

    # Log quantities.
    print('Episode: %d | Return: %f | Learner steps: %d | Actor steps: %d'%(
            episode, episode_return, learner_steps_taken, actor_steps_taken))
    returns.add(episode_return)

plt.plot(returns)
plt.show()

@tf.function(input_signature=[tf.TensorSpec(shape=(1,32), dtype=np.float32)])
def policy_inference(x):
	return policy_network(x)

p_save = snt.Module()
p_save.inference = policy_inference
p_save.all_variables = list(policy_network.variables) 
tf.saved_model.save(p_save, "p2_save")

environment = TestRSEnv()
environment = wrappers.GymWrapper(environment)
environment = wrappers.SinglePrecisionWrapper(environment)

timestep = environment.reset()
while not timestep.last():
    # Simple environment loop.
    action = actor.select_action(timestep.observation)
    timestep = environment.step(action)
    environment.render()

environment.close()
