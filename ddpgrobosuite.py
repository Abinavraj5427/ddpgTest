# import gym
import robosuite as suite
import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


# Could have also just used a Gaussian Distribution
class OUActionNoise:
    def __init__(self, mean, std_deviation, theta=0.15, dt=1e-2, x_initial=None):
        self.theta = theta
        self.mean = mean
        self.std_dev = std_deviation
        self.dt = dt
        self.x_initial = x_initial
        self.reset()

    def __call__(self):
        # Formula taken from https://www.wikipedia.org/wiki/Ornstein-Uhlenbeck_process.
        x = (
            self.x_prev
            + self.theta * (self.mean - self.x_prev) * self.dt
            + self.std_dev * np.sqrt(self.dt) * np.random.normal(size=self.mean.shape)
        )
        # Store x into x_prev
        # Makes next noise dependent on current one
        self.x_prev = x
        return x

    def reset(self):
        if self.x_initial is not None:
            self.x_prev = self.x_initial
        else:
            self.x_prev = np.zeros_like(self.mean)

class Buffer:
    def __init__(self, num_states, num_actions, buffer_capacity=100000, batch_size=64):
        # Number of "experiences" to store at max
        self.buffer_capacity = buffer_capacity
        # Num of tuples to train on.
        self.batch_size = batch_size

        # Its tells us num of times record() was called.
        self.buffer_counter = 0

        # Instead of list of tuples as the exp.replay concept go
        # We use different np.arrays for each tuple element
        self.state_buffer = np.zeros((self.buffer_capacity, num_states))
        self.action_buffer = np.zeros((self.buffer_capacity, num_actions))
        self.reward_buffer = np.zeros((self.buffer_capacity, 1))
        self.next_state_buffer = np.zeros((self.buffer_capacity, num_states))

    # Takes (s,a,r,s') obervation tuple as input
    def record(self, obs_tuple):
        # Set index to zero if buffer_capacity is exceeded,
        # replacing old records
        index = self.buffer_counter % self.buffer_capacity

        self.state_buffer[index] = obs_tuple[0]
        self.action_buffer[index] = obs_tuple[1]
        self.reward_buffer[index] = obs_tuple[2]
        self.next_state_buffer[index] = obs_tuple[3]

        self.buffer_counter += 1

    # We compute the loss and update parameters
    def sample(self):
        # Get sampling range
        record_range = min(self.buffer_counter, self.buffer_capacity)
        # Randomly sample indices
        batch_indices = np.random.choice(record_range, self.batch_size)
        # Convert to tensors
        state_batch = tf.convert_to_tensor(self.state_buffer[batch_indices])
        action_batch = tf.convert_to_tensor(self.action_buffer[batch_indices])

        reward_batch = tf.convert_to_tensor(self.reward_buffer[batch_indices])
        reward_batch = tf.cast(reward_batch, dtype=tf.float32)

        next_state_batch = tf.convert_to_tensor(self.next_state_buffer[batch_indices])
        return state_batch, action_batch, reward_batch, next_state_batch

# This update target parameters slowly
# Based on rate `tau`, which is much less than one.
@tf.function
def update_target(target_weights, weights, tau):
    for (a, b) in zip(target_weights, weights):
        a.assign(b * tau + a * (1 - tau))


# Training hyperparameter
def ddpg(
    env_fn,
    test_env_fn,
    std_dev = 0.2, 
    critic_lr = 0.002, 
    actor_lr = 0.001, 
    total_episodes = 100, 
    gamma = 0.99, 
    tau = 0.005, 
    batch_size = 64, 
    memory_cap = 50000,
    start_steps = 50):
    np.random.seed(0)
    
    env = env_fn()

    num_states = np.squeeze(env.observation_spec()['robot0_robot-state'].shape)
    print("Size of State Space ->  {}".format(num_states))
    num_actions = env.action_dim
    print("Size of Action Space ->  {}".format(num_actions))

    lower_bound, upper_bound = env.action_spec

    print("Max Value of Action ->  {}".format(upper_bound))
    print("Min Value of Action ->  {}".format(lower_bound))
    
    ou_noise = OUActionNoise(mean=np.zeros(num_actions), std_deviation=float(std_dev) * np.ones(num_actions))
    
    """
    Here we define the Actor and Critic networks. These are basic Dense models
    with `ReLU` activation.
    Note: We need the initialization for last layer of the Actor to be between
    `-0.003` and `0.003` as this prevents us from getting `1` or `-1` output values in
    the initial stages, which would squash our gradients to zero,
    as we use the `tanh` activation.
    """
    def get_actor():
        # Initialize weights between -3e-3 and 3-e3
        last_init = tf.random_uniform_initializer(minval=-0.003, maxval=0.003)

        inputs = layers.Input(shape=(num_states,))
        out = layers.Dense(256, activation="relu")(inputs)
        out = layers.Dense(256, activation="relu")(out)
        outputs = layers.Dense(num_actions, activation="tanh", kernel_initializer=last_init)(out)

        # Output controls
        center = (upper_bound + lower_bound) / 2
        multiplier = (upper_bound - lower_bound) / 2
        outputs = outputs * multiplier 
        model = tf.keras.Model(inputs, outputs)
        return model


    def get_critic():
        # State as input
        state_input = layers.Input(shape=(num_states,))
        state_out = layers.Dense(32, activation="relu")(state_input)
        state_out = layers.Dense(32, activation="relu")(state_out)

        # Action as input
        action_input = layers.Input(shape=(num_actions))
        action_out = layers.Dense(32, activation="relu")(action_input)

        # Both are passed through seperate layer before concatenating
        concat = layers.Concatenate()([state_out, action_out])

        out = layers.Dense(256, activation="relu")(concat)
        out = layers.Dense(256, activation="relu")(out)
        outputs = layers.Dense(1)(out)

        # Outputs single value for give state-action
        model = tf.keras.Model([state_input, action_input], outputs)

        return model

    actor_model = get_actor()
    critic_model = get_critic()

    target_actor = get_actor()
    target_critic = get_critic()

    # Making the weights equal initially
    target_actor.set_weights(actor_model.get_weights())
    target_critic.set_weights(critic_model.get_weights())

    critic_optimizer = tf.keras.optimizers.Adam(critic_lr)
    actor_optimizer = tf.keras.optimizers.Adam(actor_lr)


    buffer = Buffer(num_states, num_actions, memory_cap, batch_size)

    # To store reward history of each episode
    ep_reward_list = []
    # To store average reward history of last few episodes
    avg_reward_list = []

    # Takes about 4 min to train
    for ep in range(total_episodes):

        prev_state = env.reset()
        episodic_reward = 0
        total_steps = 0
        while True:
            # Uncomment this to see the Actor in action
            # But not in a python notebook.
            # env.render()
            total_steps = total_steps + 1
            prev_state = prev_state['robot0_robot-state']
            tf_prev_state = tf.expand_dims(tf.convert_to_tensor(prev_state), 0)

            def policy(state, noise_object=None):
                sampled_actions = np.squeeze(actor_model(state).numpy())
                
                # noise = 0
                # if(noise_object != None):
                #     noise = noise_object()
                noise = 0.2 * np.random.randn(num_actions)
                
                
                sampled_actions = sampled_actions + noise
                
                # Adding noise to action
                # sampled_actions = sampled_actions.numpy() + noise

                # We make sure action is within bounds
                legal_action = np.clip(sampled_actions,lower_bound,upper_bound)
                return legal_action

            
            if ep > start_steps:
                action = policy(tf_prev_state, ou_noise)
            else:
                action = np.random.randn(env.robots[0].dof) 
            

            # Recieve state and reward from environment.
            state, reward, done, info = env.step(action)

            next_state = state['robot0_robot-state']
            buffer.record((prev_state, action, reward, next_state))
            episodic_reward += reward

            if done:
                break

            prev_state = state

        for _ in range(total_steps):

            state_batch, action_batch, reward_batch, next_state_batch = buffer.sample()

            def update():
                # Training and updating Actor & Critic networks.
                # See Pseudo Code.
                with tf.GradientTape() as tape:
                    target_actions = target_actor(next_state_batch, training=True)
                    y = reward_batch + gamma * target_critic(
                        [next_state_batch, target_actions], training=True
                    )
                    critic_value = critic_model([state_batch, action_batch], training=True)
                    critic_loss = tf.math.reduce_mean(tf.math.square(y - critic_value))
                critic_grad = tape.gradient(critic_loss, critic_model.trainable_variables)
                critic_optimizer.apply_gradients(
                    zip(critic_grad, critic_model.trainable_variables)
                )
                with tf.GradientTape() as tape:
                    actions = actor_model(state_batch, training=True)
                    critic_value = critic_model([state_batch, actions], training=True)
                    # Used `-value` as we want to maximize the value given
                    # by the critic for our actions
                    actor_loss = -tf.math.reduce_mean(critic_value)
                actor_grad = tape.gradient(actor_loss, actor_model.trainable_variables)
                actor_optimizer.apply_gradients(
                    zip(actor_grad, actor_model.trainable_variables)
                )

            update()
        
        update_target(target_actor.variables, actor_model.variables, tau)
        update_target(target_critic.variables, critic_model.variables, tau)

        ep_reward_list.append(episodic_reward)

        # Mean of last 40 episodes
        avg_reward = np.mean(ep_reward_list[-40:])
        print("Episode * {} * Avg Reward is ==> {} Episode Reward is ==> {} ep len: {}".format(ep, avg_reward, episodic_reward, total_steps))
        avg_reward_list.append(avg_reward)

        test_returns = []
        def test_agent(num_episodes=5):
            test_env = test_env_fn()
            # n_steps = 0
            for j in range(num_episodes):
                s, episode_return, episode_length, d = test_env.reset(), 0, 0, False
                while not (d):
                    # Take deterministic actions at test time (noise_scale = 0)
                    s = s['robot0_robot-state']
                    s = tf.expand_dims(tf.convert_to_tensor(s), 0)
                    a = policy(s)
                    s, r, d, _ = test_env.step(a)
                    test_env.render()
                    episode_return += r
                    episode_length += 1
                print('test return:', episode_return, 'episode_length:', episode_length)
                test_returns.append(episode_return)
            test_env.close()

        if ep > 0 and ep % 15 == 0:
            test_agent()

    # Plotting graph
    # Episodes versus Avg. Rewards
    plt.plot(avg_reward_list)
    print(avg_reward_list)
    plt.xlabel("Episode")
    plt.ylabel("Avg. Epsiodic Reward")
    plt.show()
    env.close()
    
if __name__ == '__main__':
    problem = "Pendulum-v0"
    ddpg(env_fn=lambda: suite.make(
        env_name="Lift",
        robots="Sawyer",
        gripper_types="default",
        has_renderer=False,
        has_offscreen_renderer=False,
        horizon=600,
        reward_scale=100.0,
        reward_shaping=True,
	    use_camera_obs=False
    ),
    test_env_fn=lambda: suite.make(
        env_name="Lift",
        robots="Sawyer",
        gripper_types="default",
        has_renderer=True,
        horizon=600,
        has_offscreen_renderer=False,
        reward_shaping=True,
        reward_scale=100.0,
	    use_camera_obs=False
    ), total_episodes=5)
