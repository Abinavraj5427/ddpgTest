import numpy as np
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
import gym
import matplotlib.pyplot as plt

class ANN(Model):
    def __init__(self, layer_sizes, hidden_activation = 'relu', output_activation = None, action_max=1):
        super(ANN, self).__init__()
        self.total_layers = len(layer_sizes)
        self.action_max= action_max
        self.model_layers = []
        for h in layer_sizes[:-1]:
            self.model_layers.append(Dense(h, activation = hidden_activation))
        self.model_layers.append(Dense(layer_sizes[-1], activation = output_activation))
        
    def call(self, x):
        for layer in self.model_layers:
            x = layer(x)
        x *= self.action_max
        return x

    # def get_weights(self):
    #     weights = []
    #     for layer in self.layers:
    #         weights.append(layer.get_weights())
    #     return weights

    # def set_weights(self, new_weights):
    #     it = iter(new_weights)
    #     ctr = 0
    #     for w, b in zip(it, it):
    #         self.layers[ctr].set_weights([w, b])
    #         ctr += 1         

def create_networks(
        num_actions, 
        action_max, 
        hidden_sizes=(300,), 
        hidden_activation='relu',
        output_activation='tanh'):
    
    # Used for getting the optimized action
    mu_model = ANN(layer_sizes = list(hidden_sizes)+[num_actions], hidden_activation=hidden_activation, output_activation=output_activation, action_max=action_max )

    # Used for optimizing the prediction of Q-value for any action
    q_model = ANN(layer_sizes = list(hidden_sizes) + [1], hidden_activation=hidden_activation, output_activation=None)

    return mu_model, q_model

### The experience replay memory ###
class ReplayBuffer:
    def __init__(self, obs_dim, act_dim, size):
        self.obs1_buf = np.zeros([size, obs_dim], dtype=np.float32)
        self.obs2_buf = np.zeros([size, obs_dim], dtype=np.float32)
        self.acts_buf = np.zeros([size, act_dim], dtype=np.float32)
        self.rews_buf = np.zeros(size, dtype=np.float32)
        self.done_buf = np.zeros(size, dtype=np.float32)
        self.ptr, self.size, self.max_size = 0, 0, size

    def store(self, obs, act, rew, next_obs, done):
        self.obs1_buf[self.ptr] = obs
        self.obs2_buf[self.ptr] = next_obs
        self.acts_buf[self.ptr] = act
        self.rews_buf[self.ptr] = rew
        self.done_buf[self.ptr] = done
        self.ptr = (self.ptr+1) % self.max_size
        self.size = min(self.size+1, self.max_size)

    def sample_batch(self, batch_size=32):
        idxs = np.random.randint(0, self.size, size=batch_size)
        return dict(
            s=self.obs1_buf[idxs],
            s2=self.obs2_buf[idxs],
            a=self.acts_buf[idxs],
            r=self.rews_buf[idxs],
            d=self.done_buf[idxs]
        )

### implement the ddpg algorithm ###
def ddpg(
    env_fn,
    ac_kwargs = dict(),
    seed = 0,
    num_train_episodes = 100,
    test_agent_every = 25,
    replay_size = int(1e6),
    gamma = 0.99,
    decay = 0.95,
    mu_lr = 1e-3,
    q_lr = 1e-3,
    batch_size = 100,
    start_steps = 10000,
    action_noise = 0.1,
    max_episode_length = 1000):

    # randomness
    tf.random.set_seed(seed)
    np.random.seed(seed)

    env, test_env = env_fn(), env_fn()

    # state and action spaces
    num_states = env.observation_space.shape[0]
    num_actions = env.action_space.shape[0]

    # maximum possible action value for each dimension
    action_max = env.action_space.high[0]

    # replay buffer
    replay_buffer = ReplayBuffer(obs_dim=num_states, act_dim=num_actions, size=replay_size)

    mu, q = create_networks(num_actions, action_max=action_max, **ac_kwargs)
    mu_targ, q_targ = create_networks(num_actions, action_max=action_max, **ac_kwargs)

    ### NOTE: Copy target weights for init
    q_targ.set_weights(q.get_weights()) 
    mu_targ.set_weights(mu.get_weights()) 

    def get_action(s, noise_scale):
        a = mu(s)
        a += noise_scale * np.random.randn(num_actions) # For exploration
        return np.clip(a, -action_max, action_max)

    test_returns = []
    def test_agent(num_episodes=5):
        # n_steps = 0
        for j in range(num_episodes):
            s, episode_return, episode_length, d = test_env.reset(), 0, 0, False
            while not (d or (episode_length == max_episode_length)):
                # Take deterministic actions at test time (noise_scale = 0)
                test_env.render()
                a = np.squeeze(get_action(np.expand_dims(s, axis=0), 0), axis = 0)
                s, r, d, _ = test_env.step(a)
                episode_return += r
                episode_length += 1
                # n_steps += 1
            print('test return:', episode_return, 'episode_length:', episode_length)
            test_returns.append(episode_return)

    # Main loop: play episode and train
    returns = []
    q_losses = []
    mu_losses = []
    num_steps = 0

    mu_optimizer = Adam(learning_rate = mu_lr)
    q_optimizer = Adam(learning_rate= q_lr)
    for i_episode in range(num_train_episodes):
        # reset dev
        s, episode_return, episode_length, d = env.reset(), 0, 0, False
        while not (d or (episode_length == max_episode_length)):

            # For the first `start_steps` steps, use randomly sampled actions
            # in order to encourage exploration

            if num_steps == start_steps:
                print("USING AGENT ACTIONS NOW")

            if num_steps > start_steps:
                a = np.squeeze(get_action(np.expand_dims(s, axis=0), action_noise), axis=0)
            else:
                a = env.action_space.sample()

            # Keep track of the number of steps done
            num_steps += 1
            
            # Step the env
            s2, r, d, _ = env.step(a)
            episode_return += r
            episode_length += 1
            
            d_store = False if episode_length == max_episode_length else d
            replay_buffer.store(s, a, r, s2, d_store)
            s = s2
        
        for i in range(episode_length):
            batch = replay_buffer.sample_batch(batch_size)
            
            s, a, r, s2, d = batch['s'], batch['a'], batch['r'], batch['s2'], batch['d']
            s = tf.convert_to_tensor(s)
            a = tf.convert_to_tensor(a)
            with tf.GradientTape() as tape:
                # Q-loss
                # q_targ_input = tf.concat([s2, mu_targ(s2, training=True)], axis=-1)
                # q_target = r + gamma * q_targ(q_targ_input, training=True)
                # q_input = tf.concat([s, a], axis=-1)
                # q_loss = tf.math.reduce_mean(tf.math.square(q(q_input, training=True) - q_target))

                # q_losses.append(q_loss)

            # # NOTE: Update Main Q Network
            # q_grads = tape.gradient(q_loss, q.trainable_variables)
            # q_optimizer.apply_gradients(zip(q_grads, q.trainable_variables))
                target_actions = mu_targ(s2, training=True)
                y = r + gamma * q_targ(
                    tf.concat([s, target_actions], axis=-1), training=True
                )
                critic_value = q(tf.concat([s, a], axis=-1), training=True)
                critic_loss = tf.math.reduce_mean(tf.math.square(y - critic_value))
            critic_grad = tape.gradient(critic_loss, q.trainable_variables)
            q_optimizer.apply_gradients(
                zip(critic_grad, q.trainable_variables)
            )

            with tf.GradientTape() as tape:
            #     # Mu-loss
            #     q_input = tf.concat([s, mu(s, training=True)], axis=-1)
            #     mu_loss = -tf.math.reduce_mean(q(q_input, training=True))

            #     mu_losses.append(mu_loss)

            # # NOTE: Update Main mu Network
            # mu_grads = tape.gradient(mu_loss, mu.trainable_variables)
            # mu_optimizer.apply_gradients(zip(mu_grads, mu.trainable_variables))

                actions = mu(s, training=True)
                critic_value = q(tf.concat([s, actions], axis = -1), training=True)
                # Used `-value` as we want to maximize the value given
                # by the critic for our actions
                actor_loss = -tf.math.reduce_mean(critic_value)
            actor_grad = tape.gradient(actor_loss, mu.trainable_variables)
            mu_optimizer.apply_gradients(
                zip(actor_grad, mu.trainable_variables)
                )

            # def q_loss():
            #     # Q-loss
            #     q_targ_input = tf.concat([s2, mu_targ(s2)], axis=-1)
            #     q_target = r + gamma * (1 - d) * q_targ(q_targ_input)
            #     q_target = tf.stop_gradient(q_target)
            #     q_input = tf.concat([s, a], axis=-1)
            #     q_loss = tf.math.reduce_mean((q(q_input) - q_target)**2)
            #     q_losses.append(q_loss)
            #     return q_loss
            
            # def mu_loss():
            #     # Mu-loss
            #     q_input = tf.concat([s, mu(s)], axis=-1)
            #     mu_loss = -tf.math.reduce_mean(q(q_input))
            #     mu_losses.append(mu_loss)
            #     return mu_loss

            # q_optimizer.minimize(q_loss, var_list = q.trainable_variables)
            # mu_optimizer.minimize(mu_loss, var_list = mu.trainable_variables)
            
            # NOTE: Update target gradients using weighted average

            def update_target(target_weights, weights, tau):
                for (a, b) in zip(target_weights, weights):
                    a.assign(b * tau + a * (1 - tau))

            update_target(q_targ.variables, q.variables, 1-decay)
            update_target(mu_targ.variables, mu.variables, 1-decay)

            # q_targ_w, q_w = np.array(q_targ.get_weights()), np.array(q.get_weights())
            # new_q_targ_w = (q_targ_w*decay + (1-decay)*q_w).tolist()
            # q_targ.set_weights(new_q_targ_w)

            # mu_targ_w, mu_w = np.array(mu_targ.get_weights()), np.array(mu.get_weights())
            # new_mu_targ_w = (mu_targ_w*decay + (1-decay)*mu_w).tolist()
            # mu_targ.set_weights(new_mu_targ_w)

        print("Episode:", i_episode+1, "Return:", episode_return, 'episode_length:', episode_length)
        returns.append(episode_return)
        # Test the agent
        if i_episode > 0 and i_episode % test_agent_every == 0:
            test_agent()
        
    # plt.plot(returns)
    plt.plot(smooth(np.array(returns)))
    plt.title("Train returns")
    plt.show()

    plt.plot(smooth(np.array(test_returns)))
    plt.title("Test returns")
    plt.show()

    plt.plot(q_losses)
    plt.title('q losses')
    plt.show()

    plt.plot(mu_losses)
    plt.title('mu losses')
    plt.show()

# moving average of the last 100 datapoints
def smooth(x):
    n = len(x)
    y = np.zeros(n)
    for i in range(n):
        start = max(0, i-99)
        y[i] = float(x[start:i].sum()/(i-start+1))
    return y

if __name__ == '__main__':
    env = 'Pendulum-v0'
    hidden_layer_sizes = 300
    num_layers = 1
    gamma = 0.99
    seed = 0
    num_train_episodes = 100
    
    ddpg(
        lambda: gym.make('Pendulum-v0'),
        ac_kwargs = dict(hidden_sizes=[hidden_layer_sizes]*num_layers),
        gamma = gamma,
        seed = seed,
        num_train_episodes = num_train_episodes,
        start_steps = 4000
    )