%tensorflow_version 1.x
import tensorflow as tf
import gym
import numpy as np
import os

# Simple feed forward neural network
def ANN(x, layer_sizes, hidden_activation=tf.nn.relu, output_activation=None):
    for h in layer_sizes[:-1]:
        x = tf.layers.dense(x, units=h, activation=hidden_activation)
    return tf.layers.dense(x, units=layer_sizes[-1], activation=output_activation)

def CreateNetworks(
        s, a,
        num_actions,
        action_max,
        hidden_sizes=(300,),
        hidden_activation=tf.nn.relu,
        output_activation=tf.tanh):

    with tf.variable_scope('mu'):
        mu = action_max * ANN(s, list(hidden_sizes)+[num_actions], hidden_activation, output_activation)
    with tf.variable_scope('q'):
        input_ = tf.concat([s, a], axis=-1) # (state, action)
        q = tf.squeeze(ANN(input_, list(hidden_sizes)+[1], hidden_activation, None), axis=1)
    with tf.variable_scope('q', reuse=True):
        # reuse is True, so it reuses the weights from the previously defined Q network
        input_ = tf.concat([s, mu], axis=-1) # (state, mu(state))
        q_mu = tf.squeeze(ANN(input_, list(hidden_sizes)+[1], hidden_activation, None), axis=1)
    return mu, q, q_mu

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
# get all variables within a scope
def get_vars(scope):
    return [x for x in tf.global_variables() if scope in x.name]
### Implement the DDPG algorithm ###
def ddpg(
            env_fn,
            ac_kwargs=dict(),
            seed=0,
            save_folder=None,
            num_train_episodes=100,
            test_agent_every=25,
            replay_size = int(1e6),
            gamma=0.99,
            decay=0.995,
            mu_lr=1e-3,
            q_lr=1e-3,
            batch_size=100,
            start_steps=10000,
            action_noise=0.1,
            max_episode_length=1000
        ):
    tf.set_random_seed(seed)
    np.random.seed(seed)

    env, test_env = env_fn(), env_fn()

    #comment out this line if you don't want to record a video of the agent
    # if save_folder is not None:
    #     test_env = gym.wrappers.Monitor(test_env, save_folder)

    # get size of state space and action space
    num_states = env.observation_space.shape[0]
    num_actions = env.action_space.shape[0]

    # Maximum value of action
    # Assumes both low and high values are the same
    # Assumes all actions have the same bounds
    # May NOT be the case for all environments
    action_max = env.action_space.high[0]

    # Create Tensorflow placeholders (neural network inputs)
    X = tf.placeholder(dtype=tf.float32, shape=(None, num_states))
    A = tf.placeholder(dtype=tf.float32, shape=(None, num_actions))
    X2 = tf.placeholder(dtype=tf.float32, shape=(None, num_states))
    R = tf.placeholder(dtype=tf.float32, shape=(None,))
    D = tf.placeholder(dtype=tf.float32, shape=(None,))

    # Main network outputs
    with tf.variable_scope('main'):
        mu, q, q_mu = CreateNetworks(X, A, num_actions, action_max, **ac_kwargs)

    # Target networks
    with tf.variable_scope('target'):
        mu_targ, q_targ, q_mu_targ = CreateNetworks(X2, A, num_actions, action_max, **ac_kwargs)

    # s, episode_return, episode_length, d = env.reset(), 0, 0, False
    # a = env.action_space.sample()
    # print(s, a)
    # s2, r, d, _ = env.step(a)
    # print(s2, r, d)
    # return

    s = np.array([ 0.23164732,  0.97279984, -0.74811356]) 
    a = np.array([1.849152])
    s2 = [0.21903736, 0.97571647, 0.25885912] 
    r = -1.8470242261833913 
    d = False
    np.set_printoptions(threshold=np.inf)
    
    replay_buffer = ReplayBuffer(obs_dim=num_states, act_dim=num_actions, size=replay_size)
    replay_buffer.store(s, a, r, s2, d)
    batch = replay_buffer.sample_batch(batch_size)

    # Copy main network params to target networks
    target_init = tf.group(
        [tf.assign(v_targ, v_main)
            for v_main, v_targ in zip(get_vars('main'), get_vars('target'))
        ]
    )
    
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    sess.run(target_init)
    # print(get_vars("main/q/kernal:0"))
    
    # q_list = get_vars('main/q')
    # for i in range(len(q_list)):
    #   np.save("q"+str(i)+".npy", q_list[i].eval(session = sess))

    # mu_list = get_vars('main/mu')
    # for i in range(len(mu_list)):
    #   np.save("mu"+str(i)+".npy", mu_list[i].eval(session = sess))

    # q_list = get_vars('target/q')
    # for i in range(len(q_list)):
    #   np.save("q"+str(i)+".npy", q_list[i].eval(session = sess))

    # mu_list = get_vars('target/mu')
    # for i in range(len(mu_list)):
    #   np.save("mu"+str(i)+".npy", mu_list[i].eval(session = sess))
    
def load_network(sess, model_name):
    w1 = np.load(model_name+"0.npy")
    b1 = np.load(model_name+"1.npy")
    w2 = np.load(model_name+"2.npy")
    b2 = np.load(model_name+"3.npy")
     
    w1_ = sess.graph.get_tensor_by_name('main/'+ model_name+'/kernal:0')
    b1_ = sess.graph.get_tensor_by_name('main/'+ model_name+'/bias:0')
    w2_ = sess.graph.get_tensor_by_name('main/'+ model_name+'/kernal:1')
    b2_ = sess.graph.get_tensor_by_name('main/'+ model_name+'/bias:1')

    sess.run(tf.assign(w1_, w1))
    sess.run(tf.assign(b1_, b1))
    sess.run(tf.assign(w2_, w2))
    sess.run(tf.assign(b2_, b2))

    
if __name__ == '__main__':
    # import argparse
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--env', type = str, default='Pendulum-v0')
    # parser.add_argument('--hidden_layer_sizes', type=int, default=300)
    # parser.add_argument('--num_layers', type=int, default=1)
    # parser.add_argument('--gamma', type=float, default=0.99)
    # parser.add_argument('--seed', type=int, default=0)
    # parser.add_argument('--num_train_episodes', type=int, default=200)
    # parser.add_argument('--save_folder', type=str, default='ddpg_monitor')
    # args = parser.parse_args()
    ddpg(
        lambda: gym.make('Pendulum-v0'),
        ac_kwargs = dict(hidden_sizes=[300]*1),
        gamma = 0.99,
        seed = 0,
        save_folder = None,
        num_train_episodes = 200,
    )