import tensorflow as tf
import numpy as np
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
import gym

def create_networks(
        num_actions, 
        action_max, 
        hidden_sizes=(300,), 
        hidden_activation='relu',
        output_activation='tanh'):
    
    # Used for getting the optimized action
    mu_model = ANN("mu", layer_sizes = list(hidden_sizes)+[num_actions], hidden_activation=hidden_activation, output_activation=output_activation, action_max=action_max )

    # Used for optimizing the prediction of Q-value for any action
    q_model = ANN("q", layer_sizes = list(hidden_sizes) + [1], hidden_activation=hidden_activation, output_activation=None)

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


class ANN(Model):
    def __init__(self,mame, layer_sizes, hidden_activation = 'relu', output_activation = None, action_max=1):
        super(ANN, self).__init__()
        self.mame=mame
        self.total_layers = len(layer_sizes)
        self.action_max= action_max
        self.model_layers = []
        for h in layer_sizes[:-1]:
            self.model_layers.append(Dense(h, activation = hidden_activation))
        self.model_layers.append(Dense(layer_sizes[-1], activation = output_activation))
        
    def call(self, x):
        print("S", self.mame, x)
        for layer in self.model_layers:
            x = layer(x)
        x *= self.action_max
        print("E", self.mame, x)
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

    mu, q = create_networks(num_actions, action_max, **ac_kwargs)
    mu_targ, q_targ = create_networks(num_actions, action_max, **ac_kwargs)

    ### NOTE: Copy target weights for init
    # q_targ.set_weights(q.get_weights()) 
    # mu_targ.set_weights(mu.get_weights()) 

    s = np.array([ 0.23164732,  0.97279984, -0.74811356]) 
    a = np.array([1.849152])
    s2 = [0.21903736, 0.97571647, 0.25885912] 
    r = -1.8470242261833913 
    d = False

    # s = np.array([-0.6460527,  -0.76329281,  0.60891966])
    # a = np.array([1.5983584])
    # s2 = np.array([-0.63545021, -0.77214185,  0.27620381]) 
    # r = -5.2070620242099315 
    # d = False

    np.set_printoptions(threshold=np.inf)

    s = np.expand_dims(s, axis = 0)
    a = np.expand_dims(a, axis = 0)
    s2 = np.expand_dims(s2, axis = 0)

    # Initializes weights
    print("INITIAL")
    input = tf.concat([s, mu(s)], axis=-1)
    print("INITIAL")
    q(input)
    print("INITIAL")
    input = tf.concat([s2, mu_targ(s2)], axis=-1)
    print("INITIAL")
    q_targ(input)

    load_network(q, "q")
    load_network(mu, "mu")

    
    # mu_value = mu(s)
    
    # input = tf.concat([s, mu_value], axis=-1)
    # q_mu_value = q(input)

    # input = tf.concat([s, a], axis=-1)
    # q_value = q(input)
    # print("Q___", mu_value, q_value, q_mu_value)

    # print(mu_value, q_value, q_mu_value)

    mu_optimizer = Adam(learning_rate = mu_lr, epsilon=1e-08)
    q_optimizer = Adam(learning_rate= q_lr, epsilon=1e-08)

    def q_loss():
        # Q-loss
        print("Mu targ for Q loss")
        q_targ_input = tf.concat([s2, mu_targ(s2)], axis=-1)
        print("Q targ for Q loss")
        q_target = r + gamma * (1 - d) * q_targ(q_targ_input)
        q_input = tf.concat([s, a], axis=-1)
        print("Q for q loss")
        q_loss = tf.math.reduce_mean((q(q_input) - q_target)**2)
        #q_losses.append(q_loss)
        print("QLOSS", q_loss)
        return q_loss
            
    def mu_loss():
        # Mu-loss
        print("Mu for loss mu")
        q_input = tf.concat([s, mu(s)], axis=-1)
        print("Q for Mu loss")
        # print("QQQQ", q_input, q(q_input))
        mu_loss = -tf.math.reduce_mean(q(q_input))
        #mu_losses.append(mu_loss)
        print("MULOSS", mu_loss)
        return mu_loss

    print("SETTING WEIGHTS")
    q_targ.set_weights(q.get_weights()) 
    mu_targ.set_weights(mu.get_weights()) 

    q_optimizer.minimize(q_loss, var_list = q.trainable_variables)

    q_birn = tf.concat([s, a], axis=-1)
    print("Q-new", q(q_birn))
    mu_optimizer.minimize(mu_loss, var_list = mu.trainable_variables)
    # print(q.get_weights())
    # print(mu.get_weights())

def load_network(model, model_name):
    model.set_weights([np.load(model_name+"0.npy"), np.load(model_name+"1.npy"),np.load(model_name+"2.npy"),np.load(model_name+"3.npy")])

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
        start_steps = 2000
    )