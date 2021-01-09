import tensorflow as tf
import numpy as np

policy = tf.keras.models.load_model('p_save')
print(policy.inference(np.ones([1, 32])))
