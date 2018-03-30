import numpy as np
import tensorflow as tf
from tensorflow.contrib.opt.python.training.sign_decay import get_cosine_decay_fn
from tensorflow.contrib.opt.python.training.powersign import PowerSignOptimizer

x = tf.Variable(initial_value=tf.random_uniform((1,)), dtype=tf.float32)
#y = tf.nn.sigmoid(x) * (1 - tf.nn.sigmoid(x))
#y = (1 - tf.nn.tanh(x) ** 2)
#y = tf.tan(x) ** 2 - x - tf.log(tf.abs(x)) / (tf.log(10.) * (x ** 2 - 1))
y = tf.nn.sigmoid(x) + tf.nn.tanh(x)

#opt = tf.train.GradientDescentOptimizer(learning_rate=1)
#train_op = opt.minimize(y)  # minimize
#train_op = opt.minimize(-y)  # maximize

global_step = tf.Variable(0, trainable=False, name='global_step')
decay_steps = 1000
cosine_decay = get_cosine_decay_fn(decay_steps)
opt = PowerSignOptimizer(learning_rate=1, sign_decay_fn=cosine_decay)

train_op = opt.minimize(y, global_step)  # minimize
#train_op = opt.minimize(-y, global_step)  # maximize

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for i in range(1000):
    for j in range(50):
        _, loss, val = sess.run([train_op, y, x])

    if i % 50 == 0:
        print(i, "Optimal Value : ", loss, "Val (X) : ", val)