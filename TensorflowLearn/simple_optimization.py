import numpy as np
import tensorflow as tf

x = tf.Variable(initial_value=tf.random_uniform((1,)), dtype=tf.float32)
#y = tf.nn.sigmoid(x) * (1 - tf.nn.sigmoid(x))
#y = (1 - tf.nn.tanh(x) ** 2)

opt = tf.train.GradientDescentOptimizer(learning_rate=1e-2)
#train_op = opt.minimize(y)  # minimize
train_op = opt.minimize(-y)  # maximize

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for i in range(1000):
    for j in range(50):
        _, loss, val = sess.run([train_op, y, x])

    if i % 50 == 0:
        print(i, "Loss : ", loss, "Val : ", val)