import numpy as np
import tensorflow as tf
# tf.set_random_seed(0)

x = tf.Variable(tf.random_uniform((1,)), dtype=tf.float32)
a = tf.Variable(tf.random_uniform((1,)), dtype=tf.float32)
b = tf.Variable(tf.random_uniform((1,)), dtype=tf.float32)

y = tf.exp(a * x) + tf.exp(b * x)

#loss = y  # to minimize the function itself
loss = tf.nn.l2_loss(y)  # to minimize a function with variables
# loss += -(1 - x * x)
# loss += -(1 - a * a)
# loss += -(1 - b * b)

global_step = tf.Variable(0, trainable=False)
lr = tf.train.exponential_decay(0.1, global_step, decay_steps=500, decay_rate=0.95, staircase=True)

opt = tf.train.RMSPropOptimizer(lr)
train_op = opt.minimize(loss)  # minimize
#train_op = opt.minimize(-loss)  # maximize

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for i in range(10000):
    for j in range(50):
        _, loss, val_x, val_a, val_b = sess.run([train_op, y, x, a, b])

    if i % 50 == 0:
        print(i, "Y : ", loss, "X : ", val_x, "A : ", val_a, "B : ", val_b)