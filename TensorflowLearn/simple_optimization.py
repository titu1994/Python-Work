import numpy as np
import tensorflow as tf

x = tf.Variable(initial_value=tf.random_uniform((1,)), dtype=tf.float32)
#y = tf.nn.sigmoid(x) * (1 - tf.nn.sigmoid(x))
#y = (1 - tf.nn.tanh(x) ** 2)
y = tf.tan(x) ** 2 - x - tf.log(tf.abs(x)) / (tf.log(10.) * (x ** 2 - 1))

opt = tf.train.GradientDescentOptimizer(learning_rate=0.1)
#train_op = opt.minimize(y)  # minimize
train_op = opt.minimize(-y)  # maximize

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for i in range(1000):
    for j in range(50):
        _, loss, val = sess.run([train_op, y, x])

    if i % 50 == 0:
        print(i, "Loss : ", loss, "Val : ", val)