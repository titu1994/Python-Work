import numpy as np
import tensorflow as tf
# tf.set_random_seed(0)

#x = tf.Variable(tf.random_uniform((1,)), dtype=tf.float32)
x = tf.placeholder(tf.float32, shape=[None])
y = tf.placeholder(tf.float32, shape=[None])

a = tf.Variable(1, dtype=tf.float32)
b = tf.Variable(1, dtype=tf.float32)

y_pred = a * x

#loss = y  # to minimize the function itself
loss = tf.nn.l2_loss(y - y_pred)  # to minimize a function with variables
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

X = []
Y = []

for j in range(-25, 35):
    X.append(j)
    Y.append(j * (9. / 5.) + 32.)

    print(X[-1], Y[-1], 1.85 * j + 1.85 * 18)

print()

X = np.array(X, dtype='float32')
Y = np.array(Y, dtype='float32')

for i in range(10000):
    _, loss_val, val_a, val_b = sess.run([train_op, loss, a, b],
                                  feed_dict={
                                      x: X,
                                      y: Y,
                                  })

    if i % 50 == 0:
        print(i, "loss : ", loss_val, "A : ", val_a, "B : ", val_b)