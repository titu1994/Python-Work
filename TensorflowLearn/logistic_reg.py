import numpy as np

np.random.seed(1)
from sklearn.model_selection import train_test_split
from scipy.signal import sawtooth
import matplotlib.pyplot as plt

import tensorflow as tf

nb_samples = 640
nb_timesteps = 512
nb_epochs = 1000
reg_lambda = 0.5


def sin_wave():
    nb = np.random.randint(1, 100, size=1)[0]
    shift = np.random.randint(0, 91, size=1)[0]

    x = np.arange(-nb * np.pi, nb * np.pi, step=(2 * nb * np.pi / nb_timesteps))
    y = np.sin(x + (shift / 180.))

    noise = np.random.uniform(-0.1, 0.1, size=len(x))
    y += noise

    return y


def triangle_wave():
    nb = np.random.randint(1, 100, size=1)[0]
    shift = np.random.randint(0, 91, size=1)[0]

    x = np.arange(-nb * np.pi, nb * np.pi, step=(2 * nb * np.pi / nb_timesteps))
    y = sawtooth(x + (shift / 180.), width=0.5)

    noise = np.random.uniform(-0.1, 0.1, size=len(x))
    y += noise

    return y


# x, y = sin_wave()
# x, y = triangle_wave()

X = np.zeros((nb_samples, nb_timesteps), dtype='float32')
y = np.zeros((nb_samples, 2), dtype='float32')

for i in range(nb_samples // 2):
    X[i] = sin_wave()
    y[i, 0] = 1.

for i in range(nb_samples // 2):
    X[i + (nb_samples // 2)] = triangle_wave()
    y[i + (nb_samples // 2), 1] = 1.

# for i in range(3):
#     idx = np.random.randint(0, 512, size=1)[0]
#     plt.plot(X[idx])
# plt.show()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0, shuffle=True)

X = tf.placeholder(tf.float32, shape=(None, nb_timesteps), name='input')
Y = tf.placeholder(tf.float32, shape=(None, 2), name='label')

W = tf.Variable(np.random.normal(0.0, 1.0, size=(nb_timesteps, 2)), name='weights', dtype=tf.float32)
b = tf.Variable(np.zeros((2,), dtype=np.float32))

y_pred = tf.matmul(X, W) + b

accuracy = tf.reduce_mean(tf.cast(tf.equal(
    tf.argmax(Y, axis=-1),
    tf.argmax(tf.nn.softmax(y_pred), axis=-1)), tf.float32))

loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=Y, logits=y_pred, name='loss'))
loss += reg_lambda * tf.nn.l2_loss(W)

optimizer = tf.train.GradientDescentOptimizer(0.01)
train_op = optimizer.minimize(loss)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    train_accs = []
    test_accs = []

    for epoch in range(nb_epochs):
        feed_dict = {
            X: X_train,
            Y: y_train,
        }

        _, acc, l = sess.run([train_op, accuracy, loss],
                             feed_dict=feed_dict)

        test_acc = sess.run(accuracy,
                       feed_dict={
                           X: X_test,
                           Y: y_test,
                       })

        train_accs.append(acc)
        test_accs.append(test_acc)

        if epoch % 50 == 0:
            print("%d: Train accuracy : " % (epoch), acc, "Loss : ", l)
            print("%d: Test accuracy : " % (epoch), test_acc)
            print()

    acc = sess.run(accuracy,
                   feed_dict={
                       X: X_test,
                       Y: y_test,
                   })

    print()
    print("%d: Final Accuracy: " % nb_epochs, acc, "Best test score : ", max(test_accs))

    plt.plot(train_accs, label='train')
    plt.plot(test_accs, label='test')
    plt.legend()
    plt.show()
