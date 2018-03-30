import tensorflow as tf
import numpy as np

input_dim = 1
x_placeholder = tf.placeholder(tf.float32, shape=[None, input_dim])
y_placeholder = tf.placeholder(tf.float32, shape=[None, 2])

y = tf.constant(2., name='y')

weights = tf.get_variable(name='weights', shape=[input_dim, 1], dtype=tf.float32,
                          initializer=tf.initializers.random_uniform(0.0, 1.0))

bias = tf.get_variable(name='bias', shape=[1,], dtype=tf.float32,
                       initializer=tf.initializers.random_uniform(0.0, 1.0))

energies = tf.matmul(x_placeholder, weights) + bias
#energies = tf.nn.sigmoid(energies)

left = energies - y_placeholder
right = energies - y_placeholder

left = tf.pow(left, y)
right = tf.pow(right, y)

left_b = tf.rank(left)
left_b = tf.range(left_b)

left_b = tf.cast(left_b, tf.float32)
right_b = tf.cast(tf.range(tf.rank(right)), tf.float32)

left2 = left + left_b
right = right + right_b

left3 = left2 / y
right = right / y

loss = tf.abs(tf.losses.mean_squared_error(y_placeholder, left3))

train_op = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(loss,
                var_list=[weights, bias])

writer = tf.summary.FileWriter(logdir='./logs/som/', graph=tf.get_default_graph())

x_values_gen = tf.random_uniform(shape=(100, 1), minval=0.0, maxval=1.0)
y_values_gen = tf.random_uniform(shape=(100, 2), minval=0.0, maxval=1.0)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    x_values, y_values = sess.run([x_values_gen, y_values_gen])

    for i in range(5000):

        loss_val, _ = sess.run([loss, train_op],
                               feed_dict={
                                   x_placeholder: x_values,
                                   y_placeholder: y_values,
                               })

        if i % 50 == 0:
            print(i, "Loss", loss_val)
            saver = tf.train.Saver(var_list=[weights, bias], max_to_keep=1)
            saver.save(sess, save_path='models/model.ckpt')