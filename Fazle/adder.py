import tensorflow as tf

x1 = 10
x2 = 3

with tf.name_scope('inputs'):
    inputs = tf.constant([x1, x2], dtype=tf.float32, shape=(1, 2), name='inputs')

with tf.name_scope('weights'):
    weights = tf.ones(shape=(2, 1), dtype=tf.float32, name='weights')
    biases = tf.zeros(shape=(1, 1), dtype=tf.float32, name='biases')

with tf.name_scope('model'):
    prediction = tf.matmul(inputs, weights) + biases

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    writer = tf.summary.FileWriter('./logs/', graph=sess.graph)

    output = sess.run(prediction)
    output = output.flatten()[0]

    print("Sum of %0.2f + %0.2f = %0.2f" % (x1, x2, output))

    writer.close()




