import tensorflow as tf
from keras.datasets import mnist
from keras.utils import to_categorical

tf.set_random_seed(0)
tf.logging.set_verbosity(tf.logging.INFO)

(X_train, Y_train), (X_test, Y_test) = mnist.load_data()

X_train = X_train.reshape((-1, 28 * 28))
X_test = X_test.reshape((-1, 28 * 28))
Y_train = to_categorical(Y_train, 10)
Y_test = to_categorical(Y_test, 10)

learning_rate = 0.001
training_epochs = 50
batchsize = 250
l2_reg = 1e-4

number_of_inputs = X_train.shape[1]
number_of_outputs = Y_train.shape[1]
number_of_batches = X_train.shape[0] // batchsize

layer_1_nodes = 200
layer_2_nodes = 200
layer_3_nodes = 10


with tf.name_scope('inputs'):
    X_placeholder = tf.placeholder(tf.float32,shape=[None, number_of_inputs])
    y_placeholder = tf.placeholder(tf.float32, shape=[None, number_of_outputs])

with tf.variable_scope('layers', reuse=tf.AUTO_REUSE,
                       initializer=tf.keras.initializers.he_uniform(seed=0),
                       regularizer=tf.keras.regularizers.l2(l2_reg)):

    layer_1_output = tf.keras.activations.elu(tf.layers.dense(X_placeholder, layer_1_nodes,use_bias=True, name='dense_1'), alpha=1.2)
    layer_2_output = tf.keras.activations.elu(tf.layers.dense(layer_1_output, layer_2_nodes,use_bias=True, name='dense_2'), alpha=1.2)
    layer_3_output = tf.layers.dense(layer_2_output, layer_3_nodes, activation=None, use_bias=True, name='dense_3')

with tf.name_scope('predictions'):
    prediction = tf.nn.softmax(layer_3_output)

with tf.name_scope('loss'):
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_placeholder, logits=layer_3_output))
    train_op = tf.train.AdamOptimizer(learning_rate).minimize(loss)

with tf.name_scope('metrics'):
    accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(y_placeholder, axis=-1), tf.argmax(prediction, axis=-1)), tf.float32))

with tf.name_scope('dataset'):
    dataset = tf.data.Dataset.from_tensor_slices((X_placeholder, y_placeholder))
    dataset = dataset.batch(batchsize)
    dataset = dataset.shuffle(buffer_size=100, seed=0, reshuffle_each_iteration=True)
    dataset = dataset.repeat()

    dataset_initializer = dataset.make_initializable_iterator()
    dataset_iterator = dataset_initializer.get_next()

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())  # needed for tf.metrics.accuracy

    sess.run(dataset_initializer.initializer, feed_dict={X_placeholder: X_train, y_placeholder: Y_train})

    for epoch in range(training_epochs):
        for batch in range(number_of_batches):
            x_batch, y_batch = sess.run(dataset_iterator)
            sess.run(train_op, feed_dict={X_placeholder: x_batch, y_placeholder: y_batch})

        if (epoch + 1) % 5 == 0:
            train_loss = sess.run(loss, feed_dict={X_placeholder: X_train, y_placeholder: Y_train})
            test_loss = sess.run(loss, feed_dict={X_placeholder: X_test, y_placeholder: Y_test})

            print("Epoch: {} - Training Cost: {}  Testing Cost: {}".format(epoch + 1, train_loss, test_loss))

    final_training_cost = sess.run(loss, feed_dict={X_placeholder: X_train, y_placeholder: Y_train})
    final_testing_cost = sess.run(loss, feed_dict={X_placeholder: X_test, y_placeholder: Y_test})

    print("Final Training loss: {}".format(final_training_cost))
    print("Final Testing loss: {}".format(final_testing_cost))

    train_prediction = sess.run(accuracy, feed_dict={X_placeholder: X_train, y_placeholder: Y_train})
    test_prediction = sess.run(accuracy, feed_dict={X_placeholder: X_test, y_placeholder: Y_test})

    print("Final Training accuracy: {}".format(train_prediction))
    print("Final Testing accuracy: {}".format(test_prediction))