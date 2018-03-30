import tensorflow as tf 
from os import environ
import numpy as np 
from filereader import Filereader as F 

from keras.datasets import mnist

#environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.logging.set_verbosity(tf.logging.INFO)

# f = F()
# X_train, Y_train, N, M = f.getData(sample=60000)
# X_test, Y_test, N_test, M_test = f.getData(dataset="testing",sample=10000)
(X_train, Y_train), (X_test, Y_test) = mnist.load_data()

print(X_train.shape, Y_train.shape)

learning_rate = 0.001
training_epochs = 100 

number_of_inputs = X_train.shape[1]
number_of_outputs = Y_train.shape[1]

layer_1_nodes = 200
layer_2_nodes = 200
layer_3_nodes = 10


def accuracy(predictions, actuals):
	# can be replaced by tf.metrics.accuracy()
	return np.mean(np.argmax(actuals,axis=1)==np.argmax(predictions,axis=1))

with tf.variable_scope('input'):
	X = tf.placeholder(tf.float32, shape=(None, number_of_inputs))
	Y = tf.placeholder(tf.float32, shape=(None, 10))

with tf.variable_scope('layer_1'):
	weights = tf.get_variable("weights1", shape=[number_of_inputs, layer_1_nodes], initializer=tf.contrib.layers.xavier_initializer())
	biases = tf.get_variable("bias1", shape=[layer_1_nodes], initializer=tf.zeros_initializer())
	layer_1_output = tf.nn.elu(tf.matmul(X, weights) + biases)

with tf.variable_scope('layer_2'):
	weights = tf.get_variable("weights2", shape=[layer_1_nodes, layer_2_nodes], initializer=tf.contrib.layers.xavier_initializer())
	biases = tf.get_variable("bias2", shape=[layer_2_nodes], initializer=tf.zeros_initializer())
	layer_2_output = tf.nn.elu(tf.matmul(layer_1_output, weights) + biases)

with tf.variable_scope('layer_3'):
	weights = tf.get_variable("weights3", shape=[layer_2_nodes, layer_3_nodes], initializer=tf.contrib.layers.xavier_initializer())
	biases = tf.get_variable("bias3", shape=[layer_3_nodes], initializer=tf.zeros_initializer())
	prediction = tf.add(tf.matmul(layer_2_output, weights),biases)

with tf.variable_scope('loss'):
	cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=Y,logits=prediction))

with tf.variable_scope('train'):
	optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)

with tf.Session() as s:
	s.run(tf.global_variables_initializer())
	s.run(tf.local_variables_initializer())
	for epoch in range(training_epochs):
		s.run(optimizer, feed_dict={X: X_train, Y: Y_train})

		if epoch % 5 == 0:
			# Get the current accuracy scores by running the "loss" operation on the training and test data sets
			training_cost = s.run(cost, feed_dict={X: X_train, Y:Y_train})
			testing_cost = s.run(cost, feed_dict={X: X_test, Y:Y_test})

			print("Epoch: {} - Training Cost: {}  Testing Cost: {}".format(epoch, training_cost, testing_cost))

	final_training_cost = s.run(cost, feed_dict={X: X_train, Y: Y_train})
	final_testing_cost = s.run(cost, feed_dict={X: X_test, Y: Y_test})

	print("Final Training loss: {}".format(final_training_cost))
	print("Final Testing loss: {}".format(final_testing_cost))

	train_prediction = s.run(tf.nn.softmax(prediction), feed_dict={X:X_train})
	test_prediction = s.run(tf.nn.softmax(prediction), feed_dict={X:X_test})

	print("Final Training accuracy: {}".format(accuracy(train_prediction,Y_train)))
	print("Final Testing accuracy: {}".format(accuracy(test_prediction,Y_test)))

