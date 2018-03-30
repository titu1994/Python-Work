import tensorflow as tf
from tensorflow.contrib.eager.python import tfe
tf.enable_eager_execution()

x = tfe.Variable(initial_value=tf.random_uniform([1], -1., 1.), name='x')

def loss(input):
    return tf.sigmoid(input)

grad_vars = tfe.implicit_gradients(loss)
opt = tf.train.GradientDescentOptimizer(learning_rate=1)

for i in range(1000):
    for j in range(50):
        opt.apply_gradients(grad_vars(x))

    if i % 50 == 0:
        loss_val = loss(x)
        print(i, "Optimal Value : ", loss_val.numpy(), "Val (X) : ", x.numpy())