import numpy as np
import tensorflow as tf
from tensorflow.contrib.eager.python import tfe

tf.enable_eager_execution()
tf.set_random_seed(0)

device = '/gpu:0' if tfe.num_gpus() > 0 else '/cpu:0'

# variables
x = tf.get_variable('x', dtype=tf.float32, initializer=1.0)
y = tf.get_variable('y', dtype=tf.float32, initializer=1.0)

# function to optimize
def f(x, y):
    return x + y

# Constraint
# Solution must => x^2 + y^2 = 1

lambd = tf.get_variable('lambda', dtype=tf.float32, initializer=1.0,
                        constraint=lambda x: tf.clip_by_value(x, 0., np.infty))

def constraint(x, y):
    return (x * x + y * y - 1)


def L(x, y, l):
    return -f(x, y) + l * constraint(x, y)

optimizer = tf.train.GradientDescentOptimizer(0.05)

for i in range(1000):
    #loss_val, grad_vars = gradients(x, y, lambd_x)
    #optimizer.apply_gradients(grad_vars, tf.train.get_or_create_global_step())

    optimizer.minimize(lambda: L(x, y, lambd), tf.train.get_or_create_global_step())

    #lambd_x = tf.clip_by_value(lambd_x, 0., np.inf)
    print("L", lambd.numpy())

    if i % 1 == 0:
        print("X", x.numpy(), "Y", y.numpy(), "norm", (x ** 2 + y ** 2).numpy())

        loss_val = L(x, y, lambd)
        print("Iteration %d : Loss %0.4f, function value : %0.4f" % (i + 1, loss_val.numpy(), f(x, y).numpy()))
        print()


print("X", x.numpy(), "Y", y.numpy(), "norm", (x ** 2 + y ** 2).numpy())