import time
import argparse
import os

import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from tfdiffeq import odeint
tf.enable_eager_execution()

parser = argparse.ArgumentParser('ODE demo')
parser.add_argument('--method', type=str, choices=['dopri5', 'adams'], default='dopri5')
parser.add_argument('--data_size', type=int, default=20)
parser.add_argument('--batch_time', type=int, default=10)
parser.add_argument('--batch_size', type=int, default=20)
parser.add_argument('--niters', type=int, default=2000)
parser.add_argument('--test_freq', type=int, default=20)
parser.add_argument('--viz', action='store_true')
parser.add_argument('--gpu', type=int, default=0)
# parser.add_argument('--adjoint', type=eval, default=False)
parser.set_defaults(viz=True)
args = parser.parse_args()

device = 'gpu:' + str(args.gpu) if tf.test.is_gpu_available() else 'cpu:0'

true_y0 = tf.convert_to_tensor(1, dtype=tf.float64)
time = np.linspace(0, 35., num=args.data_size)
t = tf.convert_to_tensor(time, dtype=tf.float32)
# true_A = tf.convert_to_tensor([[-0.1, 2.0], [-2.0, -0.1]], dtype=tf.float64)


def true_y(t, y0):
    # dy / dt = 5x - 3
    return t + y0  # 0.2 * np.exp(5 * t + 5 * y0) + 3. / 5.


class Lambda(tf.keras.Model):

    def call(self, t, y):
        if t < 10.:
            u = 0.0
        else:
            u = 2.0

        out = (-y + u) / 5.0
        return out


real_y = [true_y(t, true_y0.numpy()) for t in time]
pred_y = odeint(Lambda(), true_y0, t, method=args.method)

mse = np.mean(np.square(real_y - pred_y.numpy()))
print("MSE :", mse)

for i, (real, pred) in enumerate(zip(real_y, pred_y.numpy())):
    print(i + 1, real, pred)

plt.plot(pred_y.numpy(), label='integral')
plt.legend()
plt.show()

