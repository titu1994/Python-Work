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

true_y0 = tf.convert_to_tensor([0, 0], dtype=tf.float64)
time = np.linspace(0, 35., num=args.data_size)
t = tf.convert_to_tensor(time, dtype=tf.float32)


class Lambda(tf.keras.Model):

    def call(self, t, y):
        dxdt = 3 * tf.exp(-t)
        dydt = 3 - y[1]
        return [dxdt, dydt]


pred_y = odeint(Lambda(), true_y0, t, method=args.method)

print("Number of solutions : ", pred_y.shape)

plt.plot(time, pred_y[:, 0].numpy(), 'r-', label='x')
plt.plot(time, pred_y[:, 1].numpy(), 'b--', label='y')
plt.legend()
plt.show()

