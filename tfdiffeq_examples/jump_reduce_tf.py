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
parser.add_argument('--data_size', type=int, default=2000)
parser.add_argument('--rtol', type=float, default=1e-7)
parser.add_argument('--atol', type=float, default=1e-9)
parser.add_argument('--viz', action='store_true')
parser.add_argument('--gpu', type=int, default=0)
# parser.add_argument('--adjoint', type=eval, default=False)
parser.set_defaults(viz=True)
args = parser.parse_args()

device = 'gpu:' + str(args.gpu) if tf.test.is_gpu_available() else 'cpu:0'

true_y0 = tf.convert_to_tensor(1, dtype=tf.float64)
t_n = np.linspace(0, 100., num=args.data_size)
t = tf.convert_to_tensor(t_n, dtype=tf.float32)


class Lambda(tf.keras.Model):

    def call(self, t, y):
        dydt = tf.exp(-t * y) * 1 / y
        return dydt

with tf.device(device):
    t1 = time.time()
    pred_y = odeint(Lambda(), true_y0, t, rtol=args.rtol, atol=args.atol, method=args.method)
    t2 = time.time()

print("Number of solutions : ", pred_y.shape)
print("Time taken : ", t2 - t1)

plt.plot(t_n, pred_y.numpy(), 'r-', label='x')
# plt.plot(time, pred_y.numpy(), 'b--', label='y')
plt.legend()
plt.show()

