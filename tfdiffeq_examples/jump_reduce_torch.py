import time
import argparse
import os

import numpy as np
import matplotlib.pyplot as plt

import torch
from torchdiffeq import odeint

parser = argparse.ArgumentParser('ODE demo')
parser.add_argument('--method', type=str, choices=['dopri5', 'adams'], default='dopri5')
parser.add_argument('--data_size', type=int, default=2000)
parser.add_argument('--rtol', type=float, default=1e-3)
parser.add_argument('--atol', type=float, default=1e-3)
parser.add_argument('--viz', action='store_true')
parser.add_argument('--gpu', type=int, default=0)
# parser.add_argument('--adjoint', type=eval, default=False)
parser.set_defaults(viz=True)
args = parser.parse_args()

torch.set_default_dtype(torch.float64)
device = torch.device('cuda:' + str(args.gpu) if torch.cuda.is_available() else 'cpu')

true_y0 = torch.tensor(1,).float().to(device)
t_n = np.linspace(0, 100., num=args.data_size)
t = torch.tensor(t_n).to(device)


class Lambda(torch.nn.Module):

    def forward(self, t, y):
        dydt = -t * y + 1 / y
        return dydt

t1 = time.time()
pred_y = odeint(Lambda(), true_y0, t, rtol=args.rtol, atol=args.atol, method=args.method)
t2 = time.time()

print("Number of solutions : ", pred_y.shape)
print("Time taken : ", t2 - t1)

plt.plot(t_n, pred_y.cpu().numpy(), 'r-', label='x')
# plt.plot(time, pred_y.numpy(), 'b--', label='y')
plt.legend()
plt.show()

