import numpy as np

from numpygrad.tensor import Tensor
from numpygrad.optim import SGD, Adam
from numpygrad.layers import Sequential, Dense, Embedding
from numpygrad.rnn import RNN
from numpygrad.losses import MSELoss, CrossEntropyLoss
from numpygrad.activations import Sigmoid, Tanh, ReLU

# https://github.com/iamtrask/Grokking-Deep-Learning/blob/master/Chapter13%20-%20Intro%20to%20Automatic%20Differentiation%20-%20Let's%20Build%20A%20Deep%20Learning%20Framework.ipynb

""" Part 1 """
# x = Tensor([1, 2, 3, 4, 5])
# print(x)
#
# y = x + Tensor([-1, -2, -3, -4, -5])
# print(y)

""" Part 2 """
# x = Tensor([1, 2, 3, 4, 5])
# y = Tensor([2, 2, 2, 2, 2])
#
# z = x + y
# z.backward(Tensor(np.array([1, 1, 1, 1, 1])))
#
# print(x.grad)
# print(y.grad)
# print(z.creators)
# print(z.creation_op)
#
# print()
#
# a = Tensor([1, 2, 3, 4, 5])
# b = Tensor([2, 2, 2, 2, 2])
# c = Tensor([5, 4, 3, 2, 1])
# d = Tensor([-1, -2, -3, -4, -5])
#
# pi = a + b
# f = c + d
# g = pi + f
#
# g.backward(Tensor(np.array([1, 1, 1, 1, 1])))
#
# print(a.grad)

""" Part 3 """
# a = Tensor([1,2,3,4,5])
# b = Tensor([2,2,2,2,2])
# c = Tensor([5,4,3,2,1])
#
# d = a + b
# pi = b + c
# f = d + pi
# f.backward(Tensor(np.array([1,1,1,1,1])))
#
# print(b.grad.data == np.array([2,2,2,2,2]))  # [False False False False False]

""" Part 4 """
# a = Tensor([1, 2, 3, 4, 5], autograd=True)
# b = Tensor([2, 2, 2, 2, 2], autograd=True)
# c = Tensor([5, 4, 3, 2, 1], autograd=True)
#
# d = a + b
# pi = b + c
# f = d + pi
#
# f.backward(Tensor(np.array([1, 1, 1, 1, 1])))
#
# print(f)
# print(b.grad.data == np.array([2, 2, 2, 2, 2]))

""" Part 5 """
# a = Tensor([1, 2, 3, 4, 5], autograd=True)
# b = Tensor([2, 2, 2, 2, 2], autograd=True)
# c = Tensor([5, 4, 3, 2, 1], autograd=True)
#
# d = a + (-b)
# pi = (-b) + c
# f = d + pi
#
# f.backward(Tensor(np.array([1, 1, 1, 1, 1])))
#
# print(b.grad.data == np.array([-2, -2, -2, -2, -2]))

""" Part 6 """
# x = Tensor(np.array([[1, 2, 3],
#                      [4, 5, 6]]), autograd=True)
#
# print(x.shape)
# print(x.sum(0))
# print(x.sum(1))
#
# print()
# y = x.expand(axis=0, repeats=4)
# print(y.shape)
# print(y)
#
# y.backward()
# print(y.grad.shape, y.shape, x.grad.shape, x.shape)

""" Part 7 """
""" manual backprop """
# np.random.seed(0)
#
# data = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
# target = np.array([[0], [1], [0], [1]])
#
# weights_0_1 = np.random.uniform(size=(2, 3))
# weights_1_2 = np.random.uniform(size=(3, 1))
#
# for i in range(15):
#     # Predict
#     layer_1 = data.dot(weights_0_1)
#     layer_2 = layer_1.dot(weights_1_2)
#
#     # Compare
#     diff = layer_2 - target
#     error = np.sum(np.square(diff))
#
#     # Learn: this is the manual backpropagation
#     layer_1_grad = diff.dot(weights_1_2.transpose())
#     weight_1_2_update = layer_1.transpose().dot(diff)
#     weight_0_1_update = data.transpose().dot(layer_1_grad)
#
#     weights_1_2 -= weight_1_2_update * 0.1
#     weights_0_1 -= weight_0_1_update * 0.1
#     print(error)
#
""" automatic backprop """
# np.random.seed(0)

# data = Tensor(np.array([[0, 0], [0, 1], [1, 0], [1, 1]]), autograd=True)
# target = Tensor(np.array([[0], [1], [0], [1]]), autograd=True)
#
# w = list()
# w.append(Tensor(np.random.rand(2, 3), autograd=True))
# w.append(Tensor(np.random.rand(3, 1), autograd=True))
#
# for i in range(10):
#     # Predict
#     pred = data.dot(w[0]).dot(w[1])
#
#     # Compare
#     loss = ((pred - target) * (pred - target)).sum(0)
#
#     # Learn
#     loss.backward()
#
#     for w_ in w:
#         w_.data -= w_.grad.data * 0.1
#         w_.grad.data *= 0
#
#     print(loss)

""" Part 8 """
# np.random.seed(0)
#
# data = Tensor(np.array([[0, 0], [0, 1], [1, 0], [1, 1]]), autograd=True)
# target = Tensor(np.array([[0], [1], [0], [1]]), autograd=True)
#
# w = list()
# w.append(Tensor(np.random.rand(2, 3), autograd=True))
# w.append(Tensor(np.random.rand(3, 1), autograd=True))
#
# # optim = SGD(w, lr=0.1)
# optim = Adam(w, lr=0.1)
#
# for i in range(10):
#     # Predict
#     pred = data.dot(w[0]).dot(w[1])
#
#     # Compare
#     loss = ((pred - target) * (pred - target)).sum(0)
#
#     # Learn
#     loss.backward()
#     optim.step()
#
#     print(loss)

""" Part 9 - 10 """
# np.random.seed(0)
#
# data = Tensor(np.array([[0, 0], [0, 1], [1, 0], [1, 1]]), autograd=True)
# target = Tensor(np.array([[0], [1], [0], [1]]), autograd=True)
#
# model = Sequential([Dense(2, 3), Dense(3, 1)])
#
# optim = SGD(parameters=model.parameters, lr=0.05)
#
# for i in range(10):
#     # Predict
#     pred = model(data)
#
#     # Compare
#     loss = ((pred - target) * (pred - target)).sum(0)
#
#     # Learn
#     loss.backward()
#     optim.step()
#     print(loss)

""" Part 11 """
# np.random.seed(0)
#
# data = Tensor(np.array([[0, 0], [0, 1], [1, 0], [1, 1]]), autograd=True)
# target = Tensor(np.array([[0], [1], [0], [1]]), autograd=True)
#
# model = Sequential([Dense(2, 3), Dense(3, 1)])
# criterion = MSELoss()
#
# optim = SGD(parameters=model.parameters, lr=0.05)
#
# for i in range(10):
#     # Predict
#     pred = model(data)
#
#     # Compare
#     loss = criterion(pred, target)
#
#     # Learn
#     loss.backward()
#     optim.step()
#     print(loss)

""" Part 12 """
# np.random.seed(0)
# #
# # data = Tensor(np.array([[0, 0], [0, 1], [1, 0], [1, 1]]), autograd=True)
# # target = Tensor(np.array([[0], [1], [0], [1]]), autograd=True)
# #
# # model = Sequential([Dense(2, 3), Tanh(), Dense(3, 1), Sigmoid()])
# # criterion = MSELoss()
# #
# # optim = Adam(parameters=model.parameters, lr=0.5)
# #
# # for i in range(10):
# #     # Predict
# #     pred = model(data)
# #
# #     # Compare
# #     loss = criterion(pred, target)
# #
# #     # Learn
# #     loss.backward()
# #     optim.step()
# #     print(loss)

""" Part 12-15 """
# x = Tensor(np.eye(5), autograd=True)
# x.index_select(Tensor([[1, 2, 3], [2, 3, 4]])).backward()
# print(x.grad)
# print()

# np.random.seed(0)
#
# data = Tensor(np.array([1, 2, 1, 2]), autograd=True)
# target = Tensor(np.array([[0], [1], [0], [1]]), autograd=True)
#
# embed = Embedding(5, 3)
# model = Sequential([embed, Tanh(), Dense(3, 1), Sigmoid()])
# criterion = MSELoss()
#
# optim = SGD(parameters=model.parameters, lr=0.5)
#
# for i in range(10):
#     # Predict
#     pred = model.forward(data)
#
#     # Compare
#     loss = criterion.forward(pred, target)
#
#     # Learn
#     loss.backward()
#     optim.step()
#     print(loss)

""" Part 16 """
# np.random.seed(0)
#
# # data indices
# data = Tensor(np.array([1, 2, 1, 2]), autograd=True)
#
# # target indices
# target = Tensor(np.array([0, 1, 0, 1]), autograd=True)
#
# model = Sequential([Embedding(3, 3), Tanh(), Dense(3, 4)])
# criterion = CrossEntropyLoss()
#
# optim = SGD(parameters=model.parameters, lr=0.5)
#
# for i in range(10):
#     # Predict
#     pred = model(data)
#
#     # Compare
#     loss = criterion(pred, target)
#
#     # Learn
#     loss.backward()
#     optim.step()
#     print(loss)

""" Part 17 """
np.random.seed(0)

timesteps = 50  # more than 50 causes recursive memory to be exhausted ; bypass it using sys.setrecursionlimit()

data = np.linspace(-np.pi * 2, np.pi * 2, num=timesteps).reshape(1, -1)
data = Tensor(data, autograd=True)

target = Tensor(np.sin(data.data), autograd=True)

rnn = RNN(n_in=1, n_hidden=5, n_out=timesteps, activation='tanh')  # more neurons = faster convergence

criterion = MSELoss()
optim = SGD(rnn.parameters, lr=0.025)  # anything above 0.025 converges perfectly

for i in range(10):  # more iterations = better convergence
    batchsize = 1

    state = rnn.init_state(batchsize)

    for t in range(timesteps):
        ip = data.data[:, [t]]
        out, state = rnn(input=ip, states=state)  # out = [1, T] ; predicts all timesteps at once.

    loss = criterion(target, out)
    loss.backward()
    optim.step()

    print('loss', loss.data.mean())


import matplotlib.pyplot as plt
plt.plot(target.data.flatten(), label='real')
plt.plot(out.data.flatten(), label='predicted')
plt.legend()
plt.show()

