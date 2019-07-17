import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy

from graph.core import Variable

def factorial(n):
    if n <= 1:
        return Variable(None, n)
    else:
        return factorial(n - 1) * Variable(None, n)

# n = 7
# f = factorial(n)
#
# print(f)
# print()
#
# f.resolve_expression(topdown=True)

""" Fibonacci """

def fibonacci (n):
    if n <= 1:
        return Variable(None, n)

    return fibonacci(n - 1) + fibonacci(n - 2)

# n = 5
# f = fibonacci(n)
#
# print(f)
# print()
#
# f.resolve_expression(topdown=True)

def fibonacci_gen(n):
    a = Variable('a', 0)
    b = Variable('b', 1)

    for _ in range(n):
        yield a
        a, b = a.assign(b, name='a'), b.assign(a + b, name='b')

# n = 5
# fs = list(fibonacci_gen(n))
#
# print(fs[-1])
# print()

""" PI Finding """

""" Track 1 """

one = Variable('1', 1.)
two = Variable('2', 2.)

# def pi(x, r):
#     square = x ** two
#     return (r - square) ** (one / two)
#
# n = 20000
#
# cx = Variable('cx', 2.0 / n)
# area = Variable('area', 0.0)
#
# for i in range(n):
#     i_node = Variable(None, i)
#
#     x = (-one) + cx * i_node
#     area += pi(x, one) * cx
#
# pi_calc = two * area
#
# print(pi_calc)
# print()
#
# # pi_calc.resolve_expression()

""" Golden Ratio Phi """

def phi(n):
    if n <= 1:
        return Variable(None, 3)

    return one + one / phi(n - 1)

# n = 5
# p = phi(n)
#
# print(p)
# print()
#
# p.resolve_expression(topdown=True)


""" Assignment """

# x = Variable('x', 1)
# y = Variable('y', 2)
#
# z = x * y
# z += x.assign(y)
# z = y.assign(z, name='z')
#
# print(x)
# print(y)
# print(z)
# print()
#
# z.resolve_expression(topdown=True)


""" Running Average """
# np.random.seed(0)
#
# def input_attenuation(n):
#     beta = Variable('beta', 0.9)
#     beta_inv = Variable('(1 - beta)', 1. - beta.data)
#
#     w = Variable('w', 0.)
#     x = Variable('x', 1.)
#
#     X = []
#     W = []
#
#     for i in range(n):
#         v = beta * w + beta_inv * x
#         x = x.assign(x - v, name='x')
#         w = w.assign(v, name='w')
#
#         print(i + 1, x)
#         print(i + 1, w)
#         print()
#
#         X.append(x.data)
#         W.append(w.data)
#
#     return X, W
#
# n = 100
# X, W = input_attenuation(n)
#
# plt.plot(X, label='X')
# plt.plot(W, label='Moving average W')
# plt.legend()
# plt.show()


# w.resolve_expression(topdown=True)
