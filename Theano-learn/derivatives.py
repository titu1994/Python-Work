import theano.tensor as T
from theano import *
import numpy as np

# Derivative of x ** 2

x = T.dscalar('x')
y = x ** 2

# Derivative of x ** 2 = 2 * x
gradY = T.grad(y, x)
print("Symbolic derivative : ", pp(gradY))

f = function([x], gradY)
print("Derivative function : ", pp(f.maker.fgraph.outputs[0])) # Should return 2 * x

print(f(4))

# Second order derivatives

d2y = T.grad(gradY, x)
print("Symbolic derivative : ", pp(d2y))

f = function([x], d2y)
print("Derivative function : ", pp(f.maker.fgraph.outputs[0])) # Should return 2

print(f(4)) # Should return 2

# Derivative of logistic function

x = T.dmatrix('x')
s = T.sum(1 / (1 + T.exp(-x)))

dS = T.grad(s, x)

dlogistic = function([x], dS)
mat1 = [[0, 1], [-1, -2]]

print(dlogistic(mat1))



