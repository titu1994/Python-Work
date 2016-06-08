from theano import *
import theano.tensor as T
import numpy as np

x = T.dscalar('x')
y = T.dscalar('y')
z = x + y

f = function([x, y], z)

print(f(2, 3))

x = T.dmatrix('x')
y = T.dmatrix('y')
z = x + y

f = function([x, y], z)

mat1 = [[10, 5], [5, 10]]
mat2 = [[5, 10], [10, 5]]

print(f(mat1, mat2))

a = T.vector('a', 'float32')
b = a + a ** 10

f = function([a], b)

print(f([0, 1, 2]))