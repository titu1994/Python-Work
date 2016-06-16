from theano import *
import theano.tensor as T
import numpy as np

# Logistic function
x = T.matrix('x', 'float32')
op = 1 / (1 + T.exp(-x))

logistic = function([x], op)

mat1 = [[0, 1], [-1, -2]]
print(logistic(mat1))

# Multiple outputs
a, b = T.fmatrices('a', 'b')
diff = a - b
absDiff = abs(diff)
sqrDiff = diff ** 2

f = function([a, b], [diff, absDiff, sqrDiff])

mat2 = [[10, 5], [5, 10]]
mat3 = [[5, 10], [10, 5]]

print(f(mat2, mat3))

# Default values

x, y = T.fscalars('x', 'y')
z = x + y

f = function([x, In(y, value=0)], z)

print(f(20))
print(f(20, 10))

# Shared variables

state = shared(0)
inc = T.iscalar('inc')

accumulator = function([inc], state, updates=[(state, state + inc)])

print("state : ", state.get_value())
accumulator(1)
print("state : ", state.get_value())
accumulator(300)
print("state : ", state.get_value())

print("resetting state")
state.set_value(0)
print("state : ", state.get_value())

# Copying functions

newState = shared(0)
acc2 = accumulator.copy(swap={state:newState})
acc2(1000)

print('original state : ', state.get_value())
print("new state : ", newState.get_value())

# Using Random Numbers

from theano.tensor.shared_randomstreams import RandomStreams
srng = RandomStreams(seed=1)

rv_u = srng.uniform((2, 2))
rv_v = srng.normal((2, 2))

f = function([], rv_u)
g = function([], rv_v, no_default_updates=True)
nearly_zeros = function([], rv_u + rv_u - 2 * rv_u)

print(f(), ' ', f())
print(g(), ' ', g())
print(nearly_zeros(), ' ', nearly_zeros())