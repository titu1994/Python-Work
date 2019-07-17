import numpy as np
from copy import deepcopy
from graph.core import Variable
from graph.cell import Cell


""" Small Cell """
# x = Variable('x', 0)
# y = Variable('y', 0)
# z = Variable('z', 0)
#
# c1 = Cell('C1', inputs=[x, y], ops=['sigmoid', 'tanh'], combine_op='add')
# c2 = Cell('C2', inputs=[z, c1], ops=['relu', 'no-op'], combine_op='concat')
# c3 = Cell('C3', inputs=[c1, c2], ops=['relu', 'tanh'], combine_op='add')
#
# print(c3)


""" LSTM Gates """

# Inputs
x = Variable('x', 0)
ht_1 = Variable('ht-1', 0)
ct_1 = Variable('ct-1', 0)

# Weights
W = Variable('w . x', 0)
U = Variable('u . ht-1', 0)

# Gates
i = Cell('i', inputs=[W, U], ops=None, combine_op='add').function('sigmoid')
f = Cell('f', inputs=[W, U], ops=None, combine_op='add').function('sigmoid')
c = Cell('c', inputs=[W, U], ops=None, combine_op='add').function('sigmoid')
o = Cell('o', inputs=[W, U], ops=None, combine_op='add').function('sigmoid')

c1 = f * ct_1
c2 = i * c
c = Cell('c', inputs=[c1, c2], ops=None, combine_op='add').function('tanh')
h = Cell('h', inputs=[o, c], ops=None, combine_op='mul')

print(h)



