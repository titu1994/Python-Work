import numpy as np
from collections import defaultdict
from uuid import uuid4


_var_global_counter = 1


class Variable(object):

    def __init__(self, name, data, parents=None, op=None, id=None):
        global _var_global_counter

        if name is None:
            name = '#%d' % (_var_global_counter)
            _var_global_counter += 1

        self.name = name
        self.data = data
        self.parents = parents
        self.op_name = op
        self.id = id if id is not None else uuid4()

    def resolve_expression(self, topdown=True):
        if self.parents is not None:
            if topdown:
                print("Operation : ", self)

            for i, parent in enumerate(self.parents):
                if parent is not None:
                    parent.resolve_expression(topdown)

                    if not topdown:
                        print(parent)

            if not topdown:
                print("Operation : ", self)
                print('-' * 10)
                print()
        else:
            if topdown:
                print(self)

    def assign(self, other, name=None):
        if name is None:
            name = '%s\'' % (self.name)

        v = Variable(name,
                     data=other.data,
                     parents=[other],
                     op='assign')

        v._assignment = '(%s = %s)' % (self.name, other.name)
        
        return v

    def __add__(self, other):
        return Variable('(%s + %s)' % (self.name, other.name),
                        data=self.data + other.data,
                        parents=[self, other],
                        op='_var_add')

    def __sub__(self, other):
        return Variable('(%s - %s)' % (self.name, other.name),
                        data=self.data - other.data,
                        parents=[self, other],
                        op='sub')

    def __mul__(self, other):
        return Variable('(%s * %s)' % (self.name, other.name),
                        data=self.data * other.data,
                        parents=[self, other],
                        op='mul')

    def __truediv__(self, other):
        return Variable('(%s / %s)' % (self.name, other.name),
                        data=self.data / other.data,
                        parents=[self, other],
                        op='div')

    def __floordiv__(self, other):
        return Variable('(%s // %s)' % (self.name, other.name),
                        data=self.data // other.data,
                        parents=[self, other],
                        op='floor_div')

    def __neg__(self):
        return Variable('(-%s)' % (self.name),
                        data=-self.data,
                        parents=[self],
                        op='neg')

    def __pow__(self, power, modulo=None):
        return Variable('(%s ** %s)' % (self.name, power.name),
                        data=self.data ** power.data,
                        parents=[self, power],
                        op='pow')

    def __repr__(self):
        if hasattr(self, '_assignment'):
            return '%s [%s] [value = %s]' % (self.name, self._assignment, str(self.data))

        return '%s [value = %s]' % (self.name, str(self.data))