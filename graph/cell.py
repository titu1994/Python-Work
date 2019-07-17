import numpy as np
from copy import deepcopy
from collections import defaultdict
from typing import List
from uuid import uuid4

from graph.core import Variable


_cell_global_counter = 1


class Cell(object):

    def __init__(self, name, inputs, ops, combine_op, id=None):
        global _cell_global_counter

        if name is None:
            name = '#C%d' % (_cell_global_counter)
            _cell_global_counter += 1

        self.name = name
        self.inputs = list(inputs)  # type: List[Variable]
        self.ops = ops  # type: List[str]
        self.combine_op = combine_op  # type: str
        self.id = id if id is not None else uuid4()

        self.cell = self._build_cell()

    def function(self, activation):
        var_name = self._get_name(self)

        if hasattr(self, 'cell') and self.cell is not None:
            self.cell.name = '%s(%s)' % (activation, var_name)
        else:
            self.name = '%s(%s)' % (activation, var_name)

        return self

    def concat(self, other):
        # type: (Cell) -> Cell
        return Cell('([%s ; %s])' % (self.name, self._get_name(other)),
                    inputs=[self, other],
                    ops=None,
                    combine_op='concat')

    def dot(self, other):
        # type: (Cell) -> Cell
        return Cell('(%s . %s)' % (self.name, self._get_name(other)),
                    inputs=[self, other],
                    ops=None,
                    combine_op='dot')

    def __add__(self, other):
        # type: (Cell) -> Cell
        return Cell('(%s + %s)' % (self.name, self._get_name(other)),
                    inputs=[self, other],
                    ops=None,
                    combine_op='add')

    def __sub__(self, other):
        # type: (Cell) -> Cell
        return Cell('(%s - %s)' % (self.name, self._get_name(other)),
                    inputs=[self, other],
                    ops=None,
                    combine_op='sub')

    def __mul__(self, other):
        # type: (Cell) -> Cell
        return Cell('(%s * %s)' % (self.name, self._get_name(other)),
                    inputs=[self, other],
                    ops=None,  # [self.ops, self._get_ops(other)]
                    combine_op='mul')

    def _build_cell(self):
        if len(self.inputs) != 2:
            return None

        if self.ops is None:
            self.ops = ['noop', 'noop']

        elif len(self.ops) != 2:
            raise ValueError('2 Operations must be provided. '
                             'If no operation is required on a cell input, use `noop`.')

        x = self.inputs[0]
        y = self.inputs[1]

        # perform respective op on inputs
        o1 = Variable(self._check_special_ops(self.ops[0], x),
                      data=0,
                      parents=[x],
                      op=self.ops[0])

        o2 = Variable(self._check_special_ops(self.ops[1], y),
                      data=0,
                      parents=[y],
                      op=self.ops[1])

        if self.combine_op == 'concat':
            cell = self._var_concat(o1, o2)
        elif self.combine_op == 'dot' or self.combine_op == 'matmul':
            cell = self._var_dot(o1, o2)
        elif self.combine_op == 'add':
            cell = self._var_add(o1, o2)
        elif self.combine_op == 'mul':
            cell = self._var_mul(o1, o2)
        else:
            raise ValueError('Cell `combine_op` must be in [`add`, `concat]')

        return cell

    def _get_name(self, var):
        if hasattr(var, 'cell') and var.cell is not None:  # is a Cell
            var_name = str(var)
        else:
            var_name = var.name  # is a Variable
        return var_name

    def _get_ops(self, x):
        if hasattr(x, 'ops'):
            return x.ops

        return None

    def _check_special_ops(self, op, var):
        var_name = self._get_name(var)

        if op == 'noop' or op == 'no-op':
            return var_name

        return '%s(%s)' % (op, var_name)

    def _var_concat(self, x, y):
        return Cell('([%s ; %s])' % (x.name, y.name),
                    inputs=[self],
                    ops=self.ops,
                    combine_op='concat')

    def _var_dot(self, x, y):
        return Cell('(%s . %s)' % (x.name, y.name),
                    inputs=[self],
                    ops=self.ops,
                    combine_op='dot')

    def _var_add(self, x, y):
        return Cell('(%s + %s)' % (x.name, y.name),
                    inputs=[self],
                    ops=self.ops,
                    combine_op='add')

    def _var_mul(self, x, y):
        return Cell('(%s * %s)' % (x.name, y.name),
                    inputs=[self],
                    ops=self.ops,
                    combine_op='mul')

    def __repr__(self):
        if self.cell is not None:
            return '{%s}' % (str(self.cell.name))
        else:
            return self.name