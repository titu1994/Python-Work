import numpy as np
from collections import defaultdict
from uuid import uuid4
from typing import List, Union


class Tensor(object):

    def __init__(self, data,
                 autograd=False,
                 creators=None,
                 creation_op=None,
                 id=None):

        self.data = np.array(data)
        self.autograd = autograd
        self.creators = creators  # type: List[Tensor]
        self.creation_op = creation_op
        self.id = id if id is not None else uuid4()
        self.grad = None

        self.children = defaultdict(int)

        if creators is not None:
            for c in creators:  # type: Tensor
                if self.id not in c.children:
                    c.children[self.id] += 1

    """ Matrix Multiplication ops """

    def dot(self, x):
        if self.autograd:
            return Tensor(self.data.dot(x.data),
                          autograd=True,
                          creators=[self, x],
                          creation_op='dot')

        return Tensor(self.data.dot(x.data))

    def transpose(self):
        if self.autograd:
            return Tensor(self.data.transpose(),
                          autograd=True,
                          creators=[self],
                          creation_op='transpose')

        return Tensor(self.data.transpose())

    def expand(self, axis=-1, repeats=1):
        if axis < 0:
            axis = self._resolve_axis(axis) + 1

        indices = list(range(len(self.shape)))
        indices.insert(axis, len(self.shape))

        data = self.data.repeat(repeats)
        data = data.reshape(list(self.shape) + [repeats])
        data = data.transpose(indices)

        if self.autograd:
            return Tensor(data,
                          autograd=True,
                          creators=[self],
                          creation_op='expand_%d' % (axis))

        return Tensor(data)

    def index_select(self, indices):
        if self.autograd:
            tensor = Tensor(self.data[indices.data],
                            autograd=True,
                            creators=[self],
                            creation_op='index_select')
            tensor.index_select_indices = indices
            return tensor

        return Tensor(self.data[indices.data])

    def cross_entropy(self, target_indices):

        temp = np.exp(self.data)
        softmax_output = temp / np.sum(temp,
                                       axis=len(self.data.shape) - 1,
                                       keepdims=True)

        t = target_indices.data.flatten()
        p = softmax_output.reshape(len(t), -1)
        target_dist = np.eye(p.shape[1])[t]
        loss = -(np.log(p) * (target_dist)).sum(1).mean()

        if self.autograd:
            out = Tensor(loss,
                         autograd=True,
                         creators=[self],
                         creation_op="cross_entropy")
            out.softmax_output = softmax_output
            out.target_dist = target_dist

            return out

        return Tensor(loss)

    """ Reduction Ops """

    def sum(self, axis=-1):
        axis = self._resolve_axis(axis)

        if self.autograd:
            return Tensor(np.sum(self.data, axis=axis),
                          autograd=True,
                          creators=[self],
                          creation_op='sum_%d' % (axis))

        return Tensor(np.sum(self.data, axis=axis))

    def mean(self, axis=-1):
        axis = self._resolve_axis(axis)

        if self.autograd:
            return Tensor(np.mean(self.data, axis=axis),
                          autograd=True,
                          creators=[self],
                          creation_op='mean_%d' % (axis))

        return Tensor(np.mean(self.data, axis=axis))

    """ Inlined Activation functions """

    def sigmoid(self):
        if self.autograd:
            return Tensor(1. / (1. + np.exp(-self.data)),
                          autograd=True,
                          creators=[self],
                          creation_op='sigmoid')

        return Tensor(1. / (1. + np.exp(-self.data)))

    def tanh(self):
        if self.autograd:
            return Tensor(np.tanh(self.data),
                          autograd=True,
                          creators=[self],
                          creation_op='tanh')

    def relu(self):
        if self.autograd:
            return Tensor(np.maximum(0., self.data),
                          autograd=True,
                          creators=[self],
                          creation_op='relu')

        return Tensor(np.maximum(0., self.data))

    """ Gradient ops """

    def backward(self, grad=None, grad_origin=None):
        # type: (Union[np.ndarray, Tensor], Tensor) -> None
        if self.autograd:
            if grad is None:
                grad = Tensor(np.ones_like(self.data))

            if grad_origin is not None:
                if self.children[grad_origin.id] == 0:
                    raise Exception("Cannot back propagate more than once.")
                else:
                    self.children[grad_origin.id] -= 1

            if self.grad is None:
                self.grad = grad
            else:
                self.grad += grad

            # grads must not have grads of their own
            assert grad.autograd == False

            # only continue backpropping if there's something to
            # backprop into and if all gradients (from children)
            # are accounted for override waiting for children if
            # "backprop" was called on this variable directly
            if self.creators is not None and (
                    self._all_children_grads_accounted_for() or grad_origin is None):
                self._resolve_backprop()

    def _resolve_backprop(self):
        if self.creation_op in ['negation', '_var_add', 'sub', 'mul']:
            self._resolve_arithmatic_backprop()

        if self.creation_op in ['dot', 'transpose', 'index_select', 'cross_entropy'] or 'expand' in self.creation_op:
            self._resolve_matrix_multiplication_backprop()

        if self.creation_op in ['sigmoid', 'tanh', 'relu']:
            self._resolve_activation_backprop()

        reduction_ops = ['sum', 'mean']
        for op in reduction_ops:
            if op in self.creation_op:
                self._resolve_reduction_backprop()

    def _resolve_arithmatic_backprop(self):
        if self.creation_op == 'negation':
            self.creators[0].backward(-self.grad)

        if self.creation_op == '_var_add':
            self.creators[0].backward(self.grad, self)
            self.creators[1].backward(self.grad, self)

        if self.creation_op == 'sub':
            self.creators[0].backward(Tensor(self.grad.data), self)
            self.creators[1].backward(Tensor(-self.grad.data), self)

        if self.creation_op == 'mul':
            self.creators[0].backward(self.grad * self.creators[1], self)
            self.creators[1].backward(self.grad * self.creators[0], self)

    def _resolve_matrix_multiplication_backprop(self):
        if self.creation_op == 'dot':
            c0 = self.creators[0]
            c1 = self.creators[1]

            grad = self.grad.dot(c1.transpose())
            c0.backward(grad)

            grad = self.grad.transpose().dot(c0).transpose()
            c1.backward(grad)

        if self.creation_op == 'transpose':
            self.creators[0].backward(self.grad.transpose())

        if 'expand' in self.creation_op:
            axis = int(self.creation_op.split('_')[1])
            self.creators[0].backward(self.grad.sum(axis))

        if self.creation_op == "index_select":
            new_grad = np.zeros_like(self.creators[0].data)
            indices_ = self.index_select_indices.data.flatten()
            grad_ = self.grad.data.reshape(len(indices_), -1)

            for i in range(len(indices_)):
                new_grad[indices_[i]] += grad_[i]

            self.creators[0].backward(Tensor(new_grad))

        if self.creation_op == 'cross_entropy':
            dx = self.softmax_output - self.target_dist
            self.creators[0].backward(Tensor(dx))

    def _resolve_activation_backprop(self):
        if self.creation_op == 'sigmoid':
            ones = Tensor(np.ones_like(self.grad.data))
            self.creators[0].backward(self.grad * (self * (ones - self)))

        if self.creation_op == 'tanh':
            ones = Tensor(np.ones_like(self.grad.data))
            self.creators[0].backward(self.grad * (ones - (self * self)))

        if self.creation_op == 'relu':
            mask = np.where(self.grad.data >= 0., 1., 0.)
            self.creators[0].backward(self.grad * mask)

    def _resolve_reduction_backprop(self):
        op, axis = self.creation_op.split('_')
        axis = int(axis)
        repeats = self.creators[0].data.shape[axis]

        if op == 'sum':
            self.creators[0].backward(self.grad.expand(axis, repeats))

        if op == 'mean':
            self.creators[0].backward(self.grad.expand(axis, repeats))

    def _all_children_grads_accounted_for(self):
        for id, count in self.children.items():
            if count != 0:
                return False

        return True

    """ Arithmetic ops """

    def __neg__(self):
        if self.autograd:
            return Tensor(-self.data,
                          autograd=True,
                          creators=[self],
                          creation_op='negation')

        return Tensor(-self.data)

    def __add__(self, other):
        if self.autograd and other.autograd:
            return Tensor(self.data + other.data,
                          autograd=True,
                          creators=[self, other],
                          creation_op='_var_add')

        return Tensor(self.data + other.data)

    def __sub__(self, other):
        if self.autograd and other.autograd:
            return Tensor(self.data - other.data,
                          autograd=True,
                          creators=[self, other],
                          creation_op='sub')

        return Tensor(self.data - other.data)

    def __mul__(self, other):
        if self.autograd and other.autograd:
            return Tensor(self.data * other.data,
                          autograd=True,
                          creators=[self, other],
                          creation_op='mul')

        return Tensor(self.data * other.data)

    """ Utility ops """

    @property
    def shape(self):
        return self.data.shape

    def _resolve_axis(self, axis):
        if axis < 0:
            axis = len(self.shape) + axis

        return axis

    def __repr__(self):
        return str(self.data.__repr__())

    def __str__(self):
        return str(self.data.__str__())
