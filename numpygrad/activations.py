import numpy as np
from numpygrad.layers import Layer


class _Activation(Layer):

    def __init__(self):
        super(_Activation, self).__init__()

    def forward(self, input, **kwargs):
        raise NotImplementedError()


class Sigmoid(_Activation):

    def forward(self, input, **kwargs):
        return input.sigmoid()


class Tanh(_Activation):

    def forward(self, input, **kwargs):
        return input.tanh()


class ReLU(_Activation):

    def forward(self, input, **kwargs):
        return input.relu()
