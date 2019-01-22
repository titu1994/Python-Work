import numpy as np
from numpygrad.layers import Layer


class _Loss(Layer):

    def __init__(self):
        super(_Loss, self).__init__()

    def __call__(self, target, prediction, **kwargs):
        return super(_Loss, self).__call__(target, prediction, **kwargs)

    def forward(self, target, predicted, **kwargs):
        raise NotImplementedError()


class MSELoss(_Loss):

    def __init__(self):
        super(MSELoss, self).__init__()

    def forward(self, target, predicted, **kwargs):
        return ((target - predicted) * (target - predicted)).sum(0)


class CrossEntropyLoss(_Loss):

    def __init__(self):
        super(CrossEntropyLoss, self).__init__()

    def forward(self, input, target, **kwargs):
        return input.cross_entropy(target)
