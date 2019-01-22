import numpy as np

class _Optimizer(object):

    def __init__(self, parameters):
        self.parameters = parameters

    def reset(self):
        for p in self.parameters:
            p.grad.data *= 0

    def step(self, zero_grad=True):
        raise NotImplementedError()


class SGD(_Optimizer):

    def __init__(self, parameters, lr=0.1):
        super(SGD, self).__init__(parameters)
        self.lr = lr

    def step(self, zero_grad=True):
        for p in self.parameters:
            p.data -= p.grad.data * self.lr

            if zero_grad:
                p.grad.data *= 0.


class Adam(_Optimizer):

    def __init__(self, parameters, lr=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        super(Adam, self).__init__(parameters)
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon

        self._prepare_weights()

    def step(self, zero_grad=True):
        self.t += 1

        for p, m, v in zip(self.parameters, self.M, self.V):
            grad = p.grad.data
            m = self.beta1 * m + (1. - self.beta1) * grad
            v = self.beta2 * v + (1. - self.beta2) * (grad * grad)
            m_hat = m / (1. - (self.beta1 ** self.t))
            v_hat = v / (1. - (self.beta2 ** self.t))

            p.data -= self.lr * (m_hat / (np.sqrt(v_hat) + self.epsilon))

            if zero_grad:
                p.grad.data *= 0.


    def _prepare_weights(self):
        self.M = []
        self.V = []
        self.t = 0

        for p in self.parameters:
            m = np.zeros_like(p)
            v = np.zeros_like(p)

            self.M.append(m)
            self.V.append(v)
