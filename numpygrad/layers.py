import numpy as np
from numpygrad.tensor import Tensor

class Layer(object):

    def __init__(self):
        self._parameters = []

    def forward(self, input, *args, **kwargs):
        raise NotImplementedError()

    def __call__(self, input, *args, **kwargs):
        if not isinstance(input, Tensor):
            input = Tensor(input, autograd=True)

        return self.forward(input, *args, **kwargs)

    @property
    def parameters(self):
        return self._parameters

    def __setattr__(self, key, value):
        super(Layer, self).__setattr__(key, value)

        if isinstance(value, Tensor) and value.autograd:
            self._parameters.append(value)

        elif isinstance(value, Layer):
            self._parameters.extend(value.parameters)


class Sequential(Layer):

    def __init__(self, layers=None):
        super(Sequential, self).__init__()

        if layers is None:
            layers = []

        self.layers = layers

        for l in layers:
            self._parameters.extend(l.parameters)

    def add(self, layer):
        self.layers.append(layer)
        self._parameters.extend(layer.parameters)

    def forward(self, input, **kwargs):
        x = input

        for layer in self.layers:
            x = layer(x)

        return x


class Dense(Layer):

    def __init__(self, n_in, n_out):
        super(Dense, self).__init__()

        W = np.random.uniform(size=(n_in, n_out)) * np.sqrt(2. / n_in)
        b = np.zeros(n_out)

        self.w = Tensor(W, autograd=True)
        self.b = Tensor(b, autograd=True)

    def forward(self, input, **kwargs):
        out = input.dot(self.w) + self.b.expand(axis=0, repeats=len(input.data))
        return out


class Embedding(Layer):

    def __init__(self, vocab_size, dim):
        super(Embedding, self).__init__()

        self.vocab_size = vocab_size
        self.dim = dim

        # this random initialiation style is just a convention from word2vec
        w = np.random.rand(vocab_size, dim) - 0.5 / dim
        self.weight = Tensor(w, autograd=True)

    def forward(self, input, *args, **kwargs):
        return self.weight.index_select(input)
