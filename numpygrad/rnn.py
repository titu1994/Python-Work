import numpy as np
from numpygrad.tensor import Tensor
from numpygrad.layers import Layer, Dense
from numpygrad.activations import Sigmoid, Tanh, ReLU


class _RNN(Layer):

    def forward(self, input, states, **kwargs):
        return super(_RNN, self).forward(input, states, **kwargs)


class RNN(_RNN):

    def __init__(self, n_in, n_hidden, n_out, activation='sigmoid'):
        super(RNN, self).__init__()

        self.n_in = n_in
        self.n_out = n_out
        self.n_hidden = n_hidden

        if activation == 'sigmoid':
            self.activation = Sigmoid()
        elif activation == 'tanh':
            self.activation = Tanh()
        else:
            self.activation = ReLU()

        self.w_ih = Dense(n_in, n_hidden)
        self.w_hh = Dense(n_hidden, n_hidden)
        self.w_ho = Dense(n_hidden, n_out)

    def forward(self, input, states, **kwargs):
        from_prev_state = self.w_hh(states)
        combined = self.w_ih(input) + from_prev_state
        new_hidden = self.activation(combined)
        output = self.w_ho(new_hidden)

        return output, new_hidden

    def init_state(self, batch_size=1):
        return Tensor(np.zeros((batch_size, self.n_hidden)), autograd=True)


