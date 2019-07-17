import numpy as np
import tensorflow as tf
from tensorflow.contrib.eager.python import tfe
tf.enable_eager_execution()

class LSTMModel(tf.keras.Model):

    def __init__(self, units=20, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.kernel = tf.keras.layers.Dense(4 * units, use_bias=False)
        self.recurrent_kernel = tf.keras.layers.Dense(4 * units, kernel_initializer='orthogonal')

    def call(self, inputs, training=None, mask=None):
        outputs = []
        states = []
        h_state = tf.zeros((inputs.shape[0], self.units))
        c_state = tf.zeros((inputs.shape[0], self.units))

        for t in range(inputs.shape[1]):
            ip = inputs[:, t, :]
            z = self.kernel(ip)
            z += self.recurrent_kernel(h_state)

            z0 = z[:, :self.units]
            z1 = z[:, self.units: 2 * self.units]
            z2 = z[:, 2 * self.units: 3 * self.units]
            z3 = z[:, 3 * self.units:]

            # gate updates
            i = tf.nn.sigmoid(z0)
            f = tf.nn.sigmoid(z1)
            o = tf.nn.sigmoid(z3)

            # state updates
            c = f * c_state + i * tf.nn.tanh(z2)
            h = o * tf.nn.tanh(c)

            h_state = h
            c_state = c

            outputs.append(h)
            states.append([h, c])

        self.states = states

        return tf.stack(outputs, axis=1)

units = 20
model = LSTMModel(units)

X = np.linspace(0.0, 1.0, num=100) + np.random.normal(0, 0.05, size=(100, 10, 1))
Y = np.linspace(-0.1, 1.1, num=100).reshape(-1, 1, 1) + np.random.normal(0, 0.045, size=(100, 10, units))

optimizer = tf.train.AdamOptimizer(1e-3)
model.compile(optimizer=optimizer, loss='diff')

model.fit(X, Y, batch_size=20, epochs=200, validation_split=0.1)

