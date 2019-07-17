import numpy as np
import tensorflow as tf
from tfdiffeq import odeint
from tfdiffeq.models.dense_odenet import ODENet
import matplotlib.pyplot as plt

from tfdiffeq_examples.utils.data_loader import load_dataset

tf.compat.v2.enable_v2_behavior()

X_train, y_train, X_test, y_test = load_dataset('adiac', normalize_timeseries=True)
print()

data_dim = X_train.shape[-1]

model = ODENet(data_dim, hidden_dim=1, output_dim=data_dim, augment_dim=1,
               non_linearity='linear', time_dependent=True, tol=1e-3)

optimizer = tf.train.AdamOptimizer(1e-2)
criterion = tf.keras.losses.MeanAbsoluteError()

BATCH_SIZE = 128
EPOCHS = 100

X_train = tf.constant(X_train)
y_train = tf.constant(y_train)
X_test = tf.constant(X_test)
y_test = tf.constant(y_test)
global_step = tf.Variable(0, dtype=tf.int64, trainable=False)

for epoch in range(EPOCHS):
    with tf.GradientTape() as tape:
        outputs = model(X_train)
        loss = criterion(y_train, outputs)

    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables), global_step)

    print("Epoch %d: Loss = %0.5f" % (epoch + 1, loss.numpy().mean()))

x = X_test[0]
x_pred = model(tf.expand_dims(x, 0))

plt.plot(x, label='original')
plt.plot(x_pred[0], label='generated', alpha=0.5)
plt.legend()
plt.show()










