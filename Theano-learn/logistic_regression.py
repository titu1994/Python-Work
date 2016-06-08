import theano.tensor as T
from theano import *
import numpy as np

rng = np.random
import sklearn.metrics as metrics

N = 400
features = 784

# generate a dataset: D = (input_values, target_class)
D = (rng.randn(N, features), rng.randint(size=N, low=0, high=2))
training_steps = 10000

x = T.fmatrix('x')
y = T.fvector('y')

# initialize the weight vector w randomly
#
# this and the following bias variable b
# are shared so they keep their values
# between training iterations (updates)
w = shared(rng.random(features) - 0.5, name="w")

# initialize the bias term
b = shared(0., name="b")

print("Initial model:")
print(w.get_value())
print(b.get_value())

p_1 = 1 / (1 + T.exp(-T.dot(x, w) - b))   # Probability that target = 1
prediction = p_1 > 0.5                    # The prediction thresholded
xent = -y * T.log(p_1) - (1-y) * T.log(1-p_1) # Cross-entropy loss function
cost = xent.mean() + 0.01 * (w ** 2).sum()# The cost to minimize
gw, gb = T.grad(cost, [w, b])             # Compute the gradient of the cost
                                          # w.r.t weight vector w and
                                          # bias term b
                                          # (we shall return to this in a
                                          # following section of this tutorial)

# Compile
train = function(
          inputs=[x,y],
          outputs=[prediction, xent],
          updates=((w, w - 0.1 * gw), (b, b - 0.1 * gb)), allow_input_downcast=True)
predict = function(inputs=[x], outputs=prediction, allow_input_downcast=True)

# Train
for i in range(training_steps):
    pred, err = train(D[0], D[1])
    if i % 100 == 0: print((i / 100.), '% done.')

print("Final model:")
print(w.get_value())
print(b.get_value())
print("target values for D:")
print(D[1])
print("prediction on D:")
preds = predict(D[0])
print(preds)

print("Accuracy :", metrics.accuracy_score(D[1], preds))