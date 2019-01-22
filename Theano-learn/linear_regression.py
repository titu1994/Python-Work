import theano.tensor as T
from theano import *
import random

x = T.vector('x')
y = T.vector('y')
y_pred = T.vector('y_pred')

meanX = T.mean(x)
meanY = T.mean(y)

beta = T.sum((x - meanX) * (y - meanY)) / T.sum((x - meanX) ** 2)
alpha = meanY - beta * meanX
_predict = beta * x + alpha

_mse = T.sum((y - y_pred) ** 2)

compute_vals = function([x, y], [alpha, beta], allow_input_downcast=True)
predict = function([x, alpha, beta], [_predict], allow_input_downcast=True)
mse = function([y, y_pred], [_mse], allow_input_downcast=True)

if __name__ == "__main__":
    import numpy as np
    random.seed(1)

    X = [i * 0.1 for i in range(1, 101)]
    y = [i + random.gauss(0, 0.33) for i in X]

    alpha, beta = compute_vals(X, y)
    print("lr : ", alpha, " - beta : ", beta)

    preds = predict(X, alpha, beta)[0]
    error = mse(y, preds)[0]

    print("Mean Squared Error : ", error)

    import seaborn as sns
    sns.set_style('white')

    sns.plt.scatter(X, y,)
    sns.plt.plot(X, preds, )
    sns.plt.show()
