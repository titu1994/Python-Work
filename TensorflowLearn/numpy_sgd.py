import numpy as np
np.random.seed(0)

x = np.random.uniform(-1.0, 1.0)

def f(x):
    return np.exp(-np.logaddexp(0, -x)) + np.tanh(x)

def df_dx(x):
    z1 = np.exp(-np.logaddexp(0, -x))
    z2 = np.tanh(x)
    return z1 * (1 - z1) + (1 - z2 ** 2)

learning_rate = 1
beta1 = 0.9
beta2 = 0.999
epsilon = 1e-8
iter = 1

M = 0.
R = 0.
R_hat = 0.

for i in range(10000):
    for j in range(50):
        dx = df_dx(x)

        M = beta1 * M + (1 - beta1) * dx
        R = beta2 * R + (1 - beta2) * dx ** 2

        R_hat = np.maximum(R_hat, R)

        lr = learning_rate / (np.sqrt(R_hat + epsilon))

        x -= lr * (M)

        R = R_hat
        iter += 1
        #x -= learning_rate * df_dx(x)

    if i % 50 == 0:
        print(i, "Optimal Value : ", f(x), "Val (X) : ", x)