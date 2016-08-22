import numpy as np
import theano.tensor as T
from theano import *

def R(rho, a, b, m, n1, n2, n3):
    rho_ = T.vector('rho')
    a_ = T.scalar('a')
    b_ = T.scalar('b')
    m_ = T.scalar('m')
    n1_ = T.scalar('n1')
    n2_ = T.scalar('n2')
    n3_ = T.scalar('n3')

    r = np.abs(np.abs(T.cos(m_ * rho_ / 4)) / a_) ** n2_ + np.abs(np.abs(T.sin(m_ * rho_ / 4)) / b_) ** n3_
    r = np.abs(r) ** (-1 / n1_)
    func = function([rho_, a_, b_, m_, n1_, n2_, n3_], [r], allow_input_downcast=True)
    return func(rho, a, b, m, n1, n2, n3)

def generalized_R(rho, a, b, y, z, n1, n2, n3):
    rho_ = T.scalar('rho')
    a_ = T.scalar('a')
    b_ = T.scalar('b')
    y_ = T.scalar('y')
    z_ = T.scalar('z')
    n1_ = T.scalar('n1')
    n2_ = T.scalar('n2')
    n3_ = T.scalar('n3')

    r = np.abs(T.cos(y_ * rho_ / 4) / a_) ** n2_ + np.abs(T.sin(z_ * rho_ / 4) / b_) ** n3_
    r = np.abs(r) ** (-1 / n1_)
    func = function([rho_, a_, b_, y_, z_, n1_, n2_, n3_], [r], allow_input_downcast=True)
    return func(rho, a, b, y, z, n1, n2, n3)

def xy(rho, a, b, m, n1, n2, n3):
    rho_ = T.vector('rho')
    r = R(rho, a, b, m, n1, n2, n3)

    x = r * T.cos(rho_)
    y = r * T.sin(rho_)
    func = function([rho_], [x, y], allow_input_downcast=True)
    vals = func(rho)
    return vals[0].flatten(), vals[1].flatten()

def xy_general(rho, a, b, y, z, n1, n2, n3):
    rho_ = T.vector('rho')
    r = generalized_R(rho, a, b, y, z, n1, n2, n3)

    x = r * T.cos(rho_)
    y = r * T.sin(rho_)
    func = function([rho_], [x, y], allow_input_downcast=True)
    vals = func(rho)
    return vals[0].flatten(), vals[1].flatten()

def xyz(R1, theta, R2, phi):
    theta_ = T.scalar('theta')
    phi_ = T.scalar('phi')

    x = R1 * T.cos(theta_) * R2 * T.cos(phi_)
    y = R2 * T.sin(theta_) * R2 * T.cos(phi_)
    z = R2 * T.sin(phi_)
    func = function([theta_, phi_], [x, y, z], allow_input_downcast=True)
    vals = func(theta, phi)
    return vals[0].flatten(), vals[1].flatten(), vals[2].flatten()

def xyz2(theta, a, b, m, n1, n2, n3, rho, a2, b2, m2, n4, n5, n6):
    theta_ = T.scalar('theta')
    rho_ = T.scalar('rho')
    R1 = R(theta, a, b, m, n1, n2, n3)
    R2 = R(rho, a2, b2, m2, n4, n5, n6)

    x = R1 * T.cos(theta_) * R2 * T.cos(rho_)
    y = R1 * T.sin(theta_) * R2 * T.cos(rho_)
    z = R2 * T.sin(rho_)
    func = function([theta_, rho_], [x, y, z], allow_input_downcast=True)
    vals = func(theta, rho)
    return vals[0].flatten(), vals[1].flatten(), vals[2].flatten()

if __name__ == "__main__":
    u = np.arange(0, 2 * np.pi, 0.001)

    # Ordinary formula
    vals = xy(u, 1, 1, 6, 1, 7, 8)
    x = vals[0]
    y = vals[1]

    import seaborn as sns
    sns.set_style("white")

    sns.plt.plot(x, y)
    sns.plt.show()