import numpy as np

def R(rho, a, b, m, n1, n2, n3):
    r = np.abs(np.abs(np.cos(m * rho / 4)) / a) ** n2 + np.abs(np.abs(np.sin(m * rho / 4)) / b) ** n3
    r = np.abs(r) ** (-1 / n1)
    return r

def generalized_R(rho, a, b, y, z, n1, n2, n3):
    r = np.abs(np.cos(y * rho / 4) / a) ** n2 + np.abs(np.sin(z * rho / 4) / b) ** n3
    r = np.abs(r) ** (-1 / n1)
    return r

def xy(rho, a, b, m, n1, n2, n3):
    x = R(rho, a, b, m, n1, n2, n3) * np.cos(rho)
    y = R(rho, a, b, m, n1, n2, n3) * np.sin(rho)
    return x, y

def xy_general(rho, a, b, y, z, n1, n2, n3):
    x = generalized_R(rho, a, b, y, z, n1, n2, n3) * np.cos(rho)
    y = generalized_R(rho, a, b, y, z, n1, n2, n3) * np.sin(rho)
    return x, y

def xyz(R1, theta, R2, phi):
    x = R1 * np.cos(theta) * R2 * np.cos(phi)
    y = R2 * np.sin(theta) * R2 * np.cos(phi)
    z = R2 * np.sin(phi)
    return x, y, z

def xyz2(theta, a, b, m, n1, n2, n3, rho, a2, b2, m2, n4, n5, n6):
    x = R(theta, a, b, m, n1, n2, n3) * np.cos(theta) * R(rho, a2, b2, m2, n4, n5, n6) * np.cos(rho)
    y = R(theta, a, b, m, n1, n2, n3) * np.sin(theta) * R(rho, a2, b2, m2, n4, n5, n6) * np.cos(rho)
    z = R(rho, a2, b2, m2, n4, n5, n6) * np.sin(rho)
    return x, y, z

if __name__ == "__main__":
    u = np.arange(0, 2 * np.pi, 0.001)

    # Ordinary formula
    vals = [xy(ui, 1, 1, 6, 1, 7, 8) for ui in u]
    x, y = [], []

    for v in vals:
        x.append(v[0])
        y.append(v[1])

    import seaborn as sns
    sns.set_style("white")

    sns.plt.plot(x, y)
    sns.plt.show()

    # Generalized formula
    vals = [xy_general(ui, 1, 1, 8, 40, -0.2, 1, 1) for ui in u]
    x.clear()
    y.clear()

    for v in vals:
        x.append(v[0])
        y.append(v[1])

    sns.plt.plot(x, y)
    sns.plt.show()
