import numpy as np
import pandas as pd

def randUniform(a, b):
    return np.random.random_sample() * (b - a) + a

def generateSpiralDataset(numSamples, noise=0.) -> list:
    n = int(numSamples / 2)
    points = []

    def spiral(deltaT, label):
        for i in range(n):
            r = i / n * 5
            t = 1.75 * i / n * 2 * np.pi + deltaT
            x = r * np.sin(t) + randUniform(-1, 1) * noise
            y = r * np.cos(t) + randUniform(-1, 1) * noise
            points.append((x, y, label))

    spiral(0, 1) # Positive Samples
    spiral(np.pi, -1) # Negative Samples
    return np.array(points)

def generateSpiralDataframe(data:np.array) -> pd.DataFrame:
    df = pd.DataFrame({'label': data[:, 2], 'x': data[:, 0], 'y': data[:, 1]})

    df["x2"] = df["x"] ** 2
    df["y2"] = df["y"] ** 2
    df["xy"] = df["x"] * df["y"]

    df["sinx"] = np.sin(df["x"])
    df["siny"] = np.sin(df["y"])

    return df

if __name__ == "__main__":
    import seaborn as sns
    sns.set_style("white")

    count = 1000
    points = generateSpiralDataset(count, noise=0.3)
    values = generateSpiralDataframe(points).values

    sns.plt.scatter(values[:count/2, 1], values[:count/2, 2], c="r")
    sns.plt.scatter(values[count/2:, 1], values[count/2:, 2], c="b")
    sns.plt.show()