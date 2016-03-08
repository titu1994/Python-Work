from sklearn.datasets import load_boston
import sklearn.metrics as metrics
import keras.layers.core as core
import keras.models as models
from sklearn.linear_model import LinearRegression
from keras.callbacks import EarlyStopping


if __name__ == "__main__":

    boston = load_boston()
    X = boston.data
    y = boston.target
    nEpochs = 200

    model = models.Sequential()
    model.add(core.Dense(1000, activation="relu", input_shape=(13,))) # 200
    model.add(core.Dropout(0.2))
    model.add(core.Dense(1000, activation="relu")) # 1000
    model.add(core.Dropout(0.2))
    model.add(core.Dense(1000, activation="relu")) # 1000
    model.add(core.Dropout(0.2))
    model.add(core.Dense(75, activation="relu")) # 200
    model.add(core.Dense(1))

    model.summary()

    model.compile(loss="mse", optimizer="adam")
    #callbacks=[EarlyStopping( patience=10)]
    model.fit(X, y, nb_epoch=nEpochs, verbose=1, )

    yPreds = model.predict(X)
    mse = metrics.mean_squared_error(y, yPreds)

    print("Neural Network : Mean Squared Error : ", mse)

    lr = LinearRegression()
    lr.fit(X, y)

    lrYPred = lr.predict(X)
    lrmse = metrics.mean_squared_error(y, lrYPred)

    print("Linear Regression : Mean Squared Error : ", lrmse)

    if lrmse >= mse: print("Neural Network >= Linear Regression. Better than standard LR")
    else: print("Neural Network < Linear Regression. Worse than standard LR")
