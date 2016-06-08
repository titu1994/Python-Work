import keras.layers.core as core
import keras.models as models
import keras.callbacks as callbacks
import keras.utils.np_utils as kutils

from sklearn.cross_validation import train_test_split

from NeuralNetworks.SpiralGenerator import generateSpiralDataframe, generateSpiralDataset

if __name__ == "__main__":

    points = generateSpiralDataset(numSamples=100, noise=0.3)
    df = generateSpiralDataframe(points)

    train, test = train_test_split(df, train_size=0.7)
    #train = train[["label", "x", "y"]]
    train = train.values

    #test = test[["label", "x", "y"]]
    test = test.values

    trainX = train[:, 1:]
    trainY = train[:, 0]
    trainY = kutils.to_categorical(trainY)

    testX = test[:, 1:]
    testY = test[:, 0]
    testY = kutils.to_categorical(testY)

    # Variables
    print(train.shape, ' ', trainX.shape)
    nbFeatures = trainX.shape[1]
    nbClasses = trainY.shape[1]

    batchSize = 32
    epochs = 500

    model = models.Sequential()

    model.add(core.Dense(8, input_shape=(nbFeatures,), activation="tanh"))
    model.add(core.Dense(8, activation="tanh"))

    model.add(core.Dense(nbClasses, activation="softmax"))

    model.compile(optimizer='adadelta', loss="binary_crossentropy", metrics=["accuracy"])

    model.fit(trainX, trainY, batch_size=batchSize, nb_epoch=epochs, validation_data=(testX, testY))

