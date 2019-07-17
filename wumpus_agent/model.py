
from keras.models import Sequential
from keras.layers import Flatten, Dense
from keras.optimizers import Adam

data_members = 57
nb_actions = 6
hidden_size = 100

def build_model():
    model = Sequential()
    model.add(Dense(hidden_size, activation='relu', input_shape=(data_members,)))
    model.add(Dense(hidden_size, activation='relu'))
    model.add(Dense(nb_actions))
    model.compile(Adam(lr=1e-3), "diff")

    return model



