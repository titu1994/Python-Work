from keras.models import Model
from keras.layers import Input, Dense, BatchNormalization, Activation
from keras.losses import categorical_crossentropy, mean_squared_error
from keras import backend as K

img_size = 32
input_dim = 3072

alpha = 0.5

ip = Input(shape=(input_dim,))

x = Dense(128)(ip)
x = BatchNormalization()(x, training=False)
x = Activation('relu')(x)

x = Dense(64)(x)
x = BatchNormalization()(x, training=False)
x = Activation('relu')(x)

x = Dense(32)(x)
x = BatchNormalization()(x, training=False)
x = Activation('relu')(x)

head_1 = Dense(input_dim, activation='softmax', name='head1')(x)

x = Dense(64)(x)
x = BatchNormalization()(x, training=False)
x = Activation('relu')(x)

x = Dense(128)(x)
x = BatchNormalization()(x, training=False)
x = Activation('relu')(x)

head_2 = Dense(input_dim, activation='linear', name='head2')(x)  # for mse, change the dense layer

model = Model(ip, [head_1, head_2])

def categorical_loss(y_true, y_pred):
    return alpha * categorical_crossentropy(y_true, y_pred)

def rmse(y_true, y_pred):
    return (1. - alpha) * K.sqrt(mean_squared_error(y_true, y_pred))


# adding the second loss
model.compile('adam', loss=[categorical_loss, rmse])

model.summary()