import numpy as np
import os

from keras.models import Model
from keras.layers import Input, Embedding, Dense
from keras.layers import LSTM, concatenate, Flatten, Bidirectional, LeakyReLU

from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from keras.utils import to_categorical
from keras.metrics import top_k_categorical_accuracy
from keras import backend as K
from keras.applications.imagenet_utils import decode_predictions
from sklearn.model_selection import train_test_split

from AirBnB.data_utils import load_prepared_dataset, load_embedding_matrix, decode_predictions, get_class_decoder
from AirBnB.data_utils import MAX_SEQUENCE_LENGTH, EMBEDDING_DIM, ENTITIY_EMBEDDING_DIM

if not os.path.exists('weights/'):
    os.makedirs('weights/')

BATCHSIZE = 128
EPOCHS = 15

WEIGHTS_PATH = 'weights/lstm_model.h5'


data, labels, countries, devices, word_index, max_num_countries, max_num_devices = load_prepared_dataset()
sentence_embedding_matrix = load_embedding_matrix()

val = np.sum(labels) / (len(labels) * 552)
print("\n\n\nMajority class (all 0s) : ", 1. - val)

num_classes = labels.shape[-1]  # Here, 551 "tags" as classes
print("Number of classes : ", num_classes)
MAX_NUM_WORDS = sentence_embedding_matrix.shape[0]

# train test split
sentence_train, sentence_test, labels_train, labels_test, countries_train, countries_test, \
            devices_train, devices_test = train_test_split(data, labels, countries, devices, test_size=0.1)


sentence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
countries_input = Input(shape=(1,), dtype='int32')
devices_input = Input(shape=(1,), dtype='int32')

# sentence embedding ; load glove embeddings and dont train
sentence_embedding = Embedding(MAX_NUM_WORDS, EMBEDDING_DIM,
                               weights=[sentence_embedding_matrix],
                               trainable=False)

# countries embedding
countries_embedding = Embedding(max_num_countries, ENTITIY_EMBEDDING_DIM)

# devices embedding
devices_embedding = Embedding(max_num_devices, ENTITIY_EMBEDDING_DIM)

# create the layers
sentence_emb = sentence_embedding(sentence_input)
countries_emb = countries_embedding(countries_input)
devices_emb = devices_embedding(devices_input)

# pass sentence through an LSTM
sentence_emb = Bidirectional(LSTM(64, return_sequences=True))(sentence_emb)

# Flatten the entity embeddings
sentence_emb = Flatten()(sentence_emb)
countries_emb = Flatten()(countries_emb)
devices_emb = Flatten()(devices_emb)

# concatenate the embeddings (concatenate at the end)
joint_embeddings = concatenate([sentence_emb, countries_emb, devices_emb], axis=-1)

# pass this to an MLP
x = Dense(1024)(joint_embeddings)
x = LeakyReLU()(x)

x = Dense(num_classes, activation='softmax')(x)

# build the model
model = Model([sentence_input, countries_input, devices_input], outputs=x)
model.summary()

#loss_scale = 1e-3
#model.add_loss(loss_scale * entropy_loss(x))

# Compile the model
optimizer = Adam(lr=1e-3)
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

# train the model
checkpoint = ModelCheckpoint(WEIGHTS_PATH, monitor='val_acc', verbose=1, save_best_only=True,
                             save_weights_only=True)

model.fit(x=[sentence_train, countries_train, devices_train], y=labels_train,
          batch_size=BATCHSIZE, epochs=EPOCHS, verbose=1,
          callbacks=[checkpoint],
          validation_data=([sentence_test, countries_test, devices_test], labels_test))

# evaluate the results one last time
model.load_weights(WEIGHTS_PATH)

# scores = model.evaluate(x=[sentence_test, countries_test, devices_test], y=labels_test, batch_size=BATCHSIZE, verbose=1)
#
# print("Final result :")
# for name, score in zip(model.metrics_names, scores):
#     print(name, score)

predictions = model.predict(x=[sentence_test, countries_test, devices_test], batch_size=BATCHSIZE, verbose=1)

k = 5
top_accuracy = top_k_categorical_accuracy(labels_test, predictions, k=k)
top_accuracy = K.get_session().run(top_accuracy)

print()
print("Top %d Accuracy : " % k, top_accuracy)
print()

ground_labels = decode_predictions(labels_test)
label_decoder = get_class_decoder()

results = []

for pred in predictions:
    top_k_preds_indices = pred.argsort()[-k:][::-1]
    result = [(label_decoder.classes_[i], pred[i]) for i in top_k_preds_indices]
    result.sort(key=lambda x: x[-1], reverse=True)
    results.append(result)

for gt, pt in zip(ground_labels, results):
    data = 'GT : %s | ' % gt
    print(data, pt)
    print()
