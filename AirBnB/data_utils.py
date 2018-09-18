import os
import numpy as np
import pandas as pd
import pickle

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.metrics import binary_accuracy
from keras import backend as K

from sklearn.preprocessing import LabelEncoder, MultiLabelBinarizer, LabelBinarizer

np.random.seed(0)

MAX_NUM_WORDS = 25000
MAX_SEQUENCE_LENGTH = 100

EMBEDDING_DIM = 300
ENTITIY_EMBEDDING_DIM = 50


def entropy_loss(x):
    entropy = K.sum(x * K.log(x))
    return -entropy


def clean_country(df:pd.DataFrame):
    df['country'] = df['country'].map(lambda x: str(x).upper())

    unique = np.unique(df.country.values)
    max_num_countries = len(unique)
    print("Unique Countries : ", unique)
    print("Num Unique Countries : ", max_num_countries)
    print()

    countries = df['country'].values

    if os.path.exists('data/countries.pkl'):
        with open('data/countries.pkl', 'rb') as f:
            encoder = pickle.load(f)

    else:
        encoder = LabelEncoder()
        encoder.fit(countries)

        with open('data/countries.pkl', 'wb') as f:
            pickle.dump(encoder, f)

    # encode the values
    countries = encoder.transform(countries)

    return df, countries, max_num_countries


def clean_device(df:pd.DataFrame):
    df['device_type'] = df['device_type'].map(lambda x: str(x).lower())

    unique = np.unique(df.device_type.values)
    max_num_devices = len(unique)
    print("Unique Devices : ", unique)
    print("Num Unique Devices : ", max_num_devices)
    print()

    devices = df['device_type'].values

    if os.path.exists('data/devices.pkl'):
        with open('data/devices.pkl', 'rb') as f:
            encoder = pickle.load(f)

    else:
        encoder = LabelEncoder()
        encoder.fit(devices)

        with open('data/devices.pkl', 'wb') as f:
            pickle.dump(encoder, f)

    # encode the values
    devices = encoder.transform(devices)

    return df, devices, max_num_devices


def extract_text(df:pd.DataFrame):
    df['feedback'] = df['feedback'].map(lambda x: str(x).lower())

    return df.feedback.values


def prepare_labels(df:pd.DataFrame):
    df['tags'] = df['tags'].map(lambda x: str(x).lower())
    tags = df['tags'].values

    tags = [tag.split('|') for tag in tags]

    for i in range(len(tags)):
        tags[i] = [t.strip() for t in tags[i]]
        tags[i] = [t for t in tags[i] if t != '']
        tags[i] = tags[i][0]  # take only 1st tag

    unique_tags = set()
    for tag in tags:
        unique_tags.update(tag)

    print("all tags : ", sorted(unique_tags))

    if os.path.exists('data/label_binarizer.pkl'):
        with open('data/label_binarizer.pkl', 'rb') as f:
            binarizer = pickle.load(f)
    else:
        #binarizer = MultiLabelBinarizer()
        binarizer = LabelBinarizer()
        binarizer.fit(tags)

        with open('data/label_binarizer.pkl', 'wb') as f:
            pickle.dump(binarizer, f)

    # convert to indices
    tags = binarizer.transform(tags)

    print("Binerizer classes : ", binarizer.classes_)

    tags = np.array(tags, dtype='float32')

    count_list = []
    for i in range(len(tags)):
        count = int(np.sum(tags[i]))
        count_list.append(count)

    #print("Label counts : ", count_list)
    print()

    return tags


def prepare_word_tokenizer(texts):
    if not os.path.exists('data/tokenizer.pkl'): # check if a prepared tokenizer is available
        tokenizer = Tokenizer(num_words=MAX_NUM_WORDS)  # if not, create a new Tokenizer
        tokenizer.fit_on_texts(texts)  # prepare the word index map

        with open('data/tokenizer.pkl', 'wb') as f:
            pickle.dump(tokenizer, f)  # save the prepared tokenizer for fast access next time

        print('Saved tokenizer.pkl')
    else:
        with open('data/tokenizer.pkl', 'rb') as f:  # simply load the prepared tokenizer
            tokenizer = pickle.load(f)
            print('Loaded tokenizer.pkl')

    sequences = tokenizer.texts_to_sequences(texts)  # transform text into integer indices lists
    word_index = tokenizer.word_index  # obtain the word index map

    print('Average sequence length: {}'.format(np.mean(list(map(len, sequences)), dtype=int)))  # compute average sequence length
    print('Max sequence length: {}'.format(np.max(list(map(len, sequences)))))  # compute maximum sequence length

    data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)  # pad the sequence to the user defined max length

    return (data, word_index)


def generate_embedding_matrix(embedding_path, word_index, print_error_words=True):
    '''
    Either loads the created embedding matrix at run time, or uses the
    GLoVe 840B word embedding to create a mini initialized embedding matrix
    for use by Keras Embedding layers

    Args:
        embedding_path: path to the 840B word GLoVe Embeddings
        word_index: indices of all the words in the current corpus
        max_nb_words: maximum number of words in corpus
        embedding_dim: the size of the embedding dimension
        print_error_words: Optional, allows to print words from GLoVe
            that could not be parsed correctly.

    Returns:
        An Embedding matrix in numpy format
    '''
    if not os.path.exists('data/embedding_matrix.npy'):
        embeddings_index = {}
        error_words = []

        print("Creating embedding matrix")
        print("Loading : ", embedding_path)

        # read the entire GLoVe embedding matrix
        f = open(embedding_path, encoding='utf8')
        for line in f:
            values = line.split()
            word = values[0]
            try:
                coefs = np.asarray(values[1:], dtype='float32')
                embeddings_index[word] = coefs
            except Exception:
                error_words.append(word)

        f.close()

        # check for words that could not be loaded properly
        if len(error_words) > 0:
            print("%d words could not be added." % (len(error_words)))
            if print_error_words:
                print("Words are : \n", error_words)

        print('Preparing embedding matrix.')

        # prepare embedding matrix
        nb_words = min(MAX_NUM_WORDS, len(word_index))
        embedding_matrix = np.zeros((nb_words, EMBEDDING_DIM))
        for word, i in word_index.items():
            if i >= nb_words:
                continue
            embedding_vector = embeddings_index.get(word)
            if embedding_vector is not None:
                # words not found in embedding index will be all-zeros.
                embedding_matrix[i] = embedding_vector

        # save the constructed embedding matrix in a file for efficient loading next time
        np.save('data/embedding_matrix.npy', embedding_matrix)

        print('Saved embedding matrix')


def load_embedding_matrix():
    if not os.path.exists('data/embedding_matrix.npy'):
        raise FileNotFoundError('The embedding matrix was not found inside data folder. Run the data_utils.py script first !')

    matrix = np.load('data/embedding_matrix.npy')
    return matrix


def load_prepared_dataset():
    df = pd.read_csv('data/NLP_TakeHome_Feedbacks.csv', header=0, encoding='latin-1')
    print(df.info())
    df, countries, max_num_countries = clean_country(df)
    df, devices, max_num_devices = clean_device(df)

    texts = extract_text(df)
    labels = prepare_labels(df)

    data, word_index = prepare_word_tokenizer(texts)

    return data, labels, countries, devices, word_index, max_num_countries, max_num_devices


def decode_predictions(predictions):
    with open('data/label_binarizer.pkl', 'rb') as f:
        label_encoder = pickle.load(f)  # type: MultiLabelBinarizer

    # print("Classes : ", label_encoder.classes_)

    labels = label_encoder.inverse_transform(predictions)
    return labels


def get_class_decoder():
    with open('data/label_binarizer.pkl', 'rb') as f:
        label_encoder = pickle.load(f)  # type: MultiLabelBinarizer
    return label_encoder



if __name__ == '__main__':
    data, labels, countries, devices, word_index, max_num_countries, max_num_devices = load_prepared_dataset()

    # generate_embedding_matrix('data/glove.42B.300d.txt', word_index)