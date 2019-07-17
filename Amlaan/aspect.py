from keras.layers import Input, Embedding, Dense, GlobalAveragePooling1D, Flatten
from keras.layers import add, multiply, LSTM
from keras.models import Model
from keras import backend as K


class MaskedGlobalAveragePooling1D(GlobalAveragePooling1D):

    def __init__(self, **kwargs):
        super(MaskedGlobalAveragePooling1D, self).__init__(**kwargs)
        self.supports_masking = True


class MaskableFlatten(Flatten):

    def __init__(self, **kwargs):
        super(MaskableFlatten, self).__init__(**kwargs)
        self.supports_masking = True

"""
What this model does:

2 ip - 1 op_name model : 2 ip = sentence and aspect sentence

Shared embedding layer = reduce # of parameters and chance to overfit.
sentence embedding = sentence passed through embedding layer (keep for later)
aspect embedding = aspect sentence passed through embedding layer 

On this aspect embedding, use attention mechanism to jointly learn what is the "best" augmentation to the sentence embedding
-   Dense layer that maps 1 : 1 between the aspect embedding and the aspect attention
    -   Softmax forces it to choose the "parts" of the sentence that help the most in training
    -   No bias needed for attention

-   Next is to actually augment the aspect embeddings with this learned attention
    -   The element-wise multiplication forces many embeddings to become close to zero
    -   Only a few will remain "strong" after this multiplication. These are the "important" words in the aspect sentence

Finally, augment the original sentence embeddings with the attended aspect embeddings
-   This will "_var_add" some strength to the embeddings of the "important" words
-   Remaining words will not be impacted at all (since they are added with near zero values)

Benefits of this model
-   Choose if you want to send a unique aspect sentence for the corresponding sentence
    -   By this I mean, you have a choice
    -   1) Use the original sentence as aspect input.
            In doing so, it is basically like saying learn on your own what the aspect word is
            It may not give much benefit, as the attended vector has the chance of being all equal (no attention)
    -   2) Use a true aspect encoding as the aspect input.
            Since you are sharing the embedding now, you cannot use random / own assigned aspects anymore.
            The aspect ids that you pass will now be from the original embedding matrix using the word_index
            dict that Keras gives you.

            In this case, an aspect sentence would be of the form : 
            [0 0 ... 32506 66049 5968 0 0 ...] 
            Here 32506 = "Apple", 66049 = "Macbook" 5968 = "Pro" (say)

"""

NUM_CLASSES = 3  # 0 = neg, 1 = neutral, 2 = pos

MAX_SENTENCE_LENGTH = 100
MAX_NUM_WORDS = 1000  # this will be number of unique "words" (n-grams etc) there are
MAX_NUM_ASPECT_WORDS = 300  # this will be the number of unique aspect "words" (uni-grams only)

EMBEDDING_DIM = 300
EMBEDDING_WEIGHTS = None

MASK_ZEROS = True  # this can be true ONLY for RNN models. If even 1 CNN is there, it will crash

#
# embedding = Embedding(MAX_NUM_WORDS, output_dim=EMBEDDING_DIM, mask_zero=MASK_ZEROS,
#                       weights=EMBEDDING_WEIGHTS, trainable=False)
#
# sentence_ip = Input(shape=(MAX_SENTENCE_LENGTH,), dtype='int32')
# aspect_ip = Input(shape=(MAX_SENTENCE_LENGTH,), dtype='int32')
#
# sentence_embedding = embedding(sentence_ip)  # Note: these are same embedding layer
# aspect_embedding = embedding(aspect_ip)  # Note: these are same embedding layer
#
# # Create the attention vector for the aspect embeddings
# aspect_attention = Dense(EMBEDDING_DIM, function='softmax', use_bias=False,
#                          name='aspect_attention')(aspect_embedding)
#
# # dampen the aspect embeddings according to the attention with an element-wise multiplication
# aspect_embedding = multiply([aspect_embedding, aspect_attention])
#
# # augment the sample embedding with information from the attended aspect embedding
# sentence_embedding = _var_add([sentence_embedding, aspect_embedding])
#
# # now you can continue with whatever layer other than CNNs
#
# x = LSTMModel(100)(sentence_embedding)
# x = Dense(NUM_CLASSES, function='softmax')(x)
#
# model = Model(inputs=[sentence_ip, aspect_ip], outputs=x)
#
# model.summary()
#
#
# from keras.utils.vis_utils import plot_model
# plot_model(model, to_file='shared_embedding.png', show_shapes=False, show_layer_names=True)
#

"""
What this model does:

2 ip - 1 op_name model : 2 ip = sentence and aspect sentence

Disjoing embedding layer = more # of parameters and chance to overfit.
sentence embedding = sentence passed through embedding layer (keep for later ; not learned)
aspect embedding = aspect sentence passed through embedding layer (learned)

Benefits of this model
-  Use a true aspect encoding as the aspect input.
   Since you are learning the embedding now, you can use own assigned aspects.
            
   In this case, an aspect sentence would be of the form : 
   [0 0 ... 2 2 2 0 0 ...] 
   Here 2 = "Apple", 2 = "Macbook" 2 = "Pro" (say)
   Therefore, the id is given by you, and is shared over all of the aspect words for a given aspect term.

"""
K.clear_session()

sentence_embedding = Embedding(MAX_NUM_WORDS, output_dim=EMBEDDING_DIM, mask_zero=MASK_ZEROS,
                      weights=EMBEDDING_WEIGHTS, trainable=False)

aspect_embedding = Embedding(MAX_NUM_ASPECT_WORDS, EMBEDDING_DIM, mask_zero=MASK_ZEROS,  # this needs to be True
                             trainable=True)

sentence_ip = Input(shape=(MAX_SENTENCE_LENGTH,), dtype='int32')
aspect_ip = Input(shape=(MAX_SENTENCE_LENGTH,), dtype='int32')

sentence_embedding = sentence_embedding(sentence_ip)  # Note: these are two different embeddings
aspect_embedding = aspect_embedding(aspect_ip)  # Note: these are two different embeddings

# Create the attention vector for the aspect embeddings
aspect_attention = Dense(EMBEDDING_DIM, activation='softmax', use_bias=False,
                         name='aspect_attention')(aspect_embedding)

# dampen the aspect embeddings according to the attention with an element-wise multiplication
aspect_embedding = multiply([aspect_embedding, aspect_attention])

# augment the sample embedding with information from the attended aspect embedding
sentence_embedding = add([sentence_embedding, aspect_embedding])

# now you can continue with whatever layer other than CNNs

#x = MaskedGlobalAveragePooling1D()(sentence_embedding)
#x = MaskableFlatten()(sentence_embedding)

x = LSTM(100)(sentence_embedding)
x = Dense(NUM_CLASSES, activation='softmax')(x)

model = Model(inputs=[sentence_ip, aspect_ip], outputs=x)

model.summary()


#from keras.utils.vis_utils import plot_model
#plot_model(model, to_file='learned_embedding.png', show_shapes=False, show_layer_names=True)