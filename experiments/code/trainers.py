from collections import OrderedDict

from keras.layers import Input, Embedding, Dropout, Dense, LSTM, merge
from keras.layers import Convolution1D, MaxPooling1D, Flatten, merge
from keras.layers import Permute, Flatten, TimeDistributed, RepeatVector, ActivityRegularization
from keras.layers.convolutional import AveragePooling1D
from keras.layers.normalization import BatchNormalization
from keras.regularizers import activity_l1, l2
from keras.models import Model

from trainer import Trainer
from support import cnn_embed, per_class_f1s, per_class_accs, average


class CNNTrainer(Trainer):
    """One-input model which embeds the document with CNN ngram filters"""

    def build_model(self, nb_filter, filter_lens, nb_hidden, hidden_dim,
            dropout_prob, dropout_emb, backprop_emb, word2vec_init):

        assert len(self.X_vecs) == 1
        vec = self.X_vecs[0]

        # input
        input = Input(shape=[vec.maxlen], dtype='int32')

        # embedding
        words = Embedding(output_dim=vec.word_dim,
                          input_dim=vec.vocab_size,
                          input_length=vec.maxlen,
                          weights=[vec.embeddings] if word2vec_init else None,
                          trainable=backprop_emb)(input)
        words = Dropout(dropout_emb)(words)

        # extract ngram features with cnn
        activations = cnn_embed(words, filter_lens, nb_filter, vec.maxlen)
        hidden = Dropout(dropout_prob)(activations)

        for _ in range(nb_hidden-1):
            hidden = Dense(output_dim=hidden_dim, activation='relu')(hidden)
            hidden = Dropout(dropout_prob)(hidden)

        probs = Dense(output_dim=self.nb_class, activation='sigmoid')(hidden)

        self.model = Model(input=input, output=probs) # define the model
