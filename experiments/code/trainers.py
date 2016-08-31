from collections import OrderedDict

from keras.layers import Input, Embedding, Dropout, Dense, LSTM, merge
from keras.layers import Convolution1D, MaxPooling1D, Flatten, merge
from keras.layers import Activation
from keras.models import Model

from trainer import Trainer
from support import cnn_embed, per_class_f1s, per_class_accs, average


class CNNTrainer(Trainer):
    """One-input model which embeds the document with CNN ngram filters"""

    def build_model(self, nb_filter, filter_lens, nb_hidden, hidden_dim,
            dropout_prob, dropout_emb, backprop_emb, word2vec_init):

        assert len(self.vecs) == 1
        vec = self.vecs[0]

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

        self.model = Model(input=input, output=probs)

class LSTMSiameseTrainer(Trainer):
    """Two-input model which embeds abstract and summary
    
    Either push them apart or pull them together depending on label. Currently
    the abstracts and summaries do not share any weights. I think this makes
    sense as what we really care about is processing abstracts, not summaries.
    However, by not sharing weights, we could be missing out on synonyms for
    words used in the summary which are not used in the abstract, which could
    help us learn better word vectors and filters.

    Embeds study and summary with an LSTM. Take the last vector.
    
    """
    def build_model(self, nb_filter, filter_lens, nb_hidden, hidden_dim,
            dropout_prob, dropout_emb, backprop_emb, word2vec_init):

        # abstract vec
        abstract = Input(shape=[self.vecs['abstracts'].maxlen], dtype='int32')
        embedded_abstract = Embedding(output_dim=self.vecs['abstracts'].word_dim,
                                      input_dim=self.vecs['abstracts'].vocab_size,
                                      input_length=self.vecs['abstracts'].maxlen,
                                      mask_zero=True,
                                      weights=None)(abstract)
        abstract_vec = LSTM(output_dim=hidden_dim, name='study_vec')(embedded_abstract)

        # summary vec
        summary = Input(shape=[self.vecs['outcomes'].maxlen], dtype='int32')
        embedded_summary = Embedding(output_dim=self.vecs['outcomes'].word_dim,
                                      input_dim=self.vecs['outcomes'].vocab_size,
                                      input_length=self.vecs['outcomes'].maxlen,
                                      mask_zero=True,
                                      weights=None)(summary)
        summary_vec = LSTM(output_dim=hidden_dim)(embedded_summary)

        # dot vectors and send through sigmoid
        score = merge(inputs=[abstract_vec, summary_vec],
                      mode='dot',
                      dot_axes=1) # won't work without `dot_axes=1` (!!)
        prob = Activation('sigmoid')(score)

        self.model = Model(input=[abstract, summary], output=score)

class CNNSiameseTrainer(Trainer):
    """Two-input model which embeds abstract and summary
    
    Either push them apart or pull them together depending on label. Currently
    the abstracts and summaries do not share any weights. I think this makes
    sense as what we really care about is processing abstracts, not summaries.
    However, by not sharing weights, we could be missing out on synonyms for
    words used in the summary which are not used in the abstract, which could
    help us learn better word vectors and filters.

    Embeds the study and summary with a CNN.
    
    """
    def build_model(self, nb_filter, filter_lens, nb_hidden, hidden_dim,
            dropout_prob, dropout_emb, backprop_emb, word2vec_init):

        # abstract vec
        abstract = Input(shape=[self.vecs['abstracts'].maxlen], dtype='int32')
        embedded_abstract = Embedding(output_dim=self.vecs['abstracts'].word_dim,
                                      input_dim=self.vecs['abstracts'].vocab_size,
                                      input_length=self.vecs['abstracts'].maxlen,
                                      weights=None)(abstract)
        abstract_vec = cnn_embed(embedded_abstract,
                                 filter_lens,
                                 nb_filter,
                                 self.vecs['abstracts'].maxlen,
                                 name='study')
        abstract_vec = Dropout(0.5)(abstract_vec)

        # summary vec
        summary = Input(shape=[self.vecs['outcomes'].maxlen], dtype='int32')
        embedded_summary = Embedding(output_dim=self.vecs['outcomes'].word_dim,
                                      input_dim=self.vecs['outcomes'].vocab_size,
                                      input_length=self.vecs['outcomes'].maxlen,
                                      weights=None)(summary)
        summary_vec = cnn_embed(embedded_summary,
                                filter_lens,
                                nb_filter,
                                self.vecs['outcomes'].maxlen,
                                name='summary')
        summary_vec = Dropout(0.5)(summary_vec)

        # dot vectors and send through sigmoid
        score = merge(inputs=[abstract_vec, summary_vec],
                      mode='dot',
                      dot_axes=1) # won't work without `dot_axes=1` (!!)

        prob = Activation('sigmoid')(score)

        self.model = Model(input=[abstract, summary], output=prob)
