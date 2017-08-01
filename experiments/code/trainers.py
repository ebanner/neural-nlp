from collections import OrderedDict

import pickle

from keras.layers import Input, Embedding, Dropout, Dense, LSTM, merge
from keras.layers import Convolution1D, MaxPooling1D, Flatten, merge
from keras.layers import Activation
from keras.models import Model
from keras.regularizers import l2

from trainer import Trainer
from support import cnn_embed, per_class_f1s, per_class_accs, average


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
            dropout_prob, dropout_emb, backprop_emb, word2vec_init, reg, loss,
            use_pretrained, update_embeddings, project_summary):

        # abstract vec
        abstract = Input(shape=[self.vecs['abstracts'].maxlen], dtype='int32')
        E = [pickle.load(open('../data/embeddings/abstracts.p'))] if use_pretrained else None
        embedded_abstract = Embedding(output_dim=self.vecs['abstracts'].word_dim,
                                      input_dim=self.vecs['abstracts'].vocab_size,
                                      input_length=self.vecs['abstracts'].maxlen,
                                      weights=E,
                                      trainable=update_embeddings,
                                      W_regularizer=l2(reg),
                                      dropout=dropout_emb)(abstract)
        abstract_vec = cnn_embed(embedded_abstract,
                                 filter_lens,
                                 nb_filter,
                                 self.vecs['abstracts'].maxlen,
                                 reg,
                                 name='study')

        # summary vec
        summary = Input(shape=[self.vecs[self.target].maxlen], dtype='int32')
        E = [pickle.load(open('../data/embeddings/{}.p'.format(self.target)))] if use_pretrained else None
        embedded_summary = Embedding(output_dim=self.vecs[self.target].word_dim,
                                     input_dim=self.vecs[self.target].vocab_size,
                                     input_length=self.vecs[self.target].maxlen,
                                     W_regularizer=l2(reg),
                                     weights=E,
                                     trainable=update_embeddings,
                                     dropout=dropout_emb)(summary)
        summary_vec = cnn_embed(embedded_summary,
                                filter_lens,
                                nb_filter,
                                self.vecs[self.target].maxlen,
                                reg,
                                name='summary_activations')
        if project_summary:
            summary_vec = Dense(output_dim=nb_filter*len(filter_lens), name='summary')(summary_vec)

        score = merge(inputs=[abstract_vec, summary_vec], mode='dot', dot_axes=1, name='raw_score')
        if loss == 'binary_crossentropy':
            score = Activation('sigmoid')(score)

        self.model = Model(input=[abstract, summary], output=score)

class SharedCNNSiameseTrainer(Trainer):
    """Two-input model which embeds abstract and target
    
    Either push them apart or pull them together depending on label. The two
    models share weights as the inputs come from the same distribution.

    """
    def build_model(self, nb_filter, filter_lens, nb_hidden, hidden_dim,
            dropout_prob, dropout_emb, backprop_emb, word2vec_init, reg, loss):

        # Embed model
        info = self.vecs['abstracts']
        source = Input(shape=[info.maxlen], dtype='int32')
        vectorized_source = Embedding(output_dim=info.word_dim,
                                      input_dim=info.vocab_size,
                                      input_length=info.maxlen,
                                      W_regularizer=l2(reg),
                                      dropout=dropout_emb)(source)
        embedded_source = cnn_embed(vectorized_source, filter_lens, nb_filter, info.maxlen, reg, name='study')
        embed_abstract = Model(input=source, output=embedded_source)

        # Embed target (share weights)
        target = Input(shape=[info.maxlen], dtype='int32')
        embedded_target = embed_abstract(target)

        # Compute similarity
        score = merge(inputs=[embedded_source, embedded_target], mode='dot', dot_axes=1, name='score')
        if loss == 'binary_crossentropy':
            score = Activation('sigmoid')(score)

        self.model = Model(input=[source, target], output=score)
