from collections import OrderedDict

import pickle

from keras.layers import Input, Embedding, Dropout, Dense, LSTM, merge
from keras.layers import Conv1D, GlobalMaxPooling1D, Flatten, merge
from keras.layers import Activation, Lambda, ActivityRegularization
from keras.layers.merge import Dot
from keras.models import Model
from keras.regularizers import l2

from trainer import Trainer
from support import cnn_embed, per_class_f1s, per_class_accs, average, norm2d


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

class AdversarialTrainer(Trainer):
    """Six-input model of abstract and aspect summaries

    The model takes in an abstract and an aspect. We are given a "valid" aspect
    summary (from the same review) and a "corrupt" aspect summary (from a
    different review). We are also given the a "same" aspect summary (for the
    given abstract) along with "same" summaries for the other abstracts (for the
    given abstract).

    We run a conv over the abstract and summaries for the given aspect. We also
    compute the norm of the embeddings for each summary so we can either enforce
    them to be big (for the words in the summaries for the aspect) or small (for
    words in other aspect summaries).
    
    """
    def build_model(self):
        maxlen = self.C['maxlen']
        aspect = self.C['aspect']

        A = ['same_abstract']
        S = ['same_'+aspect, 'corrupt_'+aspect, 'valid_'+aspect]
        O = ['same_intervention', 'same_outcome']

        I = OrderedDict() # inputs
        for s in A+S+O:
            I[s] = Input(shape=[maxlen], dtype='int32', name=s)

        W = OrderedDict() # words
        lookup = Embedding(output_dim=self.C['word_dim'], input_dim=self.C['vocab_size'], name='embedding')
        for s in A+S+O:
            W[s] = lookup(I[s])

        C, P = OrderedDict(), OrderedDict() # conv and pool
        convolve, pool = Conv1D(filters=100, kernel_size=1), GlobalMaxPooling1D(name='pool')
        for s in A+S:
            C[s] = convolve(W[s])
            P[s] = pool(C[s])

        D = OrderedDict() # dots
        for s in S:
            D[s] = Dot(axes=1, name=s+'_score')([P['same_abstract'], P[s]])

        N = OrderedDict() # norms
        for s in S+O:
            N[s] = Lambda(norm2d, output_shape=[1], name=s+'_norm')(W[s])

        for s in S:
            N[s] = Lambda(lambda x: -x, name='neg_'+s+'_norm')(N[s])

        self.model = Model(inputs=I.values(), outputs=D.values()+N.values())
