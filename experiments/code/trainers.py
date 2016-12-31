from collections import OrderedDict

import pickle

from keras.layers import Input, Embedding, Dropout, Dense, LSTM, merge
from keras.layers import Convolution1D, MaxPooling1D, Flatten, merge
from keras.layers import Activation
from keras.models import Model
from keras.regularizers import l2

from trainer import Trainer
from support import cnn_embed, per_class_f1s, per_class_accs, average


class SharedTrainer(Trainer):
    """Four-input model which embeds abstract and all summaries
    
    Pushes the abstract and summaries close together and pushes the summaries
    apart.

    """
    def build_model(self, nb_filter, filter_lens, nb_hidden, hidden_dim,
            dropout_prob, dropout_emb, backprop_emb, word2vec_init, reg, loss, loss_weights):

        # Embed abstract
        info = self.vecs['abstract']
        abstract = Input(shape=[info.maxlen], dtype='int32', name='abstract')
        embedded = Embedding(output_dim=info.word_dim, input_dim=info.vocab_size, input_length=info.maxlen)(abstract)
        a_embedding = cnn_embed(embedded, filter_lens, nb_filter, info.maxlen, reg, name='a_embedding')
        embed_text = Model(input=abstract, output=a_embedding)

        # Embed summaries
        p_summary = Input(shape=[info.maxlen], dtype='int32', name='p_summary')
        p_embedding = embed_text(p_summary)
        i_summary = Input(shape=[info.maxlen], dtype='int32', name='i_summary')
        i_embedding = embed_text(i_summary)
        o_summary = Input(shape=[info.maxlen], dtype='int32', name='o_summary')
        o_embedding = embed_text(o_summary)

        # Compute scores
        ap_score = merge(inputs=[a_embedding, p_embedding], mode='dot', dot_axes=1, name='ap')
        ai_score = merge(inputs=[a_embedding, i_embedding], mode='dot', dot_axes=1, name='ai')
        ao_score = merge(inputs=[a_embedding, o_embedding], mode='dot', dot_axes=1, name='ao')
        pi_score = merge(inputs=[p_embedding, i_embedding], mode='dot', dot_axes=1, name='pi')
        po_score = merge(inputs=[p_embedding, o_embedding], mode='dot', dot_axes=1, name='po')
        io_score = merge(inputs=[i_embedding, o_embedding], mode='dot', dot_axes=1, name='io')

        self.model = Model(input=[abstract, p_summary, i_summary, o_summary],
                           output=[ap_score, ai_score, ao_score, pi_score, po_score, io_score])

        self.model.compile(optimizer='sgd', loss='hinge', loss_weights=loss_weights)
