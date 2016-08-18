from collections import OrderedDict

from keras.layers import Input, Embedding, Dropout, Dense, Activation, LSTM, merge
from keras.layers import Convolution1D, MaxPooling1D, Flatten, merge
from keras.layers import Permute, Flatten, TimeDistributed, RepeatVector, ActivityRegularization
from keras.layers.convolutional import AveragePooling1D
from keras.layers.normalization import BatchNormalization
from keras.regularizers import activity_l1, l2
from keras.models import Model

from trainer import Trainer
from support import cnn_embed, per_class_f1s, per_class_accs, average


class CNNLSTMTrainer(Trainer, object):
    """Two-input model which embeds the question with a CNN and the title with
    an LSTM"""

    def build_model(self, nb_filter, filter_lens, nb_hidden, hidden_dim,
            dropout_prob, dropout_emb, reg, a_reg, backprop_emb, word2vec_init,
            lstm_dim, lstm_layers, use_masking, ensemble_groups, ensemble_ids):

        # embedding
        inputs, words = OrderedDict(), OrderedDict()
        for data in ['questions', 'titles']:
            inputs[data] = Input(shape=[self.doclens[data]], dtype='int32')
            init_embeddings = [self.embeddings[data]] if word2vec_init else None
            sentence_matrix = Embedding(output_dim=self.word_dims[data],
                                        input_dim=self.vocab_sizes[data],
                                        input_length=self.doclens[data],
                                        weights=init_embeddings,
                                        trainable=backprop_emb,
                                        mask_zero=data=='titles')(inputs[data])
            words[data] = Dropout(dropout_emb)(sentence_matrix)

        # cnn the question and lstm the title
        activations = OrderedDict()
        activations['question'] = cnn_embed(words['questions'], filter_lens, nb_filter, reg, self.doclens['questions'])
        lstmed = LSTM(lstm_dim)(words['titles'])
        activations['titles'] = Dropout(dropout_prob)(lstmed)

        # merge - optionally include auxillary features
        if self.aux_len: inputs['auxiliary'] = activations['auxiliary'] = Input(shape=[self.aux_len])
        hidden = merge(activations.values(), mode='concat')

        for _ in range(nb_hidden):
            hidden = Dense(output_dim=hidden_dim, activation='relu', W_regularizer=l2(reg))(hidden)
            hidden = Dropout(dropout_prob)(hidden)

        outputs = OrderedDict()
        outputs['main'] = Dense(output_dim=self.nb_class,
                                activation='softmax',
                                W_regularizer=l2(reg),
                                name='main')(hidden)

        if ensemble_groups and ensemble_groups:
            for ensemble_group, ensemble_id in zip(ensemble_groups, ensemble_ids):
                inputs['p_{}_{}'.format(ensemble_group, ensemble_id)] = Input(shape=[self.nb_class])

            ensemble_probs = [input for name, input in inputs.items() if name.startswith('p_')]
            outputs['ensemble'] = merge([outputs['main']] + ensemble_probs,
                                        mode=average,
                                        output_shape=lambda shapes: shapes[0],
                                        name='ensemble')

        self.model = Model(input=inputs.values() if len(inputs) > 1 else inputs.values()[0],
                           output=outputs.values() if len(outputs) > 1 else outputs.values()[0])

class AuxTrainer(Trainer, object):
    """One-input model which just uses auxiliary features to make predictions"""

    def build_model(self, nb_filter, filter_lens, nb_hidden, hidden_dim,
            dropout_prob, dropout_emb, reg, a_reg, backprop_emb, word2vec_init,
            lstm_dim, lstm_layers, use_masking, ensemble_groups, ensemble_ids):

        input = hidden = Input(shape=[self.aux_len])

        for _ in range(nb_hidden):
            hidden = Dense(output_dim=hidden_dim, activation='relu', W_regularizer=l2(reg))(hidden)
            hidden = Dropout(dropout_prob)(hidden)

        probs = Dense(output_dim=self.nb_class, activation='softmax', W_regularizer=l2(reg))(hidden)

        self.model = Model(input=input, output=probs)

class CNNTrainer(Trainer):
    """One-input model which embeds the document with a CNN
    
    The document is most likely the body, but the title can also be used.
    
    """
    def build_model(self, nb_filter, filter_lens, nb_hidden, hidden_dim,
            dropout_prob, dropout_emb, reg, a_reg, backprop_emb, word2vec_init,
            lstm_dim, lstm_layers, use_masking, ensemble_groups, ensemble_ids):

        assert len(self.inputs) == 1
        doc_type = self.inputs[0] # the body is assumed, although the title works, too

        # input
        input = Input(shape=[self.doclens[doc_type]], dtype='int32')

        # embedding
        words = Embedding(output_dim=self.word_dims[doc_type],
                          input_dim=self.vocab_sizes[doc_type],
                          input_length=self.doclens[doc_type],
                          weights=[self.embeddings[doc_type]] if word2vec_init else None,
                          trainable=backprop_emb)(input)
        words = Dropout(dropout_emb)(words)

        # cnn the document
        activations = cnn_embed(words, filter_lens, nb_filter, reg, self.doclens[doc_type])
        activations = Dropout(dropout_prob)(activations)

        hidden = Dense(output_dim=hidden_dim, W_regularizer=l2(reg))(activations)
        hidden = Activation('relu')(hidden)
        hidden = Dropout(dropout_prob)(hidden)

        probs = Dense(output_dim=self.nb_class, activation='softmax', W_regularizer=l2(reg))(hidden)

        self.model = Model(input=input, output=probs) # define the model

class HierarchicalTrainer(Trainer):
    """One-input model which embeds the document with a CNN using no pooling and
    a lot of unigram filters and runs an LSTM over the top.
    
    """
    def build_model(self, nb_filter, filter_lens, nb_hidden, hidden_dim,
            dropout_prob, dropout_emb, reg, a_reg, backprop_emb, word2vec_init,
            lstm_dim, lstm_layers, use_masking, ensemble_groups, ensemble_ids):

        assert len(self.inputs) == 1
        doc_type = self.inputs[0] # the body is assumed, although the title works, too

        # input
        input = Input(shape=[self.doclens[doc_type]], dtype='int32')

        # embedding
        words = Embedding(output_dim=self.word_dims[doc_type],
                          input_dim=self.vocab_sizes[doc_type],
                          input_length=self.doclens[doc_type],
                          weights=[self.embeddings[doc_type]] if word2vec_init else None,
                          trainable=backprop_emb)(input)
        words = Dropout(dropout_emb)(words)

        # cnn each word of the document
        convolved = Convolution1D(nb_filter=nb_filter,
                                  filter_length=1,
                                  activation='relu',
                                  W_regularizer=l2(reg))(words)

        # run an lstm over the unigram features and max-pool vectors
        activations = LSTM(nb_filter, return_sequences=True)(convolved)
        activations = ActivityRegularization(l1=a_reg)(activations)
        max_pooled = MaxPooling1D(pool_length=self.doclens[doc_type])(activations) # max-1 pooling
        flattened = Flatten()(max_pooled)
        probs = Dense(output_dim=self.nb_class, activation='softmax', W_regularizer=l2(reg))(flattened)

        self.model = Model(input=input, output=probs) # define the model

class LSTMTrainer(Trainer, object):
    """One-input model which embeds the document with an LSTM
    
    The document is most likely the title, but the body can also be used.

    LSTM model uses attention and uses mean-pooling to condense the vectors.
    
    """
    def build_model(self, nb_filter, filter_lens, nb_hidden, hidden_dim,
            dropout_prob, dropout_emb, reg, a_reg, backprop_emb, word2vec_init,
            lstm_dim, lstm_layers, use_masking, ensemble_groups, ensemble_ids):

        assert len(self.inputs) == 1
        doc_type = self.inputs[0] # the title is assumed, although the body works, too

        # input
        input = Input(shape=[self.doclens[doc_type]], dtype='int32')

        # embedding
        sentence_matrix = Embedding(output_dim=self.word_dims[doc_type],
                                    input_dim=self.vocab_sizes[doc_type],
                                    input_length=self.doclens[doc_type],
                                    weights=[self.embeddings[doc_type]] if word2vec_init else None,
                                    trainable=backprop_emb,
                                    mask_zero=False)(input)
        words = Dropout(dropout_emb)(sentence_matrix)

        # lstm the document 
        activations = words
        for _ in range(lstm_layers):
            activations = LSTM(lstm_dim, return_sequences=True)(activations)
            activations = ActivityRegularization(l1=a_reg)(activations)
            activations = Dropout(dropout_prob)(activations)

        # attention mask - transform from 1d to 2D for elemwise multiplication
        mask = TimeDistributed(Dense(1))(activations)
        mask = Flatten()(mask)
        mask = Activation('softmax')(mask)
        mask = RepeatVector(lstm_dim)(mask)
        mask = Permute([2, 1])(mask)

        # apply mask
        activations = merge([activations, mask], mode='mul')
        activations = AveragePooling1D(pool_length=self.doclens[doc_type])(activations)
        activations = Flatten()(activations)

        probs = Dense(output_dim=self.nb_class,
                      activation='softmax',
                      W_regularizer=l2(reg),
                      name='main')(activations)

        self.model = Model(input=input, output=probs)
