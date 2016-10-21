from collections import OrderedDict

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
            dropout_prob, dropout_emb, backprop_emb, word2vec_init, reg):

        # abstract vec
        abstract = Input(shape=[self.vecs['abstracts'].maxlen], dtype='int32')
        embedded_abstract = Embedding(output_dim=self.vecs['abstracts'].word_dim,
                                      input_dim=self.vecs['abstracts'].vocab_size,
                                      input_length=self.vecs['abstracts'].maxlen,
                                      W_regularizer=l2(reg))(abstract)

        abstract_vec = cnn_embed(embedded_abstract,
                                 filter_lens,
                                 nb_filter,
                                 self.vecs['abstracts'].maxlen,
                                 reg,
                                 name='study')

        # summary vec
        summary = Input(shape=[self.vecs[self.summary_type].maxlen], dtype='int32')
        embedded_summary = Embedding(output_dim=self.vecs[self.summary_type].word_dim,
                                     input_dim=self.vecs[self.summary_type].vocab_size,
                                     input_length=self.vecs[self.summary_type].maxlen,
                                     W_regularizer=l2(reg))(summary)

        summary_vec = cnn_embed(embedded_summary,
                                filter_lens,
                                nb_filter,
                                self.vecs[self.summary_type].maxlen,
                                reg,
                                name='summary')

        # dot vectors and send through sigmoid
        score = merge(inputs=[abstract_vec, summary_vec],
                      mode='dot',
                      dot_axes=1, # won't work without `dot_axes=1` (!!)
                      name='raw_score')

        # prob = Activation('sigmoid', name='sigmoid_score')(score)

        self.model = Model(input=[abstract, summary], output=score)
