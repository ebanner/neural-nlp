import pickle

import numpy as np
import pandas as pd

from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer

from scipy.sparse import csr_matrix


class Vectorizer:
    """Tiny class for fitting a vocabulary and vectorizing texts.

    Assumes texts have already been tokenized and that tokens are separated by
    whitespace.

    This class maintains state so that it can be pickled and used to train keras
    models.

    """
    def __init__(self):
        self.embeddings = None
        self.word_dim = 300

    def __len__(self):
        """Return the length of X"""

        return len(self.X)

    def __getitem__(self, given):
        """Return a slice of X"""

        return self.X[given]


    def fit(self, texts, maxlen=None, maxlen_ratio=.95):
        """Fit the texts with a keras tokenizer
        
        Parameters
        ----------
        texts : list of strings to fit and vectorize
        maxlen : maximum length for texts
        maxlen_ratio : compute maxlen M dynamically as follows: M is the minimum
        number such that `maxlen_ratio` percent of texts have length greater
        than or equal to M.
        
        """
        # fit vocabulary
        self.tok = Tokenizer(filters='')
        self.tok.fit_on_texts(texts)

        # set up dicts
        self.word2idx = self.tok.word_index
        self.idx2word = {idx: word for word, idx in self.word2idx.items()}

        if not maxlen:
            # compute `maxlen` dynamically
            lengths = pd.Series(len(text.split()) for text in texts)
            for length in range(min(lengths), max(lengths)):
                nb_lengths = len(lengths[lengths <= length])
                if nb_lengths / float(len(texts)) >= maxlen_ratio:
                    self.maxlen = length
                    break
        else:
            self.maxlen = maxlen

        self.texts = texts
        self.vocab_size = len(self.word2idx)

    def texts_to_sequences(self, texts, do_pad=True):
        """Vectorize texts as sequences of indices
        
        Parameters
        ----------
        texts : list of strings to vectorize into sequences of indices
        do_pad : pad the sequences to `self.maxlen` if true

        """
        self.X = self.tok.texts_to_sequences(texts)

        if do_pad:
            self.X = sequence.pad_sequences(self.X, maxlen=self.maxlen)
            self.word2idx['[0]'], self.idx2word[0] = 0, '[0]' # add padding token
            self.vocab_size += 1

        return self.X

    def texts_to_BoW(self, texts):
        """Vectorize texts as BoW
        
        Parameters
        ----------
        texts : list of strings to vectorize into BoW
        
        """
        self.X = self.tok.texts_to_matrix(texts)
        self.X = self.X[:, 1:] # ignore the padding token prepended by keras
        self.X = csr_matrix(self.X) # space-saving

        return self.X

    def extract_embeddings(self, model):
        """Pull out pretrained word vectors for every word in vocabulary
        
        Parameters
        ----------
        model : gensim word2vec model

        If word is not in `model`, then randomly initialize it.
        
        """
        self.word_dim, self.vocab_size = model.vector_size, len(self.word2idx)
        self.embeddings = np.zeros([self.vocab_size, self.word_dim])

        for i, word in sorted(self.idx2word.items()):
            self.embeddings[i] = model[word] if word in model else np.random.randn(self.word_dim)

        return self.embeddings

    def test(self, doc_idx):
        """Recover text from vectorized representation of the `doc_idx`th text

        Parameters
        ----------
        doc_idx : document index to recover text from

        This function is just for sanity checking that sequence vectorization
        works.

        """
        print self.X[doc_idx]
        print
        print ' '.join(self.idx2word[idx] for idx in self.X[doc_idx] if idx)
