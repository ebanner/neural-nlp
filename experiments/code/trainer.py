import os
import glob
import pickle

from collections import OrderedDict

import numpy as np
import pandas as pd

import scipy

import keras
from keras.models import Model
from keras.utils.np_utils import to_categorical
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.optimizers import Adam, SGD
from keras.models import model_from_json

from support import per_class_f1s, per_class_accs, stratified_batch_generator
from loggers import weights, updates, update_ratios, gradients, activations

from callbacks import Flusher, TensorLogger, CSVLogger, ProbaLogger


class Trainer:
    """Handles the following tasks:

    1. Load embeddings and labels
    2. Build a keras model
    3. Calls fit() to train

    """
    def __init__(self, exp_group, exp_id, hyperparam_dict):
        """Set attributes

        Attributes
        ----------
        exp_group : name of experiment group
        exp_id : id of experiment

        """
        self.exp_group = exp_group
        self.exp_id = exp_id
        self.hyperparam_dict = hyperparam_dict

    def load_texts(self, inputs):
        """Load inputs
        
        Parameters
        ----------
        inputs : list of vectorizer names (expected to be in ../data/vectorizers)
        
        """
        self.vecs, self.nb_train = OrderedDict(), None
        for input in inputs:
            self.vecs[input] = pickle.load(open('../data/vectorizers/{}.p'.format(input))) 
            if self.nb_train:
                assert self.nb_train == len(self.vecs[input])
            self.nb_train = len(self.vecs[input])

    def load_auxiliary(self, feature_names):
        """Load auxillary features
        
        Parameters
        ----------
        features : list of additional features to load
        
        """
        self.features = feature_names
        self.aux_len = 0
        if not feature_names:
            return

        nb_train = len(pd.read_csv('../DATA/features/{}.csv'.format(feature_names[0])))
        features = np.empty(shape=[nb_train, 0]) # to be concatenated
        for feature_name in feature_names:
            feature = pd.read_csv('../DATA/features/{}.csv'.format(feature_name))
            features = np.hstack([features, np.array(feature)])
            self.aux_len += feature.shape[1]

        self.X_train['features'] = features
        self.inputs += ['features']

    def load_labels(self, labels):
        """Load labels for dataset

        Mainly configure class names and validation data

        """
        if not labels:
            return

        self.y_vectorizer = pickle.load(open('../data/labels/{}.p'.format(labels)))
        self.y_train = self.y_vectorizer.X

        if type(self.y_train) == scipy.sparse.csr.csr_matrix:
            self.y_train = self.y_train.todense()
            sums = self.y_train.sum(axis=0)
            self.nb_class = np.count_nonzero(sums)
        else:
            uniques = np.unique(self.y_train)
            self.nb_class = len(uniques)
            self.y_train = to_categorical(self.y_train)

    def compile_model(self, metric, optimizer, lr, loss):
        """Compile keras model

        Also define functions for evaluation.

        """
        optimizers = {'adam': Adam,
                      'sgd' : SGD
        }
        self.optimizer = optimizers[optimizer](**{'lr': lr})

        # define metrics
        self.model.compile(self.optimizer,
                           loss=loss,
                           metrics=per_class_accs(self.y_train) if metric else [])

        self.model.summary()

    def save_architecture(self):
        """Write architecture of model to disk
        
        This is so we can load the architecture back, along with its weights for
        continued training or prediction.
        
        """
        json_string = self.model.to_json()
        model_loc = '../store/models/{}/{}.json'.format(self.exp_group, self.exp_id)
        open(model_loc, 'w').write(json_string)

    def train(self, train_idxs, val_idxs, nb_epoch, batch_size, nb_train,
            nb_val, callback_set, fold, metric, fit_generator):
        """Set up callbacks and start training"""

        # train and validation sets
        train_idxs, val_idxs = train_idxs[:nb_train], val_idxs[:nb_val]
        X_train, X_val = [X[train_idxs] for X in self.vecs.values()], [X[val_idxs] for X in self.vecs.values()]

        # define callbacks
        weight_str = '../store/weights/{}/{}/{}-{}.h5'
        cb = ModelCheckpoint(weight_str.format(self.exp_group, self.exp_id, fold, metric),
                             monitor='acc',
                             save_best_only=True,
                             mode='max')
        ce = ModelCheckpoint(weight_str.format(self.exp_group, self.exp_id, fold, 'loss'),
                             monitor='loss', # every time training loss goes down
                             mode='min')
        es = EarlyStopping(monitor='val_acc', patience=10, verbose=2, mode='max')
        fl = Flusher()
        cv = CSVLogger(self.exp_group, self.exp_id, self.hyperparam_dict, fold)
        # pl = ProbaLogger(self.exp_group, self.exp_id, X_val, self.nb_train, self.nb_class, batch_size, metric)
        # tl = TensorLogger(X_train, y_train, tensor_funcs=[weights, updates, update_ratios, gradients, activations])

        # filter down callbacks
        cb_shorthands, cbs = ['cb', 'ce', 'fl', 'cv', 'es'], [cb, ce, fl, cv, es]
        self.callbacks = [cb for cb_shorthand, cb in zip(cb_shorthands, cbs) if cb_shorthand in callback_set]

        if fit_generator:
            # construct y_val
            y_val = np.zeros(nb_val)
            y_val[:nb_val/2] = 1 # first half are true

            X_abstract = self.vecs['abstracts'][val_idxs]
            corrupt_idxs = val_idxs[nb_val/2:]
            val_idxs[nb_val/2:] = np.random.permutation(corrupt_idxs) # corrupt second half
            X_summary = self.vecs['outcomes'][val_idxs]

            # create batch generator
            from support import pair_generator
            gen_pairs = pair_generator(self.vecs['abstracts'][train_idxs],
                                       self.vecs['outcomes'][train_idxs],
                                       batch_size)

            self.model.fit_generator(gen_pairs,
                                     samples_per_epoch=batch_size,
                                     nb_epoch=nb_epoch,
                                     verbose=2,
                                     callbacks=self.callbacks,
                                     validation_data=([X_abstract, X_summary], y_val))
        else:
            y_train, y_val = self.y_train[train_idxs], self.y_train[val_idxs]

            # random minibatch sampling
            history = self.model.fit(X_train, y_train,
                                    batch_size=batch_size,
                                    nb_epoch=nb_epoch,
                                    verbose=2,
                                    validation_data=(X_val, y_val),
                                    callbacks=self.callbacks)
