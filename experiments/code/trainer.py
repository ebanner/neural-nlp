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
    def __init__(self, dataset, exp_group, exp_id, hyperparam_dict, trainer_type):
        """Set attributes

        Attributes
        ----------
        dataset : name of dataset to load
        exp_group : name of experiment group
        exp_id : id of experiment
        trainer_type : the name of the type of trainer you are

        """
        self.dataset = dataset
        self.exp_group = exp_group
        self.exp_id = exp_id
        self.hyperparam_dict = hyperparam_dict
        self.trainer_type = trainer_type

    def load_texts(self, inputs):
        """Load inputs
        
        Parameters
        ----------
        inputs : list of vectorizer names (expected to be in ../data/vectorizers)
        
        """
        self.X_vecs = [pickle.load(open('../data/vectorizers/{}.p'.format(input))) for input in inputs]
        self.nb_train = len(self.X_vecs[0].X)

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

    def compile_model(self, metric, optimizer, lr):
        """Compile keras model

        Also define functions for evaluation.

        """
        optimizers = {'adam': Adam,
                      'sgd' : SGD
        }
        self.optimizer = optimizers[optimizer](**{'lr': lr})

        # define metrics
        self.model.compile(self.optimizer,
                           loss='categorical_crossentropy',
                           metrics=per_class_accs(self.y_train))

        self.model.summary()

    def save_architecture(self):
        """Write architecture of model to disk
        
        This is so we can load the architecture back, along with its weights for
        continued training or prediction.
        
        """
        json_string = self.model.to_json()
        model_loc = '../store/models/{}/{}.json'.format(self.exp_group, self.exp_id)
        open(model_loc, 'w').write(json_string)

    def train(self, train_idxs, val_idxs, nb_epoch, batch_size, nb_train, callback_set, fold, metric):
        """Set up callbacks

        It's expected that the implementing subclass actually does the training
        (i.e. calls fit()).

        """
        # training set
        X_train = [X_vec.X[train_idxs][:nb_train] for X_vec in self.X_vecs]
        y_train = self.y_train[train_idxs][:nb_train]

        # validation set
        X_val = [X_vec.X[val_idxs] for X_vec in self.X_vecs]
        y_val = self.y_train[val_idxs]

        # define callbacks
        weights_str = '../store/weights/{}/{}/{}-{}.h5'
        cb = ModelCheckpoint(weights_str.format(self.exp_group, self.exp_id, fold, metric),
                             monitor='acc',
                             save_best_only=True,
                             mode='max')
        ce = ModelCheckpoint(weights_str.format(self.exp_group, self.exp_id, fold, 'loss'),
                             monitor='loss', # every time training loss goes down
                             mode='min')
        es = EarlyStopping(monitor='val_acc', patience=10, verbose=2, mode='max')
        fl = Flusher()
        cv = CSVLogger(self.exp_group, self.exp_id, self.hyperparam_dict, fold)
        pl = ProbaLogger(self.exp_group, self.exp_id, X_val, self.nb_train, self.nb_class, batch_size, metric)
        tl = TensorLogger(X_train, y_train, tensor_funcs=[weights, updates, update_ratios, gradients, activations])

        # filter down callbacks
        cb_shorthands, cbs = ['cb', 'ce', 'fl', 'cv', 'pl', 'tl', 'es'], [cb, ce, fl, cv, pl, tl, es]
        self.callbacks = [cb for cb_shorthand, cb in zip(cb_shorthands, cbs) if cb_shorthand in callback_set]

        # random minibatch sampling
        history = self.model.fit(X_train, y_train,
                                 batch_size=batch_size,
                                 nb_epoch=nb_epoch,
                                 verbose=2,
                                 validation_data=(X_val, y_val),
                                 callbacks=self.callbacks)
