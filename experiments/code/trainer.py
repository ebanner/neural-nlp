import os
import glob
import pickle

from collections import OrderedDict

import numpy as np
import pandas as pd

import scipy

import keras
from keras.models import Model
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.optimizers import Adam, SGD, Adadelta
from keras.models import model_from_json

from support import per_class_f1s, per_class_accs
from loggers import weights, updates, update_ratios, gradients, activations
import loggers

from callbacks import Flusher, TensorLogger, CSVLogger, ProbaLogger


class Trainer:
    """Handles the following tasks:

    1. Load embeddings and labels
    2. Build a keras model
    3. Calls fit() to train

    """
    def __init__(self, exp_group, exp_id, hyperparam_dict, drug_name):
        """Set attributes

        Attributes
        ----------
        exp_group : name of experiment group
        exp_id : id of experiment

        """
        self.exp_group = exp_group
        self.exp_id = exp_id
        self.hyperparam_dict = hyperparam_dict
        self.drug_name = drug_name

    def load_texts(self, inputs):
        """Load inputs
        
        Parameters
        ----------
        inputs : list of vectorizer names (expected to be in ../data/vectorizers)
        
        """
        if not inputs:
            return

        self.vecs, self.nb_train = OrderedDict(), None
        for input in inputs:
            self.vecs[input] = pickle.load(open('../data/vectorizers/{}.p'.format(input))) 
            if self.nb_train:
                assert self.nb_train == len(self.vecs[input])
            self.nb_train = len(self.vecs[input])

    def load_labels(self):
        """Load labels for dataset

        Mainly configure class names and validation data

        """
        df = pd.read_csv('../data/labels/y.csv').groupby('drug').get_group(self.drug_name)
        self.drug_idxs, self.y = df.index, np.array(df.label)

    def load_vectors(self, pico_vectors):
        """Load pico vectors
        
        Parameters
        ----------
        features : list of features to load
        mode : concatenate features together into one vector if `concat` and
        keep them as separate inputs if `channels`
        
        """
        self.X = OrderedDict()
        for pico_vector in pico_vectors:
            vector_loc = '../data/vectors/{}.p'.format(pico_vector)
            X_pico = pickle.load(open(vector_loc)) # load in all vectors
            self.X[pico_vector] = X_pico[self.drug_idxs] # filter down to only drug ones

        self.pico_elements = self.X.keys()

    def compile_model(self, metric, optimizer, lr, loss):
        """Compile keras model

        Also define functions for evaluation.

        """
        optimizers = {'adam': Adam,
                      'sgd' : SGD,
                      'adadelta': Adadelta,
        }
        self.optimizer = optimizers[optimizer](**{'lr': lr, 'clipvalue': 0.5}) # try and clip gradient norm

        # define metrics
        self.model.compile(self.optimizer, loss=loss, metrics=['accuracy'])

        self.model.summary()

    def save_architecture(self):
        """Write architecture of model to disk
        
        This is so we can load the architecture back, along with its weights for
        continued training or prediction.
        
        """
        json_string = self.model.to_json()
        model_loc = '../store/models/{}/{}.json'.format(self.exp_group, self.exp_id)
        open(model_loc, 'w').write(json_string)

    def train(self, train_idxs, val_idxs, nb_epoch, batch_size, nb_train, callback_list, fold, metric):
        """Set up callbacks

        It's expected that the implementing subclass actually does the training
        (i.e. calls fit()).

        """
        # splits
        X_train, X_val = [X[train_idxs][:nb_train] for X in self.X.values()], [X[val_idxs] for X in self.X.values()]
        y_train, y_val = self.y[train_idxs][:nb_train], self.y[val_idxs]

        # callbacks
        weights_str = '../store/weights/{}/{}/{}-{}.h5'
        cb = ModelCheckpoint(weights_str.format(self.exp_group, self.exp_id, fold, metric),
                             monitor='acc',
                             save_best_only=True,
                             mode='max')
        ce = ModelCheckpoint(weights_str.format(self.exp_group, self.exp_id, fold, 'loss'),
                             monitor='val_loss', # every time training loss goes down
                             mode='min')
        es = EarlyStopping(monitor='loss', patience=10, verbose=2, mode='min')
        fl = Flusher()
        cv = CSVLogger(self.exp_group, self.exp_id, self.hyperparam_dict, fold)
        tl = TensorLogger(X_train, y_train, tensor_funcs=[weights, updates, update_ratios, gradients, activations])

        # filter down callbacks
        callback_dict = {'cb': cb, # checkpoint best
                         'ce': ce, # checkpoint every
                         'tl': tl, # tensor logger
                         'fl': fl, # flusher
                         'es': es, # early stopping
                         'cv': cv, # should go *last* as other callbacks populate `logs` dict
        }
        self.callbacks = [callback_dict[cb_name] for cb_name in callback_list]

        # train
        history = self.model.fit(X_train, y_train,
                                 batch_size=batch_size,
                                 nb_epoch=nb_epoch,
                                 verbose=2,
                                 validation_data=(X_val, y_val),
                                 callbacks=self.callbacks)
