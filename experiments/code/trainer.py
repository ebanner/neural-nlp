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

from callbacks import Flusher, TensorLogger, CSVLogger, ProbaLogger, AUCLogger


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

    def load_labels(self, outputs):
        """Load labels for dataset

        Mainly configure class names and validation data

        """
        self.outputs = outputs # not actually doing anything with this here...

        self.y_df = pd.read_csv('../data/labels/y.csv').groupby('drug').get_group(self.drug_name)
        self.weight_df = self.y_df.copy()

        # Adjust sample weights for missing labels
        first_label = self.y_df.columns[[i+1 for i, col in enumerate(self.y_df.columns) if col == 'drug'][0]] # just apply to labels
        self.weight_df.ix[:, first_label:] = self.weight_df.ix[:, first_label:].apply(lambda col: col.map({1: 1, 0: 1, -1: 0}))
        self.drug_idxs = self.y_df.index

    def load_vectors(self, inputs):
        """Load pico vectors
        
        Parameters
        ----------
        inputs : names of vectors to use in the model
        
        """
        self.X = OrderedDict()
        for vector_type in inputs:
            vector_loc = '../data/vectors/{}.p'.format(vector_type)
            X = pickle.load(open(vector_loc)) # load in all vectors
            self.X[vector_type] = X[self.drug_idxs] # filter down to only drug ones

        self.y, self.sample_weight = OrderedDict(), OrderedDict()
        for output in self.outputs:
            self.y[output] = np.array(self.y_df[output][self.drug_idxs])
            self.sample_weight[output] = np.array(self.weight_df[output][self.drug_idxs])

        self.inputs = inputs

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
        """Carve up train/val split and call fit() with callbacks"""

        # split inputs
        X_train, X_val = OrderedDict(), OrderedDict()
        for input, X in self.X.items():
            X_train[input], X_val[input] = X[train_idxs][:nb_train], X[val_idxs]

        # split labels and sample weights
        y_train, y_val = OrderedDict(), OrderedDict()
        weight_train, weight_val = OrderedDict(), OrderedDict()
        for output, y, weight in zip(self.outputs, self.y.values(), self.sample_weight.values()):
            prob_str = '{}-prob'.format(output)
            y_train[prob_str], y_val[prob_str] = y[train_idxs][:nb_train], y[val_idxs]
            weight_train[prob_str], weight_val[prob_str] = weight[train_idxs][:nb_train], weight[val_idxs]

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
        # tl = TensorLogger(X_train, y_train, tensor_funcs=[weights, updates, update_ratios, gradients, activations])
        al = AUCLogger(X_val, y_val)

        # filter down callbacks
        callback_dict = {'cb': cb, # checkpoint best
                         'ce': ce, # checkpoint every
                         # 'tl': tl, # tensor logger
                         'fl': fl, # flusher
                         'es': es, # early stopping
                         'cv': cv, # should go *last* as other callbacks populate `logs` dict
                         'al': al, # auc logger
        }
        self.callbacks = [callback_dict[cb_name] for cb_name in callback_list]

        # train
        history = self.model.fit(dict(X_train), dict(y_train),
                                 batch_size=batch_size,
                                 nb_epoch=nb_epoch,
                                 verbose=2,
                                 callbacks=self.callbacks,
                                 validation_data=(dict(X_val), dict(y_val), dict(weight_val)) if 'al' not in callback_list else None,
                                 sample_weight=dict(weight_train))
