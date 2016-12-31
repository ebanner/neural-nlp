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
from keras.optimizers import Adam, SGD, Adadelta
from keras.models import model_from_json

from support import per_class_f1s, per_class_accs, stratified_batch_generator
from loggers import weights, updates, update_ratios, gradients, activations
import loggers

from batch_generators import study_target_generator

from callbacks import Flusher, TensorLogger, CSVLogger, StudyLogger, StudySimilarityLogger, PrecisionLogger


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

    def load_texts(self):
        """Load all PICO inputs"""

        self.vecs, self.nb_train = OrderedDict(), None
        for input in ['abstract', 'p_summary', 'i_summary', 'o_summary']:
            self.vecs[input] = pickle.load(open('../data/vectorizers/{}.p'.format(input))) 
            if self.nb_train:
                assert self.nb_train == len(self.vecs[input])
            self.nb_train = len(self.vecs[input])

    def load_labels(self):
        """Load labels for dataset

        Mainly configure class names and validation data

        """
        self.y = OrderedDict()
        for s1, s2 in zip(['ap', 'ai', 'ao'], ['pi', 'po', 'io']):
            self.y[s1], self.y[s2] = np.ones(self.nb_train), np.full(self.nb_train, -1)

    def compile_model(self, metric, optimizer, lr, loss):
        """Compile keras model

        Also define functions for evaluation.

        """
        print 'Compiling...'
        optimizers = {'adam': Adam,
                      'sgd' : SGD,
                      'adadelta': Adadelta,
        }
        self.optimizer = optimizers[optimizer](**{'lr': lr, 'clipvalue': 0.5}) # try and clip gradient norm

        # define metrics
        self.model.compile(self.optimizer, loss=loss)
        self.model.summary()

        self.loss = loss

    def save_architecture(self):
        """Write architecture of model to disk
        
        This is so we can load the architecture back, along with its weights for
        continued training or prediction.
        
        """
        json_string = self.model.to_json()
        model_loc = '../store/models/{}/{}.json'.format(self.exp_group, self.exp_id)
        open(model_loc, 'w').write(json_string)

    def train(self, train_idxs, val_idxs, nb_epoch, batch_size, callback_list,
            fold, metric, fit_generator, cdnos, nb_sample, log_full):
        """Set up callbacks and start training"""

        # train and validation sets
        nb_train, nb_val = len(train_idxs), len(val_idxs)

        weight_str = '../store/weights/{}/{}/{}-{}.h5' # where to save model weights
        cb = ModelCheckpoint(weight_str.format(self.exp_group, self.exp_id, fold, metric),
                             monitor='loss',
                             save_best_only=True,
                             mode='min')
        ce = ModelCheckpoint(weight_str.format(self.exp_group, self.exp_id, fold, 'loss'),
                             monitor='loss', # every time training loss goes down
                             mode='min')

        study_study_batch = study_target_generator(self.vecs['abstract'].X[val_idxs],
                self.vecs['abstract'].X[val_idxs], cdnos[val_idxs], self.exp_group,
                self.exp_id, nb_sample=nb_val, seed=1337, full=True, cdno_matching=False, pos_ratio=0.5)
        [X_source_val, X_target_val], y = next(study_study_batch)
        ss = StudySimilarityLogger(X_source_val, X_target_val, study_dim=self.model.get_layer('a_embedding').output_shape[-1])
        pl = PrecisionLogger(X_source_val, X_target_val, study_dim=self.model.get_layer('a_embedding').output_shape[-1])

        es = EarlyStopping(monitor='val_precision', patience=10, verbose=2, mode='max')
        fl = Flusher()
        cv = CSVLogger(self.exp_group, self.exp_id, self.hyperparam_dict, fold)
        # tl = TensorLogger([X_source, X_target], y, self.exp_group, self.exp_id,
        #                   tensor_funcs=[activations, weights, updates, update_ratios, gradients])
        # sl = StudyLogger(X_study[val_idxs], self.exp_group, self.exp_id)

        # filter down callbacks
        callback_dict = {'cb': cb, # checkpoint best
                         'ce': ce, # checkpoint every
                         # 'tl': tl, # tensor logger
                         'fl': fl, # flusher
                         'es': es, # early stopping
                         # 'sl': sl, # study logger
                         'pl': pl, # precision logger
                         'ss': ss, # study similarity logger
                         'cv': cv, # should go *last* as other callbacks populate `logs` dict
        }
        self.callbacks = [callback_dict[cb_name] for cb_name in callback_list]

        X_train = {input: vec[train_idxs] for input, vec in self.vecs.items()}
        y_train = {score: y[train_idxs] for score, y in self.y.items()}
        history = self.model.fit(X_train, y_train,
                                 batch_size=batch_size,
                                 nb_epoch=nb_epoch,
                                 verbose=2,
                                 callbacks=self.callbacks)
