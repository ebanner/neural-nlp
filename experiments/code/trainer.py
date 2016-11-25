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

from callbacks import Flusher, TensorLogger, CSVLogger, StudyLogger, StudySimilarityLogger


class Trainer:
    """Handles the following tasks:

    1. Load embeddings and labels
    2. Build a keras model
    3. Calls fit() to train

    """
    def __init__(self, exp_group, exp_id, hyperparam_dict, target):
        """Set attributes

        Attributes
        ----------
        exp_group : name of experiment group
        exp_id : id of experiment

        """
        self.exp_group = exp_group
        self.exp_id = exp_id
        self.hyperparam_dict = hyperparam_dict
        self.target = target

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

        self.source, self.target = inputs

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
                      'sgd' : SGD,
                      'adadelta': Adadelta,
        }
        self.optimizer = optimizers[optimizer](**{'lr': lr, 'clipvalue': 0.5}) # try and clip gradient norm

        # define metrics
        self.model.compile(self.optimizer,
                           loss=loss,
                           metrics=per_class_accs(self.y_train) if metric == 'acc' else [])

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
        X_study, X_target = self.vecs['abstracts'].X, self.vecs[self.target].X

        # [X_source, X_target], y = next(study_target_generator(**gen_args))
        # loggers.FULL = log_full # whether to log full tensors
        weight_str = '../store/weights/{}/{}/{}-{}.h5' # where to save model weights

        # define callbacks
        cb = ModelCheckpoint(weight_str.format(self.exp_group, self.exp_id, fold, metric),
                             monitor='loss',
                             save_best_only=True,
                             mode='min')
        ce = ModelCheckpoint(weight_str.format(self.exp_group, self.exp_id, fold, 'loss'),
                             monitor='loss', # every time training loss goes down
                             mode='min')

        study_study_batch = study_target_generator(X_study[val_idxs], X_study[val_idxs],
                cdnos[val_idxs], self.exp_group, self.exp_id, nb_sample=nb_val, seed=1337, full=True, cdno_matching=False)
        [X_source_val, X_target_val], y = next(study_study_batch)
        ss = StudySimilarityLogger(X_source_val, X_target_val, study_dim=self.model.get_layer('study').output_shape[-1])

        es = EarlyStopping(monitor='study_similarity', patience=10, verbose=2, mode='max')
        fl = Flusher()
        cv = CSVLogger(self.exp_group, self.exp_id, self.hyperparam_dict, fold)
        # tl = TensorLogger([X_source, X_target], y, self.exp_group, self.exp_id,
        #                   tensor_funcs=[activations, weights, updates, update_ratios, gradients])
        sl = StudyLogger(X_study[val_idxs], self.exp_group, self.exp_id)

        # filter down callbacks
        callback_dict = {'cb': cb, # checkpoint best
                         'ce': ce, # checkpoint every
                         # 'tl': tl, # tensor logger
                         'fl': fl, # flusher
                         'es': es, # early stopping
                         'sl': sl, # study logger
                         'ss': ss, # study similarity logger
                         'cv': cv, # should go *last* as other callbacks populate `logs` dict
        }
        self.callbacks = [callback_dict[cb_name] for cb_name in callback_list]

        # train
        gen_source_target_batches = \
                study_target_generator(X_study[train_idxs], X_target[train_idxs],
                                       nb_sample=batch_size,
                                       cdnos=cdnos[train_idxs],
                                       exp_group=self.exp_group, exp_id=self.exp_id,
                                       seed=1337, # for repeatability
                                       neg_nb=-1 if self.loss == 'hinge' else 0,
                                       cdno_matching=self.source!=self.target,
                                       full=False) # training

        self.model.fit_generator(gen_source_target_batches,
                                 samples_per_epoch=(nb_train/batch_size)*batch_size,
                                 nb_epoch=nb_epoch,
                                 verbose=2,
                                 callbacks=self.callbacks)
