import pickle

from collections import OrderedDict

import numpy as np
import pandas as pd

import scipy

import keras
from keras.utils.np_utils import to_categorical
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.optimizers import Adam, SGD, Adadelta
from keras.models import model_from_json
import keras.backend as K

from support import per_class_f1s, per_class_accs, stratified_batch_generator
from loggers import weights, updates, update_ratios, gradients, activations
import loggers

from batch_generators import bg1, bg2

from callbacks import Flusher, TensorLogger, CSVLogger, StudyLogger, StudySimilarityLogger, PrecisionLogger


class Trainer:
    """Handles the following tasks:

    1. Load embeddings and labels
    2. Build a keras model
    3. Calls fit() to train

    """
    def __init__(self, config):
        """Set attributes

        Attributes
        ----------
        config : sacred config dict

        Also do the train/val split.

        """
        self.C = config

        # load cdnos sorted by the ones with the most studies so we pick those first when undersampling
        df = pd.read_csv('../data/extra/pico_cdsr.csv')
        cdnos = np.array(df.groupby('cdno').size().sort_values(ascending=False).index)

        # split into train and validation at the cdno-level
        from sklearn.cross_validation import train_test_split
        nb_reviews = len(cdnos)
        train_size = self.C['train_size']
        train_cdno_idxs, val_cdno_idxs = train_test_split(np.arange(nb_reviews), train_size=train_size, random_state=1337)
        nb_train = len(train_cdno_idxs)
        first_train = np.floor(len(train_cdno_idxs)*nb_train)
        train_cdno_idxs = np.sort(train_cdno_idxs)[:first_train.astype('int')] # take a subset of the training cdnos
        val_cdno_idxs =  np.sort(val_cdno_idxs)
        train_cdnos, val_cdnos = set(cdnos[train_cdno_idxs]), set(cdnos[val_cdno_idxs])
        train_idxs = np.array(df[df.cdno.isin(train_cdnos)].index)
        val_idxs = np.array(df[df.cdno.isin(val_cdnos)].index)

        self.C['train_idxs'], self.C['val_idxs'] = train_idxs, val_idxs
        self.C['cdnos'] = df.cdno

    def load_data(self):
        """Load inputs
        
        Parameters
        ----------
        inputs : list of vectorizer names (expected to be in ../data/vectorizers)
        
        """
        self.X, self.nb_train = dict(), None
        for input in self.C['inputs']:
            self.C[input] = pickle.load(open('../data/vectorizers/{}.p'.format(input))) 
            if self.nb_train:
                assert self.nb_train == len(self.C[input])
            self.nb_train = len(self.C[input])

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

    def compile_model(self):
        """Compile keras model

        Also define functions for evaluation.

        """
        print 'Compiling...'
        identity = lambda y_true, y_pred: K.mean(y_pred)
        aspect = self.C['aspect']
        losses = {'same_'+aspect+'_score': 'hinge',
                  'valid_'+aspect+'_score': 'hinge',
                  'corrupt_'+aspect+'_score': 'hinge',
                  'neg_same_'+aspect+'_norm': identity,
                  'neg_valid_'+aspect+'_norm': identity,
                  'neg_corrupt_'+aspect+'_norm': identity,
                  'same_intervention_norm': identity,
                  'same_outcome_norm': identity,
        }
        self.model.compile(optimizer='adam', loss=losses)
        self.model.summary()

    def fit(self):
        """Set up callbacks and start training"""

        # define callbacks
        weight_str = '../store/weights/{}/{}/{}-{}.h5' # where to save model weights
        exp_group, exp_id = self.C['exp_group'], self.C['exp_id']
        fold, metric = self.C['fold'], self.C['metric']
        cb = ModelCheckpoint(weight_str.format(exp_group, exp_id, fold, metric),
                             monitor='val_loss',
                             save_best_only=True,
                             mode='min')
        ce = ModelCheckpoint(weight_str.format(exp_group, exp_id, fold, 'val_loss'),
                             monitor='val_loss', # every time training loss goes down
                             mode='min')

        X = {input: self.C[input].X for input in self.C['inputs']}
        train_idxs, val_idxs = self.C['train_idxs'], self.C['val_idxs']
        X_train = {input: X_[train_idxs] for input, X_ in X.items()}
        X_val = {input: X_[val_idxs] for input, X_ in X.items()}
        study_dim = self.model.get_layer('pool').output_shape[-1]
        val_cdnos = self.C['cdnos'].iloc[val_idxs].reset_index(drop=True)
        batch = bg2(X_val, val_cdnos)
        ss = StudySimilarityLogger(next(batch), study_dim)
        # pl = PrecisionLogger(X_val, study_dim=self.model.get_layer('study').output_shape[-1])

        es = EarlyStopping(monitor='val_loss', patience=10, verbose=2, mode='min')
        fl = Flusher()
        cv = CSVLogger(exp_group, exp_id, fold)
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
                         # 'pl': pl, # precision logger
                         'ss': ss, # study similarity logger
                         'cv': cv, # should go *last* as other callbacks populate `logs` dict
        }
        callback_list = self.C['callbacks'].split(',')
        self.callbacks = [callback_dict[cb_name] for cb_name in callback_list]
        train_cdnos = self.C['cdnos'].iloc[train_idxs].reset_index(drop=True)
        gen_source_target_batches = bg1(X_train, train_cdnos)
        nb_train = len(train_idxs)
        batch_size, nb_epoch = self.C['batch_size'], self.C['nb_epoch']

        self.model.fit_generator(gen_source_target_batches,
                                 samples_per_epoch=(nb_train/batch_size)*batch_size,
                                 nb_epoch=nb_epoch,
                                 verbose=2,
                                 callbacks=self.callbacks)
