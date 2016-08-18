import os
import glob
import pickle

from collections import OrderedDict

import numpy as np
import pandas as pd

import keras
from keras.models import Model
from keras.utils.np_utils import to_categorical
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.optimizers import Adam, SGD
from keras.models import model_from_json

from support import per_class_f1s, per_class_accs, stratified_batch_generator
from loggers import weights, updates, update_ratios, gradients, activations

from callbacks import Flusher, TensorLogger, CSVLogger, ProbaLogger, RandomVariableLogger


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

    def load_docs(self, inputs, phase):
        """Load word embeddings and text and unpack"""

        self.embeddings, self.X_train, = OrderedDict(), OrderedDict()
        self.doclens, self.vocab_sizes, self.word_dims = OrderedDict(), OrderedDict(), OrderedDict()
        for data in inputs:
            embeddings_dict = pickle.load(open('../DATA/embeddings/{}_{}_{}.p'.format(self.dataset, data, phase)))
            self.X_train[data] = embeddings_dict['X_train']
            self.embeddings[data] = embeddings_dict['embeddings']['google-news']
            self.doclens[data] = embeddings_dict['maxlen']
            self.vocab_sizes[data], self.word_dims[data] = self.embeddings[data].shape

        self.inputs = inputs

    def load_auxiliary(self, feature_names, phase):
        """Load auxillary features
        
        Parameters
        ----------
        features : list of additional features to load
        
        """
        self.features = feature_names
        self.aux_len = 0
        if not feature_names:
            return

        nb_train = len(pd.read_csv('../DATA/features/{}_{}.csv'.format(feature_names[0], phase)))
        features = np.empty(shape=[nb_train, 0]) # to be concatenated
        for feature_name in feature_names:
            feature = pd.read_csv('../DATA/features/{}_{}.csv'.format(feature_name, phase))
            features = np.hstack([features, np.array(feature)])
            self.aux_len += feature.shape[1]

        self.X_train['features'] = features
        self.inputs += ['features']

    def load_labels(self):
        """Load labels for dataset

        Mainly configure class names and validation data

        """
        self.ys_df = pd.read_csv(open('../DATA/labels/{}_train.csv'.format(self.dataset)))

        self.y_train = np.array(self.ys_df).flatten()
        self.y_train = to_categorical(self.y_train)

        self.nb_class = len(self.ys_df.label.unique())

    def load_ensemble(self, ensemble_groups, ensemble_ids):
        """Load ensemble probabilities
        
        This function is currently deprecated!
        
        """
        if not ensemble_groups and not ensemble_ids:
            return

        for group, id in zip(ensemble_groups, ensemble_ids):
            p_id = 'p_{}_{}'.format(group, id)
            self.X_train[p_id] = pickle.load(open('../store/probas/{}/{}.p'.format(group, id)))

        self.y_train['ensemble'] = self.y_train['labels']

    def load_architecture(self):
        arch_loc = '../store/models/{}/{}.json'.format(self.exp_group, self.exp_id)
        self.model = model_from_json(open(arch_loc).read())

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

    def train(self, train_idxs, val_idxs, nb_epoch, batch_size, mb_ratios, nb_train,
            callback_set, fold, metric):
        """Set up callbacks

        It's expected that the implementing subclass actually does the training
        (i.e. calls fit()).

        """
        # train data
        X_train = [X[train_idxs][:nb_train] for X in self.X_train.values()]
        y_train = self.y_train[train_idxs][:nb_train]

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
        pl = ProbaLogger(self.exp_group, self.exp_id, self.X_train, self.nb_class, val_idxs, batch_size, metric)
        tl = TensorLogger(X_train, y_train, tensor_funcs=[weights, updates, update_ratios, gradients, activations])
        rv = RandomVariableLogger(self.exp_group, self.exp_id)

        # filter down callbacks
        cb_shorthands, cbs = ['cb', 'ce', 'fl', 'cv', 'pl', 'tl', 'es', 'rv'], [cb, ce, fl, cv, pl, tl, es, rv]
        self.callbacks = [cb for cb_shorthand, cb in zip(cb_shorthands, cbs) if cb_shorthand in callback_set]

        # random minibatch sampling
        X_val = [X[val_idxs] for X in self.X_train.values()]
        y_val = self.y_train[val_idxs]

        history = self.model.fit(X_train, y_train,
                                 batch_size=batch_size,
                                 nb_epoch=nb_epoch,
                                 verbose=2,
                                 validation_data=(X_val, y_val),
                                 callbacks=self.callbacks)

    def predict(self, predict_groups, predict_ids, phase):
        """Make predictions for the test set"""

        # load test data
        self.load_docs(self.inputs, phase)
        self.load_auxiliary(self.features, phase)

        if self.trainer_type == 'AuxTrainer': # hack because AuxTrainer only looks at features
            del self.X_train['questions']
            del self.X_train['titles']

        self.X_test = self.X_train # for sanity

        # load class map
        idx2class = pd.read_csv('../DATA/classes.csv', index_col=0).label.to_dict()

        # load models
        models = [0]*len(predict_groups)
        for i, (group, id) in enumerate(zip(predict_groups, predict_ids)):
            probas = [0]*5
            for fold in range(5):
                # load architecture and weights
                arch_loc = '../store/models/{}/{}.json'.format(group, id)
                weights_loc = '../store/weights/{}/{}/{}-val_main_acc.h5'.format(group, id, fold)
                model = model_from_json(open(arch_loc).read())
                model.load_weights(weights_loc)

                # make predictions
                probas[fold] = model.predict(self.X_test.values())

                # write predictions to disk for just this fold
                preds = probas[fold].argmax(axis=1)
                s = pd.Series(preds).map(idx2class)
                s.index += 1 # ichi 1-based indexing
                s.index.name, s.name = 'ID', 'Category' # ichi column headers
                s.to_csv('../store/predictions/{}/{}/{}-{}.csv'.format(group, id, fold, phase), header=True)

            # ensemble across folds
            final_probas = sum(probas) / 5
            preds = final_probas.argmax(axis=1)

            # write predictions to disk
            s = pd.Series(preds).map(idx2class)
            s.index += 1 # ichi 1-based indexing
            s.index.name, s.name = 'ID', 'Category' # ichi column headers
            s.to_csv('../store/predictions/{}/{}-{}.csv'.format(group, id, phase), header=True)

    def predict_on_batches(self, predict_group, predict_id, fold):
        """Make predictions for the test set"""

        train_idxs, val_idxs = fold
        X_val, y_val = [X[val_idxs] for X in self.X_train.values()], self.y_train[val_idxs].argmax(axis=1)

        # load architecture once
        arch_loc = '../store/models/{}/{}.json'.format(predict_group, predict_id)
        model = model_from_json(open(arch_loc).read())

        batch_weights = glob.glob('../store/batch_weights/{}/{}/*.h5'.format(predict_group, predict_id))
        probas = [0]*len(batch_weights)
        for i, weight_loc in enumerate(batch_weights):
            model.load_weights(weight_loc)
            probas[i] = model.predict(X_val)
            preds = probas[i].argmax(axis=1)

            print '{} acc'.format(os.path.basename(weight_loc)), np.mean(preds == y_val)

        # write predictions to disk for just this fold
        probas = sum(probas) / len(probas)
        preds = probas.argmax(axis=1)

        print
        print 'ensembled acc', np.mean(preds == y_val)
