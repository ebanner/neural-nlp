import os
import sys

import pickle

import numpy as np
import pandas as pd

import scipy
from sklearn.metrics import f1_score, roc_auc_score

from keras.callbacks import Callback
from keras.utils.np_utils import to_categorical
import keras.backend as K


class Flusher(Callback):
    """Callback that flushes stdout after every epoch"""

    def on_epoch_end(self, epoch, logs={}):
        sys.stdout.flush()

class F1Logger(Callback):
    """Use to test that `metrics.compute_f1` is implemented correctly"""

    def __init__(self, X_val, y_val):
        super(Callback, self).__init__()
        self.X_val, self.y_val = X_val, y_val.argmax(axis=1)

    def on_epoch_end(self, epoch, logs={}):
        y_pred = self.model.predict(self.X_val).argmax(axis=1)

        f1s = f1_score(self.y_val, y_pred, average=None)

        print 'scikit f1s:', f1s
        print 'scikit f1:', np.mean(f1s)

class AUCLogger(Callback):
    """Compute AUC over the validation set"""

    def __init__(self, X_val, y_val):
        super(Callback, self).__init__()
        self.X_val = X_val.values() # only take vectors
        self.y_val = y_val.values()[-1] # only take document-level labels

    def on_epoch_end(self, epoch, logs={}):
        y_probas = self.model.predict(self.X_val)
        y_probas = y_probas[-1] if type(y_probas) == list else y_probas # special-case when only 1 output
        logs['val_auc'] = roc_auc_score(self.y_val, y_probas)

class TensorLogger(Callback):
    """Callback for monitoring value of tensors during training"""

    def __init__(self, X_train, y_train, tensor_funcs, batch_size=128, phase=1):
        """Save variables

        Parameters
        ----------
        X_train : training data
        y_train : training labels
        tensor_funcs : list of functions which take a keras model and produce two lists of names and tensors to monitor
        filters : list of names to filter tensors by
        phase : 0 for test phase and 1 for learning phase

        Note: `tensor_funcs` is a hack because the optimizers's updates don't
        become available until keras does magic in fit(). Hence we need to call
        these functions and get these tensors after training has already
        started.

        """
        super(Callback, self).__init__()

        self.X_train, self.y_train = X_train, y_train
        self.phase, self.batch_size = phase, batch_size

        self.tensor_funcs = tensor_funcs

        self.tensors, self.names = [], []
        self.values = {} # logging dict

    def on_train_begin(self, logs={}):
        """Build keras function to evaluate all tensors at once
        
        Even though some tensors may not need all of these inputs, it doesn't
        hurt to include them for those that do.
        
        """
        for tensor_func in self.tensor_funcs:
            names, tensors = tensor_func(self.model)
            self.names, self.tensors = self.names+names, self.tensors+tensors
            for name in names:
                self.values[name] = []

        inputs, output = self.model.inputs, self.model.targets[0]
        sample_weights, learning_phase = self.model.sample_weights[0], K.learning_phase()
        self.eval_tensors = K.function(inputs=inputs+[output, sample_weights, learning_phase],
                                       outputs=self.tensors)

    def on_epoch_end(self, epoch, logs={}):
        """Evaluate tensors and log their values
        
        Take a small subset of the training data to run through the network to
        compute this tensors. The subset differs each epoch.
        
        """
        subset = np.random.choice(len(self.X_train), size=self.batch_size)
        X_train, y_train = self.X_train[subset], self.y_train[subset]

        tensor_vals = self.eval_tensors([X_train, y_train, np.ones(self.batch_size), self.phase])

        for tensor_val, name in zip(tensor_vals, self.names):
            self.values[name] += [float(tensor_val)]

class CSVLogger(Callback):
    """Callback for dumping csv data during training"""

    def __init__(self, exp_group, exp_id, hyperparam_dict, fold):
        self.exp_group, self.exp_id = exp_group, exp_id
        self.fold = fold
        self.train_path = '../store/train/{}/{}/{}.csv'.format(self.exp_group, self.exp_id, self.fold)

        # write out hyperparams to disk
        hp_path = '../store/hyperparams/{}/{}.csv'.format(self.exp_group, self.exp_id)
        hp = pd.Series(hyperparam_dict)
        hp.index.name, hp.name = 'hyperparam', 'value'
        hp.to_csv(hp_path, header=True)

        super(Callback, self).__init__()

    def on_epoch_end(self, epoch, logs={}):
        """Add a line to the csv logging
        
        This csv contains only numbers related to training and nothing regarding
        hyperparameters.
        
        """
        frame = {metric: [val] for metric, val in logs.items()}
        pd.DataFrame(frame).to_csv(self.train_path,
                                   index=False,
                                   mode='a' if epoch > 0 else 'w', # overwrite if starting anew if starting anwe
                                   header=epoch==0)

class ProbaLogger(Callback):
    """Callback for dumping info for error-analysis

    Currently dump predicted probabilities for each validation example.

    """
    def __init__(self, exp_group, exp_id, X_val, nb_train, nb_class, batch_size, metric):
        self.exp_group, self.exp_id = exp_group, exp_id
        self.nb_train = nb_train
        self.nb_class = nb_class
        self.X_val = X_val
        self.batch_size = batch_size
        self.metric = metric

        self.best_score = 0
        self.proba_loc = '../store/probas/{}/{}.p'.format(self.exp_group, self.exp_id)

        # initally we haven't predicted anything
        if not os.path.exists(self.proba_loc):
            pickle.dump(np.zeros([self.nb_train, self.nb_class]), open(self.proba_loc, 'wb'))

        super(Callback, self).__init__()

    def on_epoch_end(self, epoch, logs={}):
        """Update existing predicted probas or add probs from new fold"""

        if 'acc' in logs: # fixup strings for visualization code if only using 1 output
            logs = {metric.replace('acc', 'main_acc'): val for metric, val in logs.items()}
            logs = {metric.replace('loss', 'main_loss'): val for metric, val in logs.items()}

        score = logs[self.metric]
        if score <= self.best_score:
            return

        self.best_score = score
        y_proba = pickle.load(open(self.proba_loc))
        y_proba[self.val_idxs] = self.model.predict(self.X_val, batch_size=self.batch_size)
        pickle.dump(y_proba, open(self.proba_loc, 'wb'))
