import os
import sys

import pickle

import numpy as np
import pandas as pd

import scipy
from sklearn.metrics import f1_score

from keras.callbacks import Callback
from keras.utils.np_utils import to_categorical
import keras.backend as K


class Flusher(Callback):
    """Callback that flushes stdout after every epoch"""

    def on_epoch_end(self, epoch, logs={}):
        sys.stdout.flush()

class ValidationLogger(Callback):
    """Use to test that `metrics.compute_f1` is implemented correctly"""

    def __init__(self, X_val, y_val):
        super(Callback, self).__init__()

        self.X_val, self.y_val = X_val, y_val.argmax(axis=1)

    def on_epoch_end(self, epoch, logs={}):
        y_pred = self.model.predict(self.X_val).argmax(axis=1)

        f1s = f1_score(self.y_val, y_pred, average=None)

        print 'scikit f1s:', f1s
        print 'scikit f1:', np.mean(f1s)

class StudySimilarityLogger(Callback):
    """Callback for computing study similarity during training"""

    def __init__(self, vecs, train_idxs, phase=1, nb_study=1000):
        """Save variables

        Parameters
        ----------
        X_study : vectorized studies
        X_summary : vectorized summaries
        phase : 1 for train and 0 for test
        nb_study : number of studies to send through

        """
        super(Callback, self).__init__()

        self.X_study, self.X_summary = vecs['abstracts'][train_idxs], vecs['outcomes'][train_idxs]
        self.train_idxs = train_idxs
        self.phase = phase
        self.nb_study = nb_study

        self.nb_train = len(train_idxs)
        assert len(self.X_summary) == self.nb_train

        # load cdnos
        df = pd.read_csv('../data/extra/study_inclusion.csv', index_col=0).iloc[train_idxs]
        self.cdnos = df.cdno.astype('category').cat.codes

    def on_train_begin(self, logs={}):
        """Build keras function to produce vectorized studies
        
        Even though some tensors may not need all of these inputs, it doesn't
        hurt to include them for those that do.
        
        """
        inputs = self.model.inputs + [K.learning_phase()]
        outputs = self.model.get_layer('lstm_abstract').output

        self.embed_studies = K.function(inputs, [outputs])

    def on_epoch_end(self, epoch, logs={}):
        """Compute study similarity from the same review and different reviews"""

        subset = np.random.choice(self.nb_train, size=self.nb_study)
        X_study, X_summary = self.X_study[subset], self.X_summary[subset]
        cdnos = self.cdnos.iloc[subset]

        TEST_MODE = 0 # learning phase of 0 for test mode (i.e. do *not* apply dropout)
        study_vecs = self.embed_studies([X_study, X_summary, TEST_MODE])[0]

        similarity_scores = np.dot(study_vecs, study_vecs.T) # compute similarities

        # compute mean similarities between studies from same and different reviews
        nb_same = nb_different = 0
        same_sum = different_sum = 0
        for i in range(self.nb_study):
            for j in range(i+1, self.nb_study):
                if cdnos.iloc[i] == cdnos.iloc[j]:
                    same_sum += similarity_scores[i][j]
                    nb_same += 1
                else:
                    different_sum += similarity_scores[i][j]
                    nb_different += 1
                    
        logs['same_mean'] = same_sum / nb_same
        logs['different_mean'] = different_sum / nb_different

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
                                   mode='a', # keep appending to this csv
                                   header=epoch==0 and not os.path.isfile(self.train_path))

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

        score = logs[self.metric]
        if score <= self.best_score:
            return

        self.best_score = score
        y_proba = pickle.load(open(self.proba_loc))
        y_proba[self.val_idxs] = self.model.predict(self.X_val, batch_size=self.batch_size)
        pickle.dump(y_proba, open(self.proba_loc, 'wb'))

class StudyLogger(Callback):
    """Callback for dumping study vectors at the end of training"""

    def __init__(self, X_abstract, X_summary, exp_group, exp_id):
        """Save variables

        Parameters
        ----------
        X_abstract : vectorized abstracts
        X_summary : vectorized summaries
        exp_group : experiment group
        exp_id : experiment id

        """
        super(Callback, self).__init__()

        self.X_abstract, self.X_summary = X_abstract, X_summary
        self.exp_group, self.exp_id = exp_group, exp_id

        self.dump_loc = '../store/study_vecs/{}/{}.p'.format(self.exp_group, self.exp_id)

    def on_train_begin(self, logs={}):
        """Create function to dump abstracts"""

        inputs = self.model.inputs + [K.learning_phase()]
        outputs = self.model.get_layer('lstm_abstract').output

        self.embed_abstracts = K.function(inputs, [outputs])

    def on_train_end(self, logs={}):
        """Run all abstracts through model and dump the embeddings"""

        TEST_MODE = 0 # learning phase of 0 for test mode (i.e. do *not* apply dropout)
        abstract_vecs = self.embed_abstracts([self.X_abstract, self.X_summary, TEST_MODE])
        pickle.dump(abstract_vecs[0], open(self.dump_loc, 'w'))
