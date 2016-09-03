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
    """Callback for computing and inserting the study similarity during training
    
    The study similarity is defined as the mean similarity between studies in
    the same review minus the mean similarity between studies in different
    reviews.
    
    """
    def __init__(self, X_study, X_summary, top_cdnos, cdnos, nb_sample=1000, phase=0):
        """Save variables and sample study indices

        Parameters
        ----------
        X_study : vectorized studies
        X_summary : vectorized summaries
        top_cdnos : set of `cdnos.unique()`
        cdnos : mapping from study indexes to their cdno
        nb_sample : number of studies to evaluate
        phase : 1 for train and 0 for test

        """
        super(Callback, self).__init__()

        self.X_study, self.X_summary = X_study, X_summary
        self.phase = phase

        from batch_generators import pair_generator
        study_study_pairs = pair_generator(nb_train=len(X_study),
                                           cdnos=cdnos,
                                           top_cdnos=top_cdnos),
                                           nb_sample=1000,
                                           exact_only=False)

        self.source_idxs, self.target_idxs = next(study_study_pairs)

    def on_train_begin(self, logs={}):
        """Build keras function to produce vectorized studies
        
        Even though some tensors may not need all of these inputs, it doesn't
        hurt to include them for those that do.
        
        """
        # build keras function to get study embeddings
        inputs = self.model.inputs + [K.learning_phase()]
        outputs = self.model.get_layer('study_vec').output
        self.embed_studies = K.function(inputs, [outputs])

    def on_epoch_end(self, epoch, logs={}):
        """Compute study similarity from the same review and different reviews"""

        source_vecs = self.embed_studies([self.X_study[self.source_idxs], self.X_summary, self.phase])[0]
        target_vecs = self.embed_studies([self.X_study[self.target_idxs], self.X_summary, self.phase])[0]

        score = np.sum(source_vecs*target_vecs, axis=1) # dot corresponding entries
        same_study_mean = score[:self.nb_sample/2].mean()
        different_study_mean = score[self.nb_sample/2:].mean()

        # this should be high when we're doing well
        logs['similarity_score'] = same_study_mean - different_study_mean

class TensorLogger(Callback):
    """Callback for monitoring value of tensors during training"""

    def __init__(self, X_train, y_train, tensor_funcs, batch_size=128, phase=1):
        """Save variables

        Parameters
        ----------
        X_train : training data
        y_train : training labels
        tensor_funcs : list of functions which take a keras model and produce tensors as well as their names
        phase : 0 for test phase and 1 for learning phase (defaults to 1 because
        we may want to examine number of neurons surviving dropout, for instance)

        Note: `tensor_funcs` is a hack. Ideally we would just pass the tensors
        themselves, but tensors owned by the optimizer (e.g. update tensors)
        don't become available until keras does some magic in fit(). Hence we
        work around this by providing functions that will return tensors instead
        of the tensors themselves.

        """
        super(Callback, self).__init__()

        self.phase, self.batch_size = phase, batch_size

        self.tensor_funcs = tensor_funcs

        self.tensors, self.names = [], []
        self.values = {} # logging dict

        # sample random subset of the training data for computing tensors
        nb_train = len(X_train)
        subset = np.random.choice(nb_train, size=batch_size)
        self.X_train, self.y_train = [X[subset] for X in X_train], y_train[subset]

    def on_train_begin(self, logs={}):
        """Build keras function to evaluate all tensors in one call
        
        Even though some tensors may not need all of these inputs, it doesn't
        hurt to include them for those that do.
        
        """
        for tensor_func in self.tensor_funcs:
            names, tensors = tensor_func(self.model)
            self.names, self.tensors = self.names+names, self.tensors+tensors
            for name in names:
                self.values[name] = []

        inputs = self.model.inputs
        labels, sample_weights = self.model.targets[0], self.model.sample_weights[0] # needed to compute gradient/update tensors
        learning_phase = K.learning_phase() # needed to compute forward pass (e.g. dropout)
        self.eval_tensors = K.function(inputs=inputs+[labels, sample_weights, learning_phase], outputs=self.tensors)

    def on_epoch_end(self, epoch, logs={}):
        """Evaluate tensors and log their values
        
        Take a small subset of the training data to run through the network to
        compute this tensors. The subset differs each epoch.
        
        """
        tensor_vals = self.eval_tensors([self.X_train,
                                         self.y_train,
                                         np.ones(self.batch_size), # each sample has the same weight
                                         self.phase])

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

        score = logs[self.metric]
        if score <= self.best_score:
            return

        self.best_score = score
        y_proba = pickle.load(open(self.proba_loc))
        y_proba[self.val_idxs] = self.model.predict(self.X_val, batch_size=self.batch_size)
        pickle.dump(y_proba, open(self.proba_loc, 'wb'))

class StudyLogger(Callback):
    """Callback for dumping study embeddings
    
    Dump study vectors every time we reach a new best for study similarity.
    
    """
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
        self.max_score = -np.inf # study similarity score (computed in StudySimilarityLogger)

    def on_train_begin(self, logs={}):
        """Define keras function for computing study embeddings"""

        inputs = self.model.inputs + [K.learning_phase()]
        outputs = self.model.get_layer('study_vec').output

        self.embed_abstracts = K.function(inputs, [outputs]) # don't need summary to compute study embedding

    def on_epoch_end(self, epoch, logs={}):
        """Run all abstracts through model and dump the embeddings"""

        score = logs['similarity_score']
        if score < self.max_score:
            return # only log study vectors when we reach a new best similarity score

        TEST_MODE = 0 # learning phase of 0 for test mode (i.e. do *not* apply dropout)
        abstract_vecs = self.embed_abstracts([self.X_abstract, self.X_summary, TEST_MODE])
        pickle.dump(abstract_vecs[0], open(self.dump_loc, 'w'))
