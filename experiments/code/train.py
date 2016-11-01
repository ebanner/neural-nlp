import os
import sys

import time
import plac
import pickle

import numpy as np
import pandas as pd

from sklearn.cross_validation import KFold

from trainers import PICOTrainer


@plac.annotations(
        exp_group=('the name of the experiment group for loading weights', 'option', None, str),
        exp_id=('id of the experiment - usually an integer', 'option', None, str),
        nb_epoch=('number of epochs', 'option', None, int),
        dropout_pico=('perform dropout after the embedding layer', 'option', None, str),
        reg=('l2 regularization constant', 'option', None, float),
        backprop_pico=('whether to backprop into embeddings', 'option', None, str),
        batch_size=('batch size', 'option', None, int),
        nb_train=('number of examples to train on', 'option', None, int),
        drug_name=('name of drug_name', 'option', None, str),
        n_folds=('number of folds for cross validation', 'option', None, int),
        optimizer=('optimizer to use during training', 'option', None, str),
        lr=('learning rate to use during training', 'option', None, float),
        do_cv=('do cross validation if true', 'option', None, str),
        metric=('metric to use during training (acc or f1)', 'option', None, str),
        callbacks=('list callbacks to use during training', 'option', None, str),
        trainer=('type of trainer to use', 'option', None, str),
        inputs=('data to use for input', 'option', None, str),
        pico_vectors=('list of pico vectors to use as input', 'option', None, str),
        loss=('type of loss to use during training', 'option', None, str),
)
def main(exp_group='', exp_id='', nb_epoch=5, dropout_pico='True', reg=0,
        backprop_pico='False', batch_size=128, nb_train=100000,
        drug_name='CalciumChannelBlockers', n_folds=5, optimizer='adam',
        lr=.001, do_cv='False', metric='acc', callbacks='cb,ce,fl,cv,es',
        trainer='PICOTrainer', inputs='', pico_vectors='populations,outcomes',
        loss='binary_crossentropy'):
    """Training process

    1. Parse command line arguments
    2. Load input data and labels
    3. Build the keras model
    4. Train the model
    5. Log training information

    """
    # collect hyperparams for visualization code
    args = sys.argv[1:]
    pnames, pvalues = [pname.lstrip('-') for pname in args[::2]], args[1::2]
    hyperparam_dict = {pname: pvalue for pname, pvalue in zip(pnames, pvalues)}

    # parse command line options
    backprop_pico = True if backprop_pico == 'True' else False
    dropout_pico = dropout_pico if dropout_pico == 'True' else 1e-100
    do_cv = True if do_cv == 'True' else False
    callbacks = callbacks.split(',')
    inputs = inputs.split(',')
    pico_vectors = pico_vectors.split(',')

    # load data and supervision
    trainer = eval(trainer)(exp_group, exp_id, hyperparam_dict, drug_name)
    trainer.load_labels()
    trainer.load_vectors(pico_vectors)

    # set up fold(s)
    nb_example = len(trainer.X[pico_vectors[0]])
    folds = KFold(nb_example, n_folds, shuffle=True, random_state=1337) # for reproducibility!
    if not do_cv:
        folds = list(folds)[:1] # only do the first fold if not doing cross-valiadtion

    # cross-fold training
    for fold_idx, (train_idxs, val_idxs) in enumerate(folds):
        # model
        trainer.build_model(dropout_pico, backprop_pico, reg)
        trainer.compile_model(metric, optimizer, lr, loss)

        # train
        history = trainer.train(train_idxs, val_idxs, nb_epoch, batch_size,
                nb_train, callbacks, fold_idx, metric)


if __name__ == '__main__':
    plac.call(main)
