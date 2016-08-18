import os
import sys

import time
import plac
import pickle

import numpy as np
import pandas as pd

from sklearn.cross_validation import KFold

from trainers import CNNLSTMTrainer, LSTMTrainer, AuxTrainer
from predict import make_predictions


@plac.annotations(
        exp_group=('the name of the experiment group for loading weights', 'option', None, str),
        exp_id=('id of the experiment - usually an integer', 'option', None, str),
        nb_epoch=('number of epochs', 'option', None, int),
        nb_filter=('number of filters', 'option', None, int),
        filter_lens=('length of filters', 'option', None, str),
        nb_hidden=('number of hidden states', 'option', None, int),
        hidden_dim=('size of hidden state', 'option', None, int),
        dropout_prob=('dropout probability', 'option', None, float),
        dropout_emb=('perform dropout after the embedding layer', 'option', None, str),
        reg=('l2 regularization constant', 'option', None, float),
        a_reg=('l1 regularization for lstm output', 'option', None, float),
        backprop_emb=('whether to backprop into embeddings', 'option', None, str),
        batch_size=('batch size', 'option', None, int),
        word2vec_init=('initialize embeddings with word2vec', 'option', None, str),
        nb_train=('number of examples to train on', 'option', None, int),
        dataset=('name of dataset', 'option', None, str),
        n_folds=('number of folds for cross validation', 'option', None, int),
        optimizer=('optimizer to use during training', 'option', None, str),
        lr=('learning rate to use during training', 'option', None, float),
        mb_ratios=('ratio of minibatches to use during training', 'option', None, str),
        do_cv=('do cross validation if true', 'option', None, str),
        metric=('metric to use during training (acc or f1)', 'option', None, str),
        callbacks=('list callbacks to use during training', 'option', None, str),
        trainer=('type of trainer to use (currently on CNNTrainer and CNNLSTMTrainer', 'option', None, str),
        lstm_dim=('dimension of lstm hidden state', 'option', None, int),
        inputs=('any subset of `questions`, or `answers`', 'option', None, str),
        lstm_layers=('how deep to make the lstm', 'option', None, int),
        use_masking=('whether or not to use masking for lstm embedding layer', 'option', None, str),
        features=('list of additional features to use', 'option', None, str),
        ensemble_groups=('groups to ensembling with', 'option', None, str),
        ensemble_ids=('ids to do ensembling with', 'option', None, str),
        predict=('whether to just do predictions', 'option', None, str),
        predict_groups=('groups to predict', 'option', None, str),
        predict_ids=('ids to predict', 'option', None, str),
        phase=('phase to predict', 'option', None, str),
        predict_on_batches=('whether to perform ensembling on all random variable batches', 'option', None, str),
        predict_group=('group to predict with batches', 'option', None, str),
        predict_id=('id to predict with batches', 'option', None, int),
)
def main(exp_group='', exp_id='', nb_epoch=5, nb_filter=1000, filter_lens='1,2,3', 
        nb_hidden=1, hidden_dim=1024, dropout_prob=.5, dropout_emb='True', reg=0, a_reg=0,
        backprop_emb='False', batch_size=128, word2vec_init='True', nb_train=100000,
        dataset='ICHI2016', n_folds=5, optimizer='adam', lr=.001, mb_ratios='', 
        do_cv='False', metric='val_main_acc', callbacks='cb,ce,fl,cv,es',
        trainer='CNNLSTMTrainer', lstm_dim=64, inputs='questions,titles', lstm_layers=1,
        use_masking='True', features='', ensemble_groups='', ensemble_ids='', predict='False',
        predict_groups='', predict_ids='', phase='test', predict_on_batches='False',
        predict_group='', predict_id=-1):
    """Training process

    1. Parse command line arguments
    2. Load embeddings and labels
    2. Build the keras model
    3. Train the model!

    """
    # Build exp info string for visualization code...
    args = sys.argv[1:]
    pnames, pvalues = [pname.lstrip('-') for pname in args[::2]], args[1::2]
    hyperparam_dict = {pname: pvalue for pname, pvalue in zip(pnames, pvalues)}

    # parse command line options
    filter_lens = [int(filter_len) for filter_len in filter_lens.split(',')]
    backprop_emb = True if backprop_emb == 'True' else False
    dropout_emb = dropout_prob if dropout_emb == 'True' else 1e-100
    word2vec_init = True if word2vec_init == 'True' else False
    if mb_ratios: mb_ratios = [float(mb_ratio)/10 for mb_ratio in mb_ratios.split(',')]
    do_cv = True if do_cv == 'True' else False
    nb_filter /= len(filter_lens) # make it so there are only nb_filter *total* - NOT nb_filter*len(filter_lens)
    callbacks = callbacks.split(',')
    inputs = inputs.split(',') if inputs != 'None' else []
    use_masking = True if use_masking == 'True' else False
    features = features.split(',') if features != '' else []
    ensemble_groups = ensemble_groups.split(',') if ensemble_groups != '' else ensemble_groups
    ensemble_ids = ensemble_ids.split(',') if ensemble_ids != '' else ensemble_ids
    predict = True if predict == 'True' else False
    predict_groups = predict_groups.split(',') if predict_groups != '' else predict_groups
    predict_ids = predict_ids.split(',') if predict_ids != '' else predict_ids
    predict_on_batches = True if predict_on_batches == 'True' else False

    # load data and supervision
    trainer = eval(trainer)(dataset, exp_group, exp_id, hyperparam_dict, trainer)
    trainer.load_docs(inputs, 'train')
    trainer.load_auxiliary(features, 'train')
    trainer.load_labels()
    trainer.load_ensemble(ensemble_groups, ensemble_ids)

    # training
    nb_docs = len(trainer.X_train[trainer.inputs[0]])
    folds = KFold(nb_docs, n_folds, shuffle=True, random_state=1337) # for reproducibility!
    if not do_cv:
        folds = list(folds)[:1] # only do the first fold if not doing cross-valiadtion

    if predict:
        trainer.predict(predict_groups, predict_ids, phase)
        sys.exit()

    if predict_on_batches: # random variable network ensembling
        trainer.predict_on_batches(predict_group, predict_id, folds[0])
        sys.exit()

    # cross-fold validation
    for fold_idx, (train_idxs, val_idxs) in enumerate(folds):
        # model
        trainer.build_model(nb_filter, filter_lens, nb_hidden, hidden_dim, dropout_prob,
                dropout_emb, reg, a_reg, backprop_emb, word2vec_init, lstm_dim, lstm_layers,
                use_masking, ensemble_groups, ensemble_ids)
        trainer.compile_model(metric, optimizer, lr)
        trainer.save_architecture()

        # train
        history = trainer.train(train_idxs, val_idxs, nb_epoch, batch_size, mb_ratios,
                nb_train, callbacks, fold_idx, metric)


if __name__ == '__main__':
    plac.call(main)
