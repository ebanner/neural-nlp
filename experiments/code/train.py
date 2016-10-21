import os
import sys

import time
import plac
import pickle

import numpy as np
import pandas as pd

from sklearn.cross_validation import KFold

from trainers import CNNSiameseTrainer


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
        backprop_emb=('whether to backprop into embeddings', 'option', None, str),
        batch_size=('batch size', 'option', None, int),
        word2vec_init=('initialize embeddings with word2vec', 'option', None, str),
        n_folds=('number of folds for cross validation', 'option', None, int),
        optimizer=('optimizer to use during training', 'option', None, str),
        lr=('learning rate to use during training', 'option', None, float),
        do_cv=('do cross validation if true', 'option', None, str),
        metric=('metric to use during training (acc or f1)', 'option', None, str),
        callbacks=('list callbacks to use during training', 'option', None, str),
        trainer=('type of trainer to use', 'option', None, str),
        features=('list of additional features to use', 'option', None, str),
        input=('data to use for input', 'option', None, str),
        labels=('labels to use', 'option', None, str),
        fit_generator=('whether to use a fit generator', 'option', None, str),
        loss=('type of loss to use', 'option', None, str),
        nb_train=('the percentage of dataset to use', 'option', None, float),
        nb_sample=('number of reviews sample for computing study similarity', 'option', None, int),
        log_full=('log full tensors with TensorLogger if True and magnitudes otherwise', 'option', None, str),
        train_size=('number between 0 and 1 for train/test split', 'option', None, float),
        summary_type=('`populations`, `interventions`, or `outcomes`', 'option', None, str),
)
def main(exp_group='', exp_id='', nb_epoch=5, nb_filter=1000, filter_lens='1,2,3', 
        nb_hidden=1, hidden_dim=1024, dropout_prob=.5, dropout_emb='True', reg=0,
        backprop_emb='False', batch_size=128, word2vec_init='False',
        n_folds=5, optimizer='adam', lr=.001, do_cv='False', metric='loss',
        callbacks='cb,ce,fl,cv,es', trainer='CNNSiameseTrainer', features='',
        input='abstracts', labels='None', fit_generator='True',
        loss='hinge', nb_train=1., nb_sample=1000, log_full='False', train_size=.97,
        summary_type='outcomes'):
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
    filter_lens = [int(filter_len) for filter_len in filter_lens.split(',')]
    backprop_emb = True if backprop_emb == 'True' else False
    dropout_emb = dropout_prob if dropout_emb == 'True' else 1e-100
    word2vec_init = True if word2vec_init == 'True' else False
    do_cv = True if do_cv == 'True' else False
    nb_filter /= len(filter_lens) # make it so there are only nb_filter *total* - NOT nb_filter*len(filter_lens)
    callbacks = callbacks.split(',')
    features = features.split(',') if features != '' else []
    inputs = [input, summary_type]
    labels = None if labels == 'None' else labels
    metric = None if metric == 'None' else metric
    fit_generator = True if fit_generator == 'True' else False

    # load data and supervision
    trainer = eval(trainer)(exp_group, exp_id, hyperparam_dict, summary_type)
    trainer.load_texts(inputs)
    trainer.load_auxiliary(features)
    trainer.load_labels(labels)

    # model
    trainer.build_model(nb_filter, filter_lens, nb_hidden, hidden_dim, dropout_prob,
            dropout_emb, backprop_emb, word2vec_init, reg)
    trainer.compile_model(metric, optimizer, lr, loss)

    # load cdnos sorted by the ones with the most studies so we pick those first when undersampling
    df = pd.read_csv('../data/extra/pico_cdsr.csv', index_col=0)
    cdnos = np.array(df.groupby('cdno').size().sort_values(ascending=False).index)

    # split into train and validation at the cdno-level
    from sklearn.cross_validation import train_test_split
    train_cdno_idxs, val_cdno_idxs = train_test_split(len(cdnos), train_size=train_size, random_state=1337)
    first_train = np.floor(len(train_cdno_idxs)*nb_train)
    train_cdno_idxs = np.sort(train_cdno_idxs)[:first_train.astype('int')] # take a subset (or all!) of the training cdnos
    val_cdno_idxs =  np.sort(val_cdno_idxs)
    train_cdnos, val_cdnos = set(cdnos[train_cdno_idxs]), set(cdnos[val_cdno_idxs])
    train_study_idxs = np.array(df[df.cdno.isin(train_cdnos)].index)
    val_study_idxs = np.array(df[df.cdno.isin(val_cdnos)].index)

    # train
    fold_idx, cdnos = 0, np.array(df.cdno) # fold is legacy
    history = trainer.train(train_study_idxs, val_study_idxs, nb_epoch, batch_size,
            callbacks, fold_idx, metric, fit_generator, cdnos, nb_sample, log_full)


if __name__ == '__main__':
    plac.call(main)
