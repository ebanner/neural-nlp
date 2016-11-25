
# coding: utf-8

# # Mean Study Distance
# 
# Compute the mean similarity between studies which were included in each drug
# review and the studies which were explicitly excluded. The hope is that that
# the mean distance between studies in the same review is smaller than the mean
# distance to studies which were explicitly excluded!

# ### Choose Experiment Group & ID

# In[1]:

import pickle

import numpy as np
import pandas as pd

from scipy import spatial
from sklearn.preprocessing import normalize

import keras
import keras.backend as K


MODELS = ['pop', 'outcome']
EXCL_SETS = ['pop', 'outcome']
TEST_MODE = 0

EXP_DICT = {'pop': {'exp-group': 'big-populations',
                    'exp-id': 9},
            'outcome': {'exp-group': 'big-outcomes',
                        'exp-id': 7},
            'concat': {'exp-group': 'concateds',
                       'exp-id': 8},
}

# Load keras models for each pico element
FUNCS = {}
for model_type in MODELS:
    exp_group, exp_id = EXP_DICT[model_type]['exp-group'], EXP_DICT[model_type]['exp-id']
    model = keras.models.load_model('store/weights/{}/{}/0-loss.h5'.format(exp_group, exp_id))
    inputs = [model.inputs[0], K.learning_phase()]
    outputs = model.get_layer('study').output
    f = K.function(inputs, outputs)

    FUNCS[model_type] = f


def gen_upper_triangular(N=5):
    """Generate pairs of indexes to be passed to zip() so we can extract the
    upper triangle of a matrix.
    
    """
    for i in range(N):
        for j in range(i+1, N):
            yield i, j


if __name__ == '__main__':
    # Load abstracts and vectorize
    text_df = pd.read_csv('../preprocess/test_df.csv')
    vectorizer = pickle.load(open('../preprocess/abstracts.p'))
    X = vectorizer.texts_to_sequences(text_df.text)

    # List Drugs
    print 'Drugs...'
    for drug in text_df.groupby('drug').size().index.tolist():
        print drug
    print

    df = pd.DataFrame()
    for drug, drug_df in text_df.groupby('drug'):
        d = {'drug': drug} # populate this dict with results
        X_drug = X[drug_df.index] # narrow down to just this drug's abstracts
        in_idxs = np.argwhere(drug_df.label == 'inc').flatten()
        d['nb_in'] = len(in_idxs) # record number of included studies

        # Set up arrays for composite representation
        for model_type in MODELS:
            d['model'] = model_type
            f = FUNCS[model_type]

            # Compute mean similarity for studies in this drug's review
            H_in = f([X_drug[in_idxs], TEST_MODE])
            H_in = normalize(H_in)
            S_in = np.dot(H_in, H_in.T) # included similarity
            I, J = zip(*list(gen_upper_triangular(N=len(H_in))))
            in_score = S_in[I, J].mean()
            d['in_score'] = in_score # record review statistics

            # Loop over the exclusion sets
            for excl_set in EXCL_SETS:
                d['excl_set'] = excl_set

                out_idxs = np.argwhere(drug_df.label == excl_set).flatten()
                d['nb_out'] = len(out_idxs)

                H_out = f([X[out_idxs], TEST_MODE]) if len(out_idxs) else np.nan
                H_out = normalize(H_out) if len(out_idxs) else np.nan # normalize vectors
                d['out_score'] = np.dot(H_out, H_in.T).mean()

                score_df = pd.DataFrame(d, index=[0])
                df = pd.concat([df, score_df])
                df.to_csv('results.csv', index=False) # keep appending
                print df
                print
