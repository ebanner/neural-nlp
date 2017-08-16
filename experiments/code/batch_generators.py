import numpy as np
import pandas as pd


def cdno_matrix_generator(X, cdnos, nb_sample, seed):
    """Wrapper generator around same_study_generator() and

    Parameters
    ----------
    cdnos : pandas series which maps each study to its cdno
    seed : the random seed to use
    nb_sample : number of samples to return
    cdno_matching : yield same indexes for positive examples if `True` else yield
    indexes just in the same study if `False`

    The first half of pairs are of the form ([study, corresponding-summary],  1)
    and second half are of the form ([study, summary-from-different-review], neg_nb).

    """
    fields = ['abstract', 'population', 'intervention', 'outcome']
    random = np.random.RandomState(seed) if seed else np.random # for reproducibility!
    nb_studies, cdno_set = len(cdnos), set(np.unique(cdnos))

    # dict to sample studies with same cdno
    cdno2study_idxs = {}
    for cdno in cdno_set:
        study_idxs = np.argwhere(cdnos == cdno).flatten()
        cdno2study_idxs[cdno] = set(study_idxs)

    all_study_idxs = set(cdnos.index)
    while True:
        study_idxs = random.choice(nb_studies, size=nb_sample, replace=False)

        same_study_idxs = []
        for study_idx in study_idxs:
            cdno = cdnos[study_idx]
            valid_study_idxs = cdno2study_idxs[cdno]
            valid_study_idxs = valid_study_idxs - set([study_idx]) # remove study iteself from consideration
            valid_study_idxs = list(valid_study_idxs)
            same_study_idx = random.choice(valid_study_idxs)
            same_study_idxs.append(same_study_idx)

        different_study_idxs = []
        for study_idx in study_idxs:
            cdno = cdnos[study_idx]
            valid_study_idxs = cdno2study_idxs[cdno]
            corrupt_study_idxs = list(all_study_idxs - valid_study_idxs)
            different_study_idx = random.choice(corrupt_study_idxs)
            different_study_idxs.append(different_study_idx)

        X_study = {'same_'+field: X[field][study_idxs] for field in fields}
        X_same_study = {'valid_'+field: X[field][same_study_idxs] for field in fields}
        X_different_study = {'corrupt_'+field: X[field][different_study_idxs] for field in fields}

        X_batch = dict(X_study.items() + X_same_study.items() + X_different_study.items())
        yield X_batch

def bg1(X, cdnos, nb_sample=128, seed=1337):
    """Batch generator 1

    Samples a batch dict containing

    - {X_a, X_p, X_i, X_o,
            X_p',
            X_p~
      }

    """
    fields = ['same_abstract',
              'same_population',
              'same_intervention',
              'same_outcome',
              'valid_population',
              'corrupt_population',
    ]

    y_batch = {'same_population_score': np.ones(nb_sample),
               'valid_population_score': np.ones(nb_sample),
               'corrupt_population_score': np.full(shape=nb_sample, fill_value=-1),
               'neg_same_population_norm': np.random.randn(nb_sample),
               'neg_valid_population_norm': np.random.randn(nb_sample),
               'neg_corrupt_population_norm': np.random.randn(nb_sample),
               'same_intervention_norm': np.random.randn(nb_sample),
               'same_outcome_norm': np.random.randn(nb_sample),
    }

    batch = cdno_matrix_generator(X, cdnos, nb_sample, seed)
    while True:
        X_batch = next(batch)
        X_batch = {key: X_ for key, X_ in X_batch.items() if key in fields}
        yield X_batch, y_batch

def bg2(X, cdnos, nb_sample=128, seed=1337):
    """Batch generator 2

    Samples a batch dict containing

    - {X_a,
       X_a',
       X_a~
      }

    """
    fields = ['same_abstract', 'valid_abstract', 'corrupt_abstract']

    batch = cdno_matrix_generator(X, cdnos, nb_sample, seed)
    X_batch = next(batch)
    X_batch = {key: X_ for key, X_ in X_batch.items() if key in fields}

    yield X_batch
