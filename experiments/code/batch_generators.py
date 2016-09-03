# batch generators

def pair_generator(nb_train, cdnos, top_cdnos, nb_sample=128, phase=0, exact_only=False):
    """Generator for generating batches of source_idxs to target_idxs

    Parameters
    ----------
    nb_train : total number of studies (and abstracts)
    cdnos : mapping from study indexes to their cdno
    top_cdnos : set of `cdnos.unique()`
    nb_sample : number of studies to evaluate
    phase : 1 for train and 0 for test
    target : token for determining what `target_idxs` will be

    Returns {(study_idxs, study_idxs)} if `target` == 'study' and {(study_idxs,
    summary_idxs)} if `target` == 'summary. In practice, `target` == 'study'
    when when called by StudySimilarityLogger and `target` == 'summary' when
    called by training generator.

    """
    while True:
        # sample study indices
        study_idxs = np.random.choice(nb_train, size=nb_sample)

        # build dicts to enable sampling corrupt studies
        all_cdno_idxs = set(np.arange(nb_train))
        cdno2corrupt_study_idxs = {}
        for cdno in top_cdnos:
            cdno_idxs = set(np.argwhere(cdnos == cdno).flatten())
            cdno2corrupt_study_idxs[cdno] = list(all_cdno_idxs - cdno_idxs)
            
        target_idxs = study_idxs.copy()
        if not exact_only:
            cdno2valid_study_idxs = {}
            for cdno in top_cdnos:
                cdno_idxs = set(np.argwhere(cdnos == cdno).flatten())
                cdno2valid_study_idxs[cdno] = cdno_idxs
                
            # find study idxs in the same study
            valid_range = range(nb_sample/2)
            for i, study_idx in zip(valid_range, study_idxs[:nb_sample/2]):
                cdno = cdnos[study_idx]
                valid_study_idxs = cdno2valid_study_idxs[cdno]
                valid_study_idxs = valid_study_idxs - set([study_idx]) # remove study iteself from consideration
                valid_study_idx = np.random.choice(list(valid_study_idxs))
                target_idxs[i] = valid_study_idx
                
        # always compute corrupt idxs same way
        corrupt_range = range(nb_sample/2, nb_sample)
        for j, study_idx in zip(corrupt_range, study_idxs[nb_sample/2:]):
            cdno = cdnos[study_idx]
            corrupt_study_idxs = cdno2corrupt_study_idxs[cdno]
            corrupt_study_idx = np.random.choice(corrupt_study_idxs)
            target_idxs[j] = corrupt_study_idx

        yield study_idxs, target_idxs

def study_summary_generator(**kwargs):
    """Wrapper generator around pair_generator() for yielding batches of
    ([study, summary], y) pairs.

    The first half of pairs are of the form ([study, corresponding-summary], y)
    and second half are of the form ([study, summary-from-different-review], y).

    """
    study_summary_batch = pair_generator(nb_train=len(kwargs['X_summary']), exact_only=True, **kwargs)

    nb_sample = kwargs['nb_sample']
    y = np.zeros(nb_sample)
    y[:nb_sample/2] = 1 # first half of samples are good always

    X_study, X_summay = kwargs['X_study'], kwargs['X_summary']
    while True:
        study_idxs, summary_idxs = next(study_summary_batch)

        yield [X_study[study_idxs], X_summary[summary_idxs]], y
