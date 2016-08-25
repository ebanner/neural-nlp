import tempfile

import itertools

from collections import OrderedDict

from IPython import get_ipython


def args_generator(args, num_exps, grid_search):
    """Generates list of tuples corresponding to args hyperparam dict

    Parameters
    ----------
    args : dict of hyperparam values

        e.g. {p1: g1(), p2: g2(), ..., pn: gn()},

            where pi is the i'th parameter and gi is the generator which
            generates a random value for pi.

    """
    od = OrderedDict(args)

    if grid_search:
        keys = [param for param in od]

        for i, vals in enumerate(itertools.product(*[value for param, value in od.items()])):
            yield zip(keys + ['-exp-id'], [str(val) for val in vals] + [str(i)])
    else:
        for _ in range(num_exps):
            args_setting = [(pname, str(next(pvalue))) for pname, pvalue in od.items()]

            yield args_setting

def make_exp(exp_group, args):
    """Perform setup work for a condor experiment
    
    Parameters
    ----------
    exp_group : name of this experiment group
    args : list of (arg_str, arg_val) tuples

    1. Remove any output from a previous experiment with the same name
    2. Do the same thing for weights
    3. Create the condor job file
    
    """
    equalized_args = ['='.join(tup) for tup in args]
    stripped_args = [arg.lstrip('-') for arg in equalized_args]
    exp_name = [pvalue for (pname, pvalue) in args if pname == '-exp-id'][0]
    
    unrolled_args = [arg for arg_tup in args for arg in arg_tup]
    arg_str = ' '.join(unrolled_args) + ' -exp-group ' + exp_group

    get_ipython().system(u'mkdir -p ../store/output/$exp_group/$exp_name')
    get_ipython().system(u'mkdir -p ../store/weights/$exp_group/$exp_name')
    get_ipython().system(u'mkdir -p ../store/probas/$exp_group')
    get_ipython().system(u'mkdir -p ../store/train/$exp_group/$exp_name')
    get_ipython().system(u'mkdir -p ../store/hyperparams/$exp_group')
    get_ipython().system(u'mkdir -p ../store/models/$exp_group')
    get_ipython().system(u'mkdir -p ../store/study_vecs/$exp_group')

    get_ipython().system(u'mkdir -p exps/$exp_group')
    get_ipython().system(u"sed 's/ARGUMENTS/$arg_str/g' job_template | sed 's/EXP_GROUP/$exp_group/g' | sed 's/EXPERIMENT/$exp_name/g' > exps/$exp_group/$exp_name")
    
def make_exps(exp_group, args, num_exps=32, grid_search=False, baseline_exp_groups=[]):
    """Wrapper around make_exp()
    
    Call make_exp() with `num_exps` number of experiments.
    
    """
    args_list = list(args_generator(args, num_exps, grid_search))

    # Remove exp_group directories!
    get_ipython().system(u'rm -rf exps/$exp_group')
    get_ipython().system(u'rm -rf ../store/output/$exp_group')
    get_ipython().system(u'rm -rf ../store/weights/$exp_group')
    get_ipython().system(u'rm -rf ../store/probas/$exp_group')
    get_ipython().system(u'rm -rf ../store/train/$exp_group')
    get_ipython().system(u'rm -rf ../store/hyperparams/$exp_group')
    get_ipython().system(u'rm -rf ../store/models/$exp_group')
    get_ipython().system(u'rm -rf ../store/study_vecs/$exp_group')

    for i, args_setting in enumerate(args_list):
        make_exp(exp_group, args_setting)

    # Copy in existing experiments - for learning curve visualizations
    for baseline_exp_group in baseline_exp_groups:
        get_ipython().system(u'cp -r ../output/$baseline_exp_group/* ../output/$exp_group')


if __name__ == '__main__':
    # Example experiment group!

    args = {'-nb-filter': [1, 2, 3, 4, 5], '-hidden-dim': [32, 64]}
    make_exps(exp_group='test', args=args, grid_search=True)
