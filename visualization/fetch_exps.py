# coding: utf-8

import plac


dirs = ['train',
        'hyperparams',
        'weights'
]

store_dir='/u/ebanner/scratch/neural-nlp/experiments/store'


def canonicalize(dir, exp_id):
    csv_dirs = {'hyperparams'}
    return '{}.csv'.format(exp_id) if dir in csv_dirs else exp_id


@plac.annotations(
        exps=('experiments to fetch', 'option', None, str),
)
def main(exps=''):
    exp_groups, exp_ids = zip(*[exp.split(':') for exp in exps.split(',')]) if exps else ([], [])
    
    if exp_groups and exp_ids:
        for dir in dirs:
            get_ipython().system(u'mkdir -p store/$dir')
            for exp_group, exp_id in zip(exp_groups, exp_ids):
                exp_id = canonicalize(dir, exp_id) # hack because exp group/id directory structure is not uniform
                get_ipython().system(u'mkdir -p store/$dir/$exp_group')
                get_ipython().system(u'scp -r submit64.cs.utexas.edu:$store_dir/$dir/$exp_group/$exp_id store/$dir/$exp_group')
    else:
        print 'removing everything...'
        get_ipython().system(u'rm -rf store')
        get_ipython().system(u'mkdir store')
        
        for dir in dirs:
            get_ipython().system(u'scp -r submit64.cs.utexas.edu:$store_dir/$dir store')


if __name__ == '__main__':
    plac.call(main)
