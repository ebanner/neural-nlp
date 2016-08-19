# example test experiment script

import os
import random

import numpy as np

from condor_create import make_exps


exp_group = os.path.basename(__file__).split('.')[-2]


args = {'-nb-filter': [1, 2, 3, 4, 5],
        '-hidden-dim': [32, 64]
}

make_exps(exp_group, args, grid_search=True)
