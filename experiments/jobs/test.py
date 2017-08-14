# example test experiment script

import os
import random

import numpy as np

from condor_create import make_exps


exp_group = os.path.basename(__file__).split('.')[-2]


args = {'-callbacks': ['cb,ss,fl,es,cv'],
        '-nb-epoch': [32],
        '-trainer': ['SharedCNNSiameseTrainer'],
        '-nb-train': [1.],
        '-target': ['abstracts'],
        '-reg': [0., 1e-6],
        '-dropout-prob': [0., .5],
}

make_exps(exp_group, args, grid_search=True)
