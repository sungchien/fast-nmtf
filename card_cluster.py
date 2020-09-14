#!/usr/bin/env python

import numpy as np
import pandas as pd
from datetime import datetime

from fnmtf.loader import load_numpy, save_numpy
from fnmtf.engine import Engine
from fnmtf.common import *
from fnmtf.cod import nmtf_cod

X = load_numpy('data/card_sorting.npz')
if X is None:
    raise Exception("Unable to open file")
print(f'讀入資料{X.shape[0]}列，{X.shape[1]}行！')
X = X.astype(np.float64)

epsilon = 6
engine = Engine(epsilon=epsilon, parallel=1)

rank = 12
seed_ix = datetime.now().second
print(f'random seed {seed_ix}')
params = {'engine': engine, 'X': X, 'k': rank, 'k2': rank, 'seed': seed_ix, 'method': 'nmtf',
	'technique': 'cod', 'max_iter': 100, 'min_iter': 1, 'epsilon': epsilon,
	'verbose': False, 'store_history': True, 'store_results': False,
	'basename': 'aldigs', 'label': 'aldigs'}

factors, err = nmtf_cod(params)
U, S, V = factors
save_csv('U.csv', U)
save_csv('V.csv', V)
save_csv('S.csv', S)
print(f'Reconstruction error:{err[-1]}')