#!/usr/bin/env python

import numpy as np
import pandas as pd

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

err_set = []
for seed_ix in range(1, 21, 1):
	for i in range(1, 21, 1):
		params = {'engine': engine, 'X': X, 'k': i, 'k2': i, 'seed': seed_ix, 'method': 'nmtf',
			'technique': 'cod', 'max_iter': 100, 'min_iter': 1, 'epsilon': epsilon,
			'verbose': False, 'store_history': True, 'store_results': False,
			'basename': 'aldigs', 'label': 'aldigs'}

		factors, err = nmtf_cod(params)
	
		print(f'與原先資料的誤差 (Reconstruction error): {err[-1]}')
		err_set.append((seed_ix, i, err[-1]))
		
err_df = pd.DataFrame(err_set, columns=['seed', 'k', 'err'])
err_df.to_csv('card_sorting_err.csv')