
import pandas as pd
from amc.helper import calc_implied_vols_from_prices, plot_implied_vols


# data = pd.read_csv('data/spx_opt.csv', parse_dates=['date', 'exdate'])
# new = calc_implied_vols_from_prices(data)
# plot_implied_vols(new)
# print(new[new.t_exp < 0.006])

import numpy as np

res1 = []
res2 = []
b = np.sqrt(0.6) / 2
for _ in range(10000):
    p = np.random.uniform(0.5 - b, 0.5 + b, 1)
    res1.append(np.random.binomial(10, p))
    res2.append(np.random.binomial(90, p) + res1[-1])

res1 = np.array(res1)
res2 = np.array(res2)
res = np.hstack([res1, res2])
print(np.corrcoef(res))