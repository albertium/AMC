
import pandas as pd
from amc.helper import calc_implied_vols_from_prices


data = pd.read_csv('data/spx_opt.csv', parse_dates=['date', 'exdate'])
new = calc_implied_vols_from_prices(data)
print(new[new.t_exp < 0.006])
