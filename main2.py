
import pandas as pd
from amc.helper import calc_implied_vols_from_prices, plot_implied_vols


# data = pd.read_csv('data/spx_opt.csv', parse_dates=['date', 'exdate'])
# new = calc_implied_vols_from_prices(data)
# plot_implied_vols(new)
# print(new[new.t_exp < 0.006])

from amc.security import HeatSecurity
from amc.engine import FiniteDifferenceEngine, ExplicitScheme
from amc.pde import HeatPDE


sec = HeatSecurity(1)
pde = HeatPDE()
engine = FiniteDifferenceEngine(sec, pde, ExplicitScheme())
ans = engine.price({'t': 42, 'Heat': 20}, scale=1)
print(ans)
