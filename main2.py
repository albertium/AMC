
import pandas as pd
import numpy as np
from amc.helper import calc_implied_vols_from_prices, plot_implied_vols


# data = pd.read_csv('data/spx_opt.csv', parse_dates=['date', 'exdate'])
# new = calc_implied_vols_from_prices(data)
# plot_implied_vols(new)
# print(new[new.t_exp < 0.006])

from amc.security import EuropeanCall, HeatSecurity
from amc.engine import FiniteDifferenceEngine, CrankNicolsonScheme, ExplicitScheme
from amc.pde import BlackScholesPDE1D, HeatPDE
from amc.helper import get_european_call_bs


asset = 'AAPL'
S = 248
K = 300
r = 254 / S - 1
q = 0
sig = 0.72
t = 0.25
# S, K, r, q, sig, t = 100, 100, 0.05, 0.02, 0.35, 1

sec = EuropeanCall(asset=asset, strike=K, tenor=t)
pde = BlackScholesPDE1D(asset=asset, spot=S, r=r, q=q, sig=sig)
engine = FiniteDifferenceEngine(sec, pde, CrankNicolsonScheme())
vals, xs = engine.price({'t': 100, asset: 600}, scale=5)
ans = np.interp(S, xs, vals)

# mask = (xs > 200) & (xs < 300)
# print(vals[mask])
# print(ans.interp({asset: S}))
print(ans)
print(get_european_call_bs(S, K, r, q, sig, t))
# print([get_european_call_bs(x, K, r, q, sig, t) for x in xs[mask]])