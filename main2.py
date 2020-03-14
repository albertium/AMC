
import pandas as pd
from amc.helper import calc_implied_vols_from_prices, plot_implied_vols


# data = pd.read_csv('data/spx_opt.csv', parse_dates=['date', 'exdate'])
# new = calc_implied_vols_from_prices(data)
# plot_implied_vols(new)
# print(new[new.t_exp < 0.006])

from amc.security import EuropeanCall, HeatSecurity
from amc.engine import FiniteDifferenceEngine, CrankNicolsonScheme, ExplicitScheme
from amc.pde import BlackScholesPDE1D, HeatPDE
from amc.helper import get_european_call_bs


# asset = 'AAPL'
# S = 248
# K = 300
# r = 254 / S - 1
# q = 0
# sig = 0.72
# t = 0.25
#
#
# sec = EuropeanCall(asset=asset, strike=K, tenor=t)
# pde = BlackScholesPDE1D(asset=asset, spot=S, r=r, q=q, sig=sig)
# engine = FiniteDifferenceEngine(sec, pde, CrankNicolsonScheme())
# vals, xs = engine.price({'t': 10, asset: 200}, scale=3)
#
# mask = (xs > 200) & (xs < 300)
# print(vals[mask])
# # print(ans.interp({asset: S}))
# print([get_european_call_bs(x, K, r, q, sig, t) for x in xs[mask]])

print('started')
n = 4
alpha = 0.25  # stability condition: alpha = dt / (dx)^2 <= 0.5
m = int((n / 4) ** 2 / alpha)  # range is from -2 to 2. See below

sec = HeatSecurity(1)
pde = HeatPDE()  # pde fix the atm vol to be 2
engine = FiniteDifferenceEngine(sec, pde, CrankNicolsonScheme())
ans, _ = engine.price({'t': m, 'Heat': n}, scale=1)
print(ans)