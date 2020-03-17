
import pandas as pd
import numpy as np
from scipy.interpolate import interp2d
from amc.helper import calc_implied_vols_from_prices, plot_implied_vols


# data = pd.read_csv('data/spx_opt.csv', parse_dates=['date', 'exdate'])
# new = calc_implied_vols_from_prices(data)
# plot_implied_vols(new)
# print(new[new.t_exp < 0.006])

from amc.security import EuropeanCall, HeatSecurity, ExchangeOption
from amc.engine import FiniteDifferenceEngine, CrankNicolsonScheme, ExplicitScheme, DouglasScheme
from amc.pde import BlackScholesPDE1D, HeatPDE, BlackScholesPDE2D
from amc.helper import get_european_call_bs
from amc.data import EquityFactor
import cProfile


print('start')
asset = 'AAPL'
s1, q1, sig1 = 90, 0, 0.3
s2, q2, sig2 = 110, 0, 0.4
r = 0.01
t = 0.4
rho = 0.4
# S, K, r, q, sig, t = 100, 100, 0.05, 0.02, 0.35, 1
eq1 = EquityFactor('AAPL', s1, q1, sig1)
eq2 = EquityFactor('MSFT', s2, q2, sig2)
sec = ExchangeOption(eq1, eq2, t)
pde = BlackScholesPDE2D(eq1, eq2, r, rho)
engine = FiniteDifferenceEngine(sec, pde, DouglasScheme())
# profile = cProfile.run("vals, states = engine.price({'t': 100, 'AAPL': 100, 'MSFT': 100}, scale=5)", sort=2)
# print(profile)
vals, states = engine.price({'t': 10, 'AAPL': 10, 'MSFT': 8}, scale=5)
f = interp2d(states['MSFT'], states['AAPL'], vals)
ans = f(s2, s1)

# mask = (xs > 200) & (xs < 300)
# print(vals[mask])
# print(ans.interp({asset: S}))
print(ans)
print(get_european_call_bs(s1, s2, 0, 0, np.sqrt(sig1 ** 2 + sig2 ** 2 - 2 * rho * sig1 * sig2), t))
# print([get_european_call_bs(x, K, r, q, sig, t) for x in xs[mask]])