
from amc.security import EuropeanCall
from amc.pde import BlackScholesPDE1D
from amc.engine import FiniteDifferenceEngine, CrankNicolsonScheme

m = 100  # time steps
n = 600

asset = 'AAPL'
s, k = 248, 300
r, q, sig, t = 254 / s - 1, 0, 0.72, 0.25

sec = EuropeanCall(asset=asset, strike=k, tenor=t)
pde = BlackScholesPDE1D(asset=asset, spot=s, r=r, q=q, sig=sig)
engine = FiniteDifferenceEngine(sec, pde, CrankNicolsonScheme())
ans, states = engine.price({'t': m, asset: n}, scale=5)
print(ans)