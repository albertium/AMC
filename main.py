
from amc import Pricer
from amc.security import EuropeanOption
from amc.simulation import BlackScholes
from amc.fitter import LASSOFitter


if __name__ == '__main__':
    S = 100
    K = 100
    r = 0.01
    q = 0
    sig = 0.3
    T = 0.5
    M = 2
    N = int(1E6)

    sec = EuropeanOption(asset='stock', strike=K, tenor=T)
    bs = BlackScholes(spot=S, interest=r, dividend=q, volatility=sig)
    lasso = LASSOFitter()
    pricer = Pricer([sec], model=bs, fitter=lasso)
    ans = pricer.price(num_steps=M, num_paths=N)
    print(ans)