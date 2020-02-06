
import numpy as np
import time
import cProfile

from amc import Pricer
from amc.security import EuropeanOption
from amc.simulation import BlackScholes
from amc.fitter import LASSOFitter
from amc.helper import get_blackscholes_call


if __name__ == '__main__':
    S = 100
    K = 100
    r = 0.01
    q = 0
    sig = 0.3
    T = 0.5
    M = 1
    N = int(1E5)

    sec = EuropeanOption(asset='stock', strike=K, tenor=T)
    bs = BlackScholes(spot=S, interest=r, dividend=q, volatility=sig)
    lasso = LASSOFitter()
    pricer = Pricer([sec], model=bs, fitter=lasso)

    # pricing
    # cProfile.run('np.array([pricer.price(num_steps=M, num_paths=N) for _ in range(1000)])', sort='cumtime')
    start = time.time()
    ans = np.array([pricer.price(num_steps=M, num_paths=N) for _ in range(1000)])
    real = get_blackscholes_call(S, K, r, q, sig, T)
    error = ans - real
    print(N)
    print(f'Error: {np.sqrt(np.mean(error ** 2)) / real:.3%}')
    print(f'Time used: {time.time() - start:.1f}s')
