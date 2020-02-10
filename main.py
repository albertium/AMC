
from amc import PricingEngine
from amc.security import EuropeanOption, AmericanPut, AmericanCall
from amc.simulation import BlackScholes
from amc.fitter import LASSOFitter
from amc import helper, util

if __name__ == '__main__':
    S = 100
    K = 100
    r = 0.02
    q = 0.05
    sig = 0.3
    T = 1
    M = 50
    N = int(1E5)

    sec = AmericanCall('stock', K, T)
    bs = BlackScholes(spot=S, interest=r, dividend=q, volatility=sig)
    lasso = LASSOFitter()
    engine = PricingEngine([sec], model=bs, fitter=lasso)

    # pricing
    # ans = engine.price(num_steps=M, num_paths=N)
    real = helper.get_european_put_bs(S, K, r, q, sig, T)

    util.get_pricing_stats(engine, real, M, N, repeat=20)

    # print(ans)
    print(real)
