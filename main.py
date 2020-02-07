
from amc import PricingEngine
from amc.security import EuropeanOption, AmericanOption, PutPayoff, CallPayoff
from amc.simulation import BlackScholes
from amc.fitter import LASSOFitter
from amc import helper, util

if __name__ == '__main__':
    S = 100
    K = 100
    r = 0.05
    q = 0
    sig = 0.3
    T = 1
    M = 1
    N = int(1E5)

    sec = EuropeanOption(PutPayoff(asset='stock', strike=K), tenor=T)
    bs = BlackScholes(spot=S, interest=r, dividend=q, volatility=sig)
    lasso = LASSOFitter()
    engine = PricingEngine([sec], model=bs, fitter=lasso)

    # pricing
    # ans = engine.price(num_steps=M, num_paths=N)
    real = helper.get_european_put(S, K, r, q, sig, T)

    util.get_pricing_stats(engine, real, M, N)

    # print(ans)
    print(real)
