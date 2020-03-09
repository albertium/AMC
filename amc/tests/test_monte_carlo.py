
import unittest
from amc.security import EuropeanCall, EuropeanPut, AmericanCall, AmericanPut
from amc.simulation import BlackScholes
from amc.engine import MonteCarloEngine
from amc.fitter import LASSOFitter
from amc.helper import get_european_call_bs, get_european_put_bs
from numpy.testing import assert_allclose


class TestVanilla(unittest.TestCase):
    def test_european(self):
        s = 100
        k = 100
        r = 0.02
        q = 0.05
        sig = 0.3
        t = 1
        m = 1
        n = int(1E7)

        # test call and put
        secs = [EuropeanCall('stock', k, t), EuropeanPut('stock', k, t)]
        bs = BlackScholes(spot=s, interest=r, dividend=q, volatility=sig)
        engine = MonteCarloEngine(secs, model=bs)
        price = engine.price(m, n)
        actual = [get_european_call_bs(s, k, r, q, sig, t), get_european_put_bs(s, k, r, q, sig, t)]
        assert_allclose(actual, price, rtol=1e-4, atol=1e-2)  # test both relative diff and absolute diff

    def test_american_to_bs(self):
        """
        American call and put without interest and dividend should equal Black Scholes prices
        """
        s = 100
        k = 100
        r = 0
        q = 0
        sig = 0.3
        t = 1
        m = 50
        n = int(1E5)

        # American options without interest and dividend should equal Black Scholes prices
        secs = [AmericanCall('stock', k, t), AmericanPut('stock', k, t)]
        bs = BlackScholes(spot=s, interest=r, dividend=q, volatility=sig)
        lasso = LASSOFitter()
        engine = MonteCarloEngine(secs, model=bs, fitter=lasso)
        price = engine.price(m, n)
        actual = [get_european_call_bs(s, k, r, q, sig, t), get_european_put_bs(s, k, r, q, sig, t)]
        assert_allclose(actual, price, rtol=0.01, atol=0.1)

    def test_american_fixed(self):
        """
        Test American options WITH interest and dividend against pre-calculated fixed numbers
        """
        s = 100
        k = 100
        r = 0.02
        q = 0.03
        sig = 0.3
        t = 1
        m = 50
        n = int(1E5)

        # American options without interest and dividend should equal Black Scholes prices
        secs = [AmericanCall('stock', k, t), AmericanPut('stock', k, t)]
        bs = BlackScholes(spot=s, interest=r, dividend=q, volatility=sig)
        lasso = LASSOFitter()
        engine = MonteCarloEngine(secs, model=bs, fitter=lasso)
        price = engine.price(m, n)
        assert_allclose([11.34603464348194, 12.192072067889066], price, rtol=0.01, atol=0.1)
