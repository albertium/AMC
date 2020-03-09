
import unittest
from amc.security import EuropeanCall, EuropeanPut
from amc.simulation import BlackScholes
from amc.engine import MonteCarloEngine
from amc.helper import get_european_call_bs, get_european_put_bs
from numpy.testing import assert_allclose


class TestVanilla(unittest.TestCase):
    def test_black_scholes(self):
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
