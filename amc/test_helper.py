
import unittest
import numpy as np
from numpy.testing import assert_array_almost_equal
from .helper import get_european_call, get_european_put, get_european_call_bs, get_european_put_bs, get_implied_vol_call


class TestBlackScholes(unittest.TestCase):
    def test_black_formula(self):
        S, K, r, q, sig, T = 100, np.array([100, 90]), 0.02, 0.03, np.array([0.2, 0.3]), 0.5
        F = S * np.exp((r - q) * T)
        df = np.exp(-r * T)
        ans = get_european_call(F, K, df, sig, T)
        assert_array_almost_equal(ans, [5.323767401196562, 13.49375893227614], 11)

        ans = get_european_put(F, K, df, sig, T)
        assert_array_almost_equal(ans, [5.817556815807109, 4.087050009394993], 11)

    def test_black_scholes_formula(self):
        S, K, r, q, sig, T = 100, 100, 0.02, 0.03, 0.2, 0.5
        F = S * np.exp((r - q) * T)
        df = np.exp(-r * T)
        ans = get_european_call_bs(S, K, r, q, sig, T)
        self.assertAlmostEqual(ans, 5.323767401196562, 11)

        ans = get_european_put_bs(S, K, r, q, sig, T)
        self.assertAlmostEqual(ans, 5.817556815807109, 11)

    def test_implied_vol(self):
        S, K, r, q, sig, T = 100, np.array([100, 90]), 0.02, 0.03, np.array([0.2, 0.3]), np.array([0.5, 1])
        F = S * np.exp((r - q) * T)
        df = np.exp(-r * T)

        prices = get_european_call(F, K, df, sig, T)
        ans = get_implied_vol_call(prices, F, K, df, T)
        assert_array_almost_equal(ans, [0.2, 0.3], 9)