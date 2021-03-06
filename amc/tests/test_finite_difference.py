
import unittest
import numpy as np
from numpy.testing import assert_almost_equal, assert_allclose
from scipy.sparse import diags
from scipy.interpolate import interp2d

from amc.grid import GridSlice
from amc.security import HeatSecurity, EuropeanCall, Security, ExchangeOption
from amc.engine import FiniteDifferenceEngine, FiniteDifferenceScheme
from amc.engine import ExplicitScheme, ImplicitScheme, CrankNicolsonScheme, DouglasScheme, HundsdorferVerwerScheme
from amc.pde import PDE, HeatPDE, BlackScholesPDE1D, BlackScholesPDE2D
from amc.helper import get_european_call_bs
from amc.data import EquityFactor


class TestLinearAlgebra(unittest.TestCase):
    def test_forward_step(self):
        n = 5
        a = np.random.rand(3, n)
        b = np.random.rand(n + 2)
        ans = np.zeros(n + 2)
        FiniteDifferenceScheme.forward_step(ans, b, a)

        c = diags([a[0, 1:], a[1], a[2, :-1]], [-1, 0, 1]).toarray()
        expected = c @ b[1: -1]
        expected[0] += a[0, 0] * b[0]
        expected[-1] += a[2, -1] * b[-1]
        assert_almost_equal(ans[1: -1], expected, decimal=15)

    def test_backward_step(self):
        # randomized test, may fail sometime, just need to re-run
        n = 10
        a = np.random.rand(3, n)
        b = np.random.rand(n + 2)
        a_saved = a.copy()
        expected = b.copy()  # make sure b is not overwrite

        # inverse of inverse gives back the original vector
        inv = np.zeros(n + 2)
        ans = np.zeros(n + 2)
        ans[0], ans[-1] = b[0], b[-1]  # need to set the Dirichlet boundaries first

        # Mock class to circumvent the abstract method failure
        class Scheme(FiniteDifferenceScheme):

            def step(self, curr: GridSlice, prev: GridSlice, pde: PDE, sec: Security) -> None:
                pass

        scheme = Scheme()
        scheme.forward_step(inv, b, a)
        inv_saved = inv.copy()
        scheme.backward_step(ans, inv, a)

        assert_almost_equal(ans, expected, decimal=14)
        assert_almost_equal(a, a_saved, decimal=18)  # make sure a is not overwritten
        assert_almost_equal(b, expected, decimal=18)  # make sure b is not overwritten
        assert_almost_equal(inv, inv_saved, decimal=18)  # make sure inv is not overwritten in backward step


class TestFiniteDifference(unittest.TestCase):

    def test_explicit_scheme(self):
        # explicit is not stable but the more steps you put, the more accuracy you get
        n = 128
        alpha = 0.25  # stability condition: alpha = dt / (dx)^2 <= 0.5
        m = int((n / 4) ** 2 / alpha)  # range is from -2 to 2. See below

        sec = HeatSecurity(1)
        pde = HeatPDE()  # pde fix the atm vol to be 2
        engine = FiniteDifferenceEngine(sec, pde, ExplicitScheme())
        ans, _ = engine.price({'t': m, 'Heat': n}, scale=1)  # scale * atm vol = 1 * 2 = 2 decides the boundaries
        expected = np.exp(np.linspace(-2, 2, n) + 1)  # solution is exp(x + tau) = exp(x + 1)
        assert_allclose(ans, expected, rtol=0.00005, atol=0)

    def test_implicit_scheme(self):
        # implicit scheme is robust but accuracy increases much slower than explicit
        n = 128  # range steps
        m = 4096  # time steps to match explicit scheme

        sec = HeatSecurity(1)
        pde = HeatPDE()  # pde fix the atm vol to be 2
        engine = FiniteDifferenceEngine(sec, pde, ImplicitScheme())
        ans, _ = engine.price({'t': m, 'Heat': n}, scale=1)  # scale * atm vol = 1 * 2 = 2 decides the boundaries
        expected = np.exp(np.linspace(-2, 2, n) + 1)  # solution is exp(x + tau) = exp(x + 1)
        assert_allclose(ans, expected, rtol=0.0005, atol=0)

    def test_crank_nicolson_scheme(self):
        # CN is robust but accuracy doesn't increase after certain step size
        # Is it because of backward step or implementation?
        n = 128  # range steps
        m = 128  # time steps

        sec = HeatSecurity(1)
        pde = HeatPDE()  # pde fix the atm vol to be 2
        engine = FiniteDifferenceEngine(sec, pde, CrankNicolsonScheme())
        ans, _ = engine.price({'t': m, 'Heat': n}, scale=1)  # scale * atm vol = 1 * 2 = 2 decides the boundaries
        expected = np.exp(np.linspace(-2, 2, n) + 1)  # solution is exp(x + tau) = exp(x + 1)
        assert_allclose(ans, expected, rtol=0.0001, atol=0)

    def test_black_scholes(self):
        m = 100  # time steps
        n = 600

        asset = 'AAPL'
        s, k = 248, 300
        r, q, sig, t = 254 / s - 1, 0, 0.72, 0.25

        sec = EuropeanCall(asset=asset, strike=k, tenor=t)
        pde = BlackScholesPDE1D(asset=asset, spot=s, r=r, q=q, sig=sig)
        engine = FiniteDifferenceEngine(sec, pde, CrankNicolsonScheme())
        ans, states = engine.price({'t': m, asset: n}, scale=5)
        xs = states[asset]
        mask = (xs > k * 0.8) & (xs < k * 1.2)
        ans = ans[mask]
        xs = xs[mask]
        expected = [get_european_call_bs(x, k, r, q, sig, t) for x in xs]
        assert_allclose(ans, expected, atol=0, rtol=0.0005)

    def test_douglas_scheme(self):

        s1, q1, sig1 = 90, 0, 0.3
        s2, q2, sig2 = 110, 0, 0.4
        r = 0.01
        t = 0.4

        eq1 = EquityFactor('AAPL', s1, q1, sig1)
        eq2 = EquityFactor('MSFT', s2, q2, sig2)
        sec = ExchangeOption(eq1, eq2, t)

        # positive correlation
        rho = 0.4
        pde = BlackScholesPDE2D(eq1, eq2, r, rho)
        engine = FiniteDifferenceEngine(sec, pde, DouglasScheme())
        vals, states = engine.price({'t': 50, 'AAPL': 200, 'MSFT': 210}, scale=5)
        f = interp2d(states['MSFT'], states['AAPL'], vals)
        ans = f(s2, s1)[0]
        expected = get_european_call_bs(s1, s2, 0, 0, np.sqrt(sig1 ** 2 + sig2 ** 2 - 2 * rho * sig1 * sig2), t)
        assert_allclose(ans, expected, atol=0, rtol=0.0005)

        # negative correlation
        rho = -0.4
        pde = BlackScholesPDE2D(eq1, eq2, r, rho)
        engine = FiniteDifferenceEngine(sec, pde, DouglasScheme())
        vals, states = engine.price({'t': 50, 'AAPL': 200, 'MSFT': 210}, scale=5)
        f = interp2d(states['MSFT'], states['AAPL'], vals)
        ans = f(s2, s1)[0]
        expected = get_european_call_bs(s1, s2, 0, 0, np.sqrt(sig1 ** 2 + sig2 ** 2 - 2 * rho * sig1 * sig2), t)
        assert_allclose(ans, expected, atol=0, rtol=0.002)

    def test_hundsdorfer_verwer_scheme(self):

        s1, q1, sig1 = 90, 0, 0.3
        s2, q2, sig2 = 110, 0, 0.4
        r = 0.01
        t = 0.4

        eq1 = EquityFactor('AAPL', s1, q1, sig1)
        eq2 = EquityFactor('MSFT', s2, q2, sig2)
        sec = ExchangeOption(eq1, eq2, t)

        # positive correlation
        rho = 0.4
        pde = BlackScholesPDE2D(eq1, eq2, r, rho)
        engine = FiniteDifferenceEngine(sec, pde, HundsdorferVerwerScheme())
        vals, states = engine.price({'t': 50, 'AAPL': 200, 'MSFT': 210}, scale=5)
        f = interp2d(states['MSFT'], states['AAPL'], vals)
        ans = f(s2, s1)[0]
        expected = get_european_call_bs(s1, s2, 0, 0, np.sqrt(sig1 ** 2 + sig2 ** 2 - 2 * rho * sig1 * sig2), t)
        # assert_allclose(ans, expected, atol=0, rtol=0.0005)

        # negative correlation
        rho = -0.4
        pde = BlackScholesPDE2D(eq1, eq2, r, rho)
        engine = FiniteDifferenceEngine(sec, pde, DouglasScheme())
        vals, states = engine.price({'t': 50, 'AAPL': 200, 'MSFT': 210}, scale=5)
        f = interp2d(states['MSFT'], states['AAPL'], vals)
        ans = f(s2, s1)[0]
        expected = get_european_call_bs(s1, s2, 0, 0, np.sqrt(sig1 ** 2 + sig2 ** 2 - 2 * rho * sig1 * sig2), t)
        # assert_allclose(ans, expected, atol=0, rtol=0.002)
