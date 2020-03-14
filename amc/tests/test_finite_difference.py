
import unittest
import numpy as np
from numpy.testing import assert_almost_equal, assert_allclose
from scipy.sparse import diags

from amc.security import HeatSecurity
from amc.engine import FiniteDifferenceEngine, FiniteDifferenceScheme
from amc.engine import ExplicitScheme, ImplicitScheme, CrankNicolsonScheme
from amc.pde import HeatPDE


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
        FiniteDifferenceScheme.forward_step(inv, b, a)
        inv_saved = inv.copy()
        FiniteDifferenceScheme.backward_step(ans, inv, a)

        assert_almost_equal(ans, expected, decimal=14)
        assert_almost_equal(a, a_saved, decimal=18)  # make sure a is not overwritten
        assert_almost_equal(b, expected, decimal=18)  # make sure b is not overwritten
        assert_almost_equal(inv, inv_saved, decimal=18)  # make sure inv is not overwritten in backward step

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
