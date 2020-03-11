
import unittest
import numpy as np
from numpy.testing import assert_almost_equal
from scipy.sparse import diags

from amc.engine import FiniteDifferenceScheme


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

