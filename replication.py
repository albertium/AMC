
import numpy as np
import scipy.linalg as linalg
from amc.helper import get_european_call_bs


class OptionsPricing(object):
    def __init__(self, S0, K, r, T, sigma, is_call=True):
        self.S0 = S0
        self.K = K
        self.r = r
        self.T = T
        self.sigma = sigma
        self.is_call = is_call


class FiniteDifferences(OptionsPricing):
    """ Shared attributes and functions of FD """

    def __init__(self, S0, K, r, T, sigma, Smax, M, N, is_call=True):
        super(FiniteDifferences, self).__init__(S0, K, r, T, sigma, is_call)
        self.Smax = Smax
        self.M, self.N = int(M), int(N)  # Ensure M&N are integers

        self.dS = Smax / float(self.M)
        self.dt = T / float(self.N)
        self.iValues = np.arange(1, self.M)
        self.jValues = np.arange(self.N)
        self.grid = np.zeros(shape=(self.M+1, self.N+1)) # grid is M+1 by N+1
        self.SValues = np.linspace(0, Smax, self.M+1)

    def _setup_boundary_conditions_(self):
        pass

    def _setup_coefficients_(self):
        pass

    def _traverse_grid_(self):
        """  Iterate the grid backwards in time """
        pass

    def _interpolate_(self):
        """
        Use piecewise linear interpolation on the initial
        grid column to get the closest price at S0.
        """
        return np.interp(self.S0,
                         self.SValues,
                         self.grid[:, 0])

    def price(self):
        self._setup_coefficients_()
        self._setup_boundary_conditions_()
        self._traverse_grid_()
        return self._interpolate_()


class ExplicitEu(FiniteDifferences):

    def _setup_coefficients_(self):
        self.alpha = 0.5 * self.dt * (self.sigma ** 2 * self.iValues ** 2 - self.r * self.iValues)
        self.beta = - self.dt * (self.sigma ** 2 * self.iValues ** 2 + self.r)
        self.gamma = 0.5 * self.dt * (self.sigma ** 2 * self.iValues ** 2 + self.r * self.iValues)
        self.coeffs = np.diag(self.alpha[1:], -1) + \
                      np.diag(1 + self.beta) + \
                      np.diag(self.gamma[:-1], 1)

    def _setup_boundary_conditions_(self):
        # terminal condition
        if self.is_call:
            self.grid[:, -1] = np.maximum(self.SValues - self.K, 0)
        else:
            self.grid[:, -1] = np.maximum(self.K - self.SValues, 0)

        # side boundary conditions
        self.coeffs[0, 0] += 2 * self.alpha[0]
        self.coeffs[0, 1] -= self.alpha[0]
        self.coeffs[-1, -1] += 2 * self.gamma[-1]
        self.coeffs[-1, -2] -= self.gamma[-1]

    def _traverse_grid_(self):
        for j in reversed(self.jValues):
            self.grid[1:-1, j] = np.dot(self.coeffs, self.grid[1:-1, j + 1])
            self.grid[0, j] = 2 * self.grid[1, j] - self.grid[2, j]
            self.grid[-1, j] = 2 * self.grid[-2, j] - self.grid[-3, j]


class CNEu(ExplicitEu):

    def _setup_coefficients_(self):
        self.alpha = 0.25 * self.dt * (self.sigma ** 2 * self.iValues ** 2 - self.r * self.iValues)
        self.beta = -0.5 * self.dt * (self.sigma ** 2 * self.iValues ** 2 + self.r)
        self.gamma = 0.25 * self.dt * (self.sigma ** 2 * self.iValues ** 2 + self.r * self.iValues)
        self.coeffs = np.diag(self.alpha[1:], -1) + \
                      np.diag(1 + self.beta) + \
                      np.diag(self.gamma[:-1], 1)
        self.coeffs_ = np.diag(-self.alpha[1:], -1) + \
                       np.diag(1 - self.beta) + \
                       np.diag(-self.gamma[:-1], 1)

    def _setup_boundary_conditions_(self):
        super(CNEu, self)._setup_boundary_conditions_()
        self.coeffs_[0, 0] -= 2 * self.alpha[0]
        self.coeffs_[0, 1] += self.alpha[0]
        self.coeffs_[-1, -1] -= 2 * self.gamma[-1]
        self.coeffs_[-1, -2] += self.gamma[-1]

    def _traverse_grid_(self):
        P, L, U = linalg.lu(self.coeffs_)
        for j in reversed(self.jValues):
            Ux = linalg.solve(L, np.dot(self.coeffs, self.grid[1:-1, j + 1]))
            self.grid[1:-1, j] = linalg.solve(U, Ux)
            self.grid[0, j] = 2 * self.grid[1, j] - self.grid[2, j]
            self.grid[-1, j] = 2 * self.grid[-2, j] - self.grid[-3, j]

S0 = 248
K = 300
r = 254 / S0 - 1
T = 0.25
sigma = 0.72
Smax = 1000
M = 100  # S
N = 1000 # t
is_call = True

option = CNEu(S0, K, r, T, sigma, Smax, M, N, is_call)
print(option.price())
print(get_european_call_bs(S0, K, r, 0, sigma, T))
