
import abc
from typing import List, Union, Dict, Tuple
import numpy as np
from scipy.linalg import solve_banded
from numba import jit, njit

from amc.security import Security
from amc.grid import Simulator, Grid, FiniteDifferenceGrid, GridSlice
from amc.fitter import Fitter
from amc.pde import PDE1D, PDE2D, PDE
from amc.const import Op
from amc.helper import solve_tridiagonal


class PricingEngine(abc.ABC):
    def __init__(self, securities: List[Security], model: Grid):
        self.securities = securities
        self.tenor = max(security.tenor for security in securities)
        self.model = model

    @abc.abstractmethod
    def price(self, num_steps: int, num_paths: int):
        """
        :param num_steps: number of time steps
        :param num_paths: number of steps in states
        :return:
        """
        pass


# ================================= Monte Carlo =================================
class MonteCarloEngine(PricingEngine):
    def __init__(self, securities: List[Security], model: Simulator, fitter: Union[None, Fitter] = None):
        super(MonteCarloEngine, self).__init__(securities, model)
        self.fitter = fitter

    def price(self, num_steps: int, num_paths: int):
        slices = self.model.simulate_states(self.tenor, num_steps, num_paths)  # in reversed order
        values = np.zeros((num_steps, len(self.securities), num_paths))

        # price securities on each time step
        for t_idx, time_slice in enumerate(slices):
            for s_idx, security in enumerate(self.securities):
                prev = values[t_idx - 1, s_idx]  # the 1st step will reference values[-1], which is all 0's
                # estimate continuation value
                if t_idx == 0 or not security.need_continuation:
                    continuation = None
                else:
                    mask = security.mask(time_slice) if security.mask is not None else None
                    continuation = self.fitter.fit_predict(prev, time_slice, security.factors, mask)

                raw = security.backprop(time_slice, prev, continuation)
                values[t_idx, s_idx] = raw / time_slice.numeraire  # deflate current value

        return np.mean(values[-1], axis=1)


# ================================= Finite Difference =================================
class FiniteDifferenceScheme(abc.ABC):
    def __init__(self):
        self.cp = None  # for Thomas algorithm

    @abc.abstractmethod
    def step(self, curr: GridSlice, prev: GridSlice, pde: PDE, sec: Security) -> None:
        """
        change v_new inplace
        """
        pass

    @staticmethod
    def forward_step(v_new: np.ndarray, v: np.ndarray, diff_op, op_type=Op.ADD) -> None:
        """
        explicit step, solve v_new = diff_op * v
        diff_op[0] is the upper diagonal
        diff_op[2] is the lower diagonal

        :param v_new: v(t+1) of size n + 2
        :param v: v(t) of size n + 2
        :param diff_op: differential operator
        :param op_type: add or subtract
        """
        # use += because in 2D case, we need to call forward_step multiple times and we need to accumulate the results
        tmp = diff_op[0] * v[: -2] + diff_op[1] * v[1: -1] + diff_op[2] * v[2:]
        if op_type == Op.ADD:
            v_new[1: -1] += tmp
        else:
            v_new[1: -1] -= tmp

    def backward_step(self, v_new: np.ndarray, v: np.ndarray, diff_op: np.ndarray) -> None:
        """
        implicit step, solve diff_op * v_new = v
        diff_op[0] is the upper diagonal
        diff_op[2] is the lower diagonal

        :param v_new: v(t+1) of size n + 2
        :param v: v(t) of size n + 2
        :param b0: left boundary
        :param b1: right boundary
        :param diff_op: differential operator
        """
        # The adjustments (last 2 arguments) are to take care Dirichlet boundary
        # The full specification is a * v[0] + b * v[1] + c * v[2] = u[1]
        # When there is Dirichlet boundary, v[0] is known (none zero), u[1] is given.
        # Thus, we have b * v[1] + c * v[2] = u[1] - a * v[0]. Same for another boundary
        if self.cp is None or len(self.cp) < len(v_new):
            self.cp = np.zeros(len(v_new))
        solve_tridiagonal(self.cp, v_new[1: -1], v[1: -1], diff_op, diff_op[0, 0] * v_new[0], diff_op[2, -1] * v_new[-1])

    @staticmethod
    def nd_forward_step(v_new: np.ndarray, v: np.ndarray, diff_op, op_type=Op.ADD, shift: int = 0) -> None:
        """
        2d version of forward step
        """
        for idx in range(v_new.shape[0] - 2):
            FiniteDifferenceScheme.forward_step(v_new[idx + 1], v[idx + 1 + shift], diff_op[:, idx], op_type=op_type)

    def nd_backward_step(self, v_new: np.ndarray, v: np.ndarray, diff_op) -> None:
        """
        2d version of backward step
        """
        for idx in range(v_new.shape[0] - 2):
            self.backward_step(v_new[idx + 1], v[idx + 1], diff_op[:, idx])

    @staticmethod
    def forward_step_cross(v_new: np.ndarray, v: np.ndarray, diff_op: np.ndarray) -> None:
        for idx in range(3):
            FiniteDifferenceScheme.nd_forward_step(v_new, v, diff_op[idx], shift=idx - 1)


class ExplicitScheme(FiniteDifferenceScheme):
    def step(self, curr: GridSlice, prev: GridSlice, pde: PDE1D, sec: Security) -> None:
        # TODO: need to copy lx, may not be efficient
        lx = curr.dt * pde.differential_operator(curr.states)
        lx[1] += 1
        sec.update_differential_operator(pde.x_name, lx, curr)
        self.forward_step(curr.values, prev.values, lx)


class ImplicitScheme(FiniteDifferenceScheme):
    def step(self, curr: GridSlice, prev: GridSlice, pde: PDE1D, sec: Security) -> None:
        lx = -curr.dt * pde.differential_operator(curr.states)
        lx[1] += 1
        sec.update_differential_operator(pde.x_name, lx, curr)
        self.backward_step(curr.values, prev.values, lx)


class CrankNicolsonScheme(FiniteDifferenceScheme):
    def step(self, curr: GridSlice, prev: GridSlice, pde: PDE1D, sec: Security) -> None:
        # TODO: need to copy lx, may not be efficient
        lx_base = (0.5 * curr.dt) * pde.differential_operator(curr.states)

        # forward step
        lx = lx_base.copy()
        lx[1] += 1
        sec.update_differential_operator(pde.x_name, lx, curr)
        self.forward_step(curr.values, prev.values, lx)

        # backward step
        lx = -lx_base
        lx[1] += 1  # just to make 1 - 0.5 * original lx
        sec.update_differential_operator(pde.x_name, lx, curr)
        self.backward_step(curr.values, curr.values, lx)  # backward step only need v_new[1: -1]. Boundaries doesn't affect


class DouglasScheme(FiniteDifferenceScheme):
    """
    Douglas Scheme
    V0 = (1 + dt(Lx + Ly + Lxy)) * U
    (1 - 0.5 * dt * Lx) * V1 = V0 - 0.5 * dt * Lx * U [=(1 + dt(0.5 * Lx + Ly + Lxy)) * U]
    (1 - 0.5 * dt * Ly) * V2 = V1 - 0.5 * dt * Ly * U
    V = V2

    see "The Heston Model and its Extensions in Matlab and C#" by Rouah and Heston pg. 322
    """

    def step(self, curr: GridSlice, prev: GridSlice, pde: PDE2D, sec: Security) -> None:
        # !!! don't change lx_base, ly_base and lxy since they may be re-used in non-dynamic pde !!!
        lx_base, ly_base, lxy = pde.differential_operator(curr.states)  # type: np.ndarray
        # Lx times 0.5 because we need to subtract it back later anyway. See above description
        lx_base, ly_base, lxy = 0.5 * curr.dt * lx_base, curr.dt * ly_base, curr.dt * lxy

        # ----- explicit step on Lx, Ly and Lxy -----
        # TODO: Lxy to be added

        # This operates on diff_op[0 - 2, y_dix, :], which gives you the whole x.
        # However, we want whole y for a given x. That's why we need transpose
        lx, ly = lx_base.copy(), ly_base.copy()
        sec.update_differential_operator(pde.x_name, lx.transpose((0, 2, 1)), curr)
        sec.update_differential_operator(pde.y_name, ly.transpose((0, 2, 1)), curr)

        # TODO: apply Lx, Ly and Lxy sequentially is not efficient enough. Can we do L * prev directly?
        self.forward_step_cross(curr.values.T, prev.values.T, lxy)
        self.nd_forward_step(curr.values.T, prev.values.T, lx)

        # use transpose because y is primary now
        # use prev instead of curr as 2nd argument because we are doing curr = (Lx + Ly) * prev
        # If we put curr as 2nd argument, then we will do curr = Lx * Ly * prev
        self.nd_forward_step(curr.values, prev.values, ly)
        curr.values[1: -1, 1: -1] += prev.values[1: -1, 1: -1]

        # ----- implicit step on Lx, Ly -----
        ly_base = 0.5 * ly_base
        lx, ly = -lx_base, -ly_base
        lx[1] += 1
        ly[1] += 1

        sec.update_differential_operator(pde.x_name, lx.transpose((0, 2, 1)), curr)
        # This a new diff_op (times 0.5), need to update again
        sec.update_differential_operator(pde.y_name, ly_base.transpose((0, 2, 1)), curr)
        sec.update_differential_operator(pde.y_name, ly.transpose((0, 2, 1)), curr)

        # Lx is already halved, no need to subtract
        self.nd_backward_step(curr.values.T, curr.values.T, lx)
        self.nd_forward_step(curr.values, prev.values, ly_base, op_type=Op.MINUS)  # subtract half Ly back
        self.nd_backward_step(curr.values, curr.values, ly)


class FiniteDifferenceEngine(PricingEngine):
    """
    only price 1D PDE for now
    """
    def __init__(self, security: Security, pde: PDE, scheme: FiniteDifferenceScheme):
        super(FiniteDifferenceEngine, self).__init__([security], FiniteDifferenceGrid(security.tenor, pde))
        self.pde = pde
        self.scheme = scheme

    def price(self, steps: Dict[str, int], scale: float = 3) -> Tuple[np.ndarray, np.ndarray]:
        sec = self.securities[0]
        for idx, curr, prev in self.model.run(steps, scale=scale):
            if idx > 0:
                sec.update_values(curr)
                self.scheme.step(curr, prev, self.pde, sec)
                sec.post_process_values(curr)
            curr.values[:] = sec.backprop(curr, curr.values, curr.values)  # in FD, continuation is curr

        return self.model.values[-1], curr.states
