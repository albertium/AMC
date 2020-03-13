
import abc
from typing import List, Union, Dict
import numpy as np
from scipy.linalg import solve_banded

from .security import Security, FiniteDifferenceMixin
from .simulation import Simulator, Grid, FiniteDifferenceGrid
from .fitter import Fitter
from .pde import PDE


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


class FiniteDifferenceScheme(abc.ABC):
    @abc.abstractmethod
    def step(self, v_new: np.ndarray, v: np.ndarray, lx: np.ndarray) -> None:
        """
        change v_new inplace
        """
        pass

    @staticmethod
    def forward_step(v_new: np.ndarray, v: np.ndarray, diff_op) -> None:
        """
        explicit step, solve v_new = diff_op * v
        diff_op[0] is the upper diagonal
        diff_op[2] is the lower diagonal

        :param v_new: v(t+1) of size n + 2
        :param v: v(t) of size n + 2
        :param diff_op: differential operator
        """
        v_new[1: -1] = diff_op[0] * v[: -2] + diff_op[1] * v[1: -1] + diff_op[2] * v[2:]

    @staticmethod
    def backward_step(v_new: np.ndarray, v: np.ndarray, diff_op) -> None:
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
        v = v[1: -1].copy()
        v[0] -= diff_op[0, 0] * v_new[0]
        v[-1] -= diff_op[2, -1] * v_new[-1]

        # mold diff_op into the format required by solve_banded
        adjusted_op = np.zeros_like(diff_op)
        adjusted_op[2, :-1] = diff_op[0, 1:]
        adjusted_op[1] = diff_op[1]
        adjusted_op[0, 1:] = diff_op[2, :-1]

        # free to alter adjusted_op for performance
        v_new[1: -1] = solve_banded((1, 1), adjusted_op, v, overwrite_ab=True)


class ExplicitScheme(FiniteDifferenceScheme):
    def step(self, v_new: np.ndarray, v: np.ndarray, lx: np.ndarray) -> None:
        # TODO: need to copy lx, may not be efficient
        lx = lx.copy()
        lx[1] += 1
        self.forward_step(v_new, v, lx)


class ImplicitScheme(FiniteDifferenceScheme):
    def step(self, v_new: np.ndarray, v: np.ndarray, lx: np.ndarray) -> None:
        lx = -lx
        lx[1] += 1
        self.backward_step(v_new, v, lx)


class CrankNicolsonScheme(FiniteDifferenceScheme):
    def step(self, v_new: np.ndarray, v: np.ndarray, lx: np.ndarray) -> None:
        # TODO: need to copy lx, may not be efficient
        # forward step
        lx = 0.5 * lx  # half step
        lx[1] += 1
        self.forward_step(v_new, v, lx)

        # backward step
        lx = -lx
        lx[1] += 2  # just to make 1 - 0.5 * original lx
        self.backward_step(v_new, v_new, lx)  # backward step only need v_new[1: -1]. Boundaries doesn't affect


class FiniteDifferenceEngine(PricingEngine):
    """
    only price 1D PDE for now
    """
    def __init__(self, security: Security, pde: PDE, scheme: FiniteDifferenceScheme):
        if not isinstance(security, FiniteDifferenceMixin):
            raise RuntimeError('Security doesnt support finite difference')

        # TODO: Mixin doesn't seem to work here
        super(FiniteDifferenceEngine, self).__init__([security], FiniteDifferenceGrid(security.tenor, pde))
        self.pde = pde
        self.scheme = scheme

    def price(self, steps: Dict[str, int], scale: float = 5):
        for idx, dt, prev, curr, time_slice in self.model.run(steps, scale=scale):
            if idx > 0:
                lx = dt * self.pde.differential_operator(time_slice.states)
                self.securities[0].update_boundary(curr, time_slice)
                self.scheme.step(curr, prev, lx)
            curr[:] = self.securities[0].backprop(time_slice, curr, curr)  # in FD, continuation is curr

        return self.model.values[-1]
