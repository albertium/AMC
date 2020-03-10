
import abc
from typing import List, Union
import numpy as np

from .security import Security
from .simulation import Simulator, GridSimulator
from .fitter import Fitter
from .pde import PDE1D


class PricingEngine(abc.ABC):
    def __init__(self, securities: List[Security], model: Simulator):
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


class FiniteDifferenceEngine(PricingEngine):
    """
    only price 1D PDE for now
    """
    def __init__(self, security, pde: PDE1D):
        super(FiniteDifferenceEngine, self).__init__([security], GridSimulator(pde))
        self.pde = pde

    def price(self, num_steps: int, num_paths: int, std: float = 5):
        slices = self.model.simulate_states(num_steps, num_paths, std)
        values = np.zeros((num_steps + 1, num_paths + 2))  # +1 for final step, +2 for boundaries

        # price securities on each time step
        for t_idx, time_slice in enumerate(slices):
            prev = values[t_idx - 1]
            continuation = None  # finite difference is continuation already
            raw = self.securities[0].backprop(time_slice, prev, continuation)
            lx = self.pde.lx(time_slice.state('Stock'))
            values[t_idx, s_idx] = raw / time_slice.numeraire  # deflate current value

        return np.mean(values[-1], axis=1)