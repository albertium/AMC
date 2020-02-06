
from typing import List, Union
import numpy as np

from .security import Security
from .simulation import Simulator
from .fitter import Fitter


class Pricer:
    def __init__(self, securities: List[Security], model: Simulator, fitter: Union[None, Fitter]):
        self.securities = securities
        self.tenor = max(security.tenor for security in securities)
        self.model = model
        self.fitter = fitter

    def price(self, num_steps: int, num_paths: int):
        slices = self.model.simulate_states(self.tenor, num_steps, num_paths)  # in reversed order
        values = np.zeros((num_steps, len(self.securities), num_paths))

        # price securities on each time step
        for t_idx, time_slice in enumerate(slices):
            for s_idx, security in enumerate(self.securities):
                # estimate continuation value
                if t_idx == 0 or not security.need_continuation:
                    continuation = None
                else:
                    # last values here is already deflated
                    continuation = self.fitter.fit_predict(values[t_idx - 1, s_idx], time_slice, security.factors)

                raw = security.backprop(time_slice, values[t_idx - 1, s_idx], continuation)
                values[t_idx, s_idx] = raw / time_slice.numeraire  # deflate current value

        return np.mean(values[-1], axis=1)
