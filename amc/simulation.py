
import abc
from typing import Tuple, Dict, List, Generator
import numpy as np
from sklearn import preprocessing

from .pde import PDE


class TimeSlice:
    def __init__(self, timestamp: float, states: Dict[str, np.ndarray]):
        self.timestamp = timestamp
        self.states = states

    @property
    def time(self):
        return self.timestamp

    @property
    def numeraire(self):
        return self.states['numeraire']

    def state(self, name):
        return self.states[name]


class RNGenerator:
    def __init__(self, moment_matching: bool = True, antithetic: bool = True):
        self.moment_matching = moment_matching
        self.antithetic = antithetic

    def generate(self, num_steps: int, num_paths: int):
        if self.antithetic:
            if num_paths % 2 != 0:
                raise ValueError('Number of paths should be even when using antithetic')

            rands = np.random.normal(0, 1, (num_steps, int(num_paths / 2)))
            rands = np.hstack((rands, -rands))
        else:
            rands = np.random.normal(0, 1, (num_steps, num_paths))

        if self.moment_matching:
            preprocessing.scale(rands, axis=1, copy=False)

        return rands


class Simulator(abc.ABC):
    @abc.abstractmethod
    def _simulate_states(self, tenor: float, num_steps: int, num_paths: int) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        pass

    def simulate_states(self, tenor: float, num_steps: int, num_paths: int) -> List[TimeSlice]:
        timestamps, states = self._simulate_states(tenor, num_steps, num_paths)

        sim = []
        for idx in range(len(timestamps)):
            sim.append(TimeSlice(timestamps[idx], {k: v[idx] if v.ndim > 1 else v for k, v in states.items()}))
        sim.reverse()

        return sim


class BlackScholes(Simulator):
    def __init__(self, spot: float, interest: float, dividend: float, volatility: float):
        self.spot = spot
        self.interest = interest
        self.dividend = dividend
        self.volatility = volatility

    def _simulate_states(self, tenor: float, num_steps: int, num_paths: int) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        dt = tenor / num_steps
        rands = RNGenerator(antithetic=False, moment_matching=True).generate(num_steps, num_paths)
        rands = (self.interest - self.dividend - 0.5 * self.volatility ** 2) * dt + self.volatility * np.sqrt(dt) * rands
        underlying = self.spot * np.exp(np.cumsum(rands, axis=0))
        numeraire = np.tile(np.exp(self.interest * dt), (num_steps, num_paths))
        return np.linspace(dt, tenor, num_steps), {'numeraire': numeraire, 'stock': underlying}


class Grid(abc.ABC):
    # TODO: to replace simulator
    def __init__(self, tenor: float, pde: PDE):
        self.tenor = tenor
        self.pde = pde
        self.factors = pde.factors
        self.factor_names = ['t'] + [factor.name for factor in self.factors]
        self.values = None

    @abc.abstractmethod
    def run(self, steps: Dict[str, int], scale: float = None) \
            -> Generator[Tuple[int, float, np.ndarray, np.ndarray, TimeSlice], None, None]:
        pass


class FiniteDifferenceGrid(Grid):

    def __init__(self, tenor: float, pde: PDE):
        super(FiniteDifferenceGrid, self).__init__(tenor=tenor, pde=pde)

    def run(self, steps: Dict[str, int], scale: float = 5) \
            -> Generator[Tuple[int, float, np.ndarray, np.ndarray, TimeSlice], None, None]:

        if len(set(self.factor_names) - steps.keys()) > 0:
            raise ValueError('Not all steps are specified')

        # +1 for time since 1 step means payoff plus one step
        self.values = np.zeros([steps[name] + 1 if name == 't' else steps[name] for name in self.factor_names])
        ts = np.linspace(self.tenor, 0, steps['t'] + 1)
        xs = {}
        for factor in self.factors:
            if factor.is_normal:
                deviation = scale * factor.atm_vol
                lb = factor.spot - deviation
                ub = factor.spot + deviation
            else:  # exponential
                deviation = np.exp(scale * factor.atm_vol)
                lb = factor.spot / deviation
                ub = factor.spot * deviation

            xs[factor.name] = np.linspace(lb, ub, steps[factor.name])

        prev_t = ts[0]
        for t_idx, t in enumerate(ts):
            yield t_idx, prev_t - t, self.values[t_idx - 1], self.values[t_idx], TimeSlice(timestamp=t, states=xs)
            prev_t = t
