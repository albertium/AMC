
import abc
from typing import Tuple, Dict, List, Generator
import numpy as np
from sklearn import preprocessing

from .pde import PDE


class TimeSlice(abc.ABC):
    def __init__(self, t: float, states: Dict[str, np.ndarray]):
        self.t = t
        self.states = states

    @property
    def time(self):
        return self.t

    def state(self, name):
        return self.states[name]


class MCSlice(TimeSlice):
    def __init__(self, t: float, states: Dict[str, np.ndarray]):
        super(MCSlice, self).__init__(t=t, states=states)
        self.t = t
        self.states = states

    @property
    def numeraire(self):
        return self.states['numeraire']


class GridSlice(TimeSlice):
    # TODO: converge with time slice?
    def __init__(self, t: float, dt: float, values: np.ndarray, dims: list, states: Dict[str, np.ndarray]):
        super(GridSlice, self).__init__(t=t, states=states)
        self.dt = dt
        self.values = values
        self.dims = dims

    def value(self, name: str, idx: int):
        slicer = tuple(slice(idx, idx + 1 if idx >= 0 else None) if x == name else None for x in self.dims)
        return self.values[slicer]


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

    def simulate_states(self, tenor: float, num_steps: int, num_paths: int) -> List[MCSlice]:
        timestamps, states = self._simulate_states(tenor, num_steps, num_paths)

        sim = []
        for idx in range(len(timestamps)):
            sim.append(MCSlice(timestamps[idx], {k: v[idx] if v.ndim > 1 else v for k, v in states.items()}))
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
            -> Generator[Tuple[int, GridSlice, GridSlice], None, None]:
        pass


class FiniteDifferenceGrid(Grid):

    def __init__(self, tenor: float, pde: PDE):
        super(FiniteDifferenceGrid, self).__init__(tenor=tenor, pde=pde)

    def run(self, steps: Dict[str, int], scale: float = 5) \
            -> Generator[Tuple[int, GridSlice, GridSlice], None, None]:

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

        dims = self.factor_names[1:]
        prev_t = ts[0]
        prev_slice = GridSlice(t=prev_t, dt=0, values=self.values[-1], dims=dims, states=xs)
        for t_idx, t in enumerate(ts):
            curr_slice = GridSlice(t=t, dt=prev_t - t, values=self.values[t_idx], dims=dims, states=xs)
            yield t_idx, curr_slice, prev_slice
            prev_t = t
            prev_slice = curr_slice
