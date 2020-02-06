
import abc
from typing import Tuple, Dict, List
import numpy as np
from sklearn import preprocessing


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
        timestamp, states = self._simulate_states(tenor, num_steps, num_paths)

        sim = []
        for idx in range(len(timestamp)):
            sim.append(TimeSlice(timestamp[idx], {k: v[idx] for k, v in states.items()}))
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