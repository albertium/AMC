
import abc
from typing import List, Union
import numpy as np

from .simulation import TimeSlice


class Security(abc.ABC):
    def __init__(self, tenor: float, factors: List[str] = None, need_continuation: bool = False):
        self.tenor = tenor
        self.need_continuation = need_continuation
        self.factors = factors

    @abc.abstractmethod
    def backprop(self, time_slice: TimeSlice, continuation: np.ndarray) -> Union[None, np.ndarray]:
        pass


class EuropeanOption(Security):
    def __init__(self, asset: str, strike: float, tenor: float):
        super(EuropeanOption, self).__init__(tenor)
        self.asset = asset
        self.strike = strike

    def backprop(self, time_slice: TimeSlice, continuation: np.ndarray) -> Union[None, np.ndarray]:
        if time_slice.time == self.tenor:
            return np.maximum(time_slice.state(self.asset) - self.strike, 0)
        return None

