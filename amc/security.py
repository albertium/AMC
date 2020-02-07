
import abc
from typing import List, Union
import numpy as np

from .simulation import TimeSlice


class Payoff(abc.ABC):
    def __init__(self, factors: List[str]):
        self.factors = factors

    @abc.abstractmethod
    def __call__(self, time_slice: TimeSlice) -> np.ndarray:
        pass


class CallPayoff(Payoff):
    def __init__(self, asset: str, strike: float):
        super(CallPayoff, self).__init__([asset])
        self.asset = asset
        self.strike = strike

    def __call__(self, time_slice: TimeSlice) -> np.ndarray:
        return np.maximum(time_slice.state(self.asset) - self.strike, 0)


class PutPayoff(Payoff):
    def __init__(self, asset: str, strike: float):
        super(PutPayoff, self).__init__([asset])
        self.asset = asset
        self.strike = strike

    def __call__(self, time_slice: TimeSlice) -> np.ndarray:
        return np.maximum(self.strike - time_slice.state(self.asset), 0)


class Security(abc.ABC):
    def __init__(self, tenor: float, factors: List[str] = None, need_continuation: bool = False):
        self.tenor = tenor
        self.need_continuation = need_continuation
        self.factors = factors

    @abc.abstractmethod
    def backprop(self, time_slice: TimeSlice, last_values: np.ndarray, continuation: np.ndarray) -> Union[None, np.ndarray]:
        pass


class EuropeanOption(Security):
    def __init__(self, payoff: Payoff, tenor: float):
        super(EuropeanOption, self).__init__(tenor)
        self.payoff = payoff

    def backprop(self, time_slice: TimeSlice, last_values: np.ndarray, continuation: np.ndarray) -> Union[None, np.ndarray]:
        if time_slice.time == self.tenor:
            return self.payoff(time_slice)
        return last_values


class AmericanOption(Security):
    def __init__(self, payoff: Payoff, tenor: float):
        super(AmericanOption, self).__init__(tenor, factors=payoff.factors, need_continuation=True)
        self.payoff = payoff

    def backprop(self, time_slice: TimeSlice, last_values: np.ndarray, continuation: np.ndarray) -> Union[None, np.ndarray]:
        intrinsic_values = self.payoff(time_slice)
        if time_slice.time == self.tenor:
            return intrinsic_values
        return np.where(continuation > intrinsic_values, last_values, intrinsic_values)
