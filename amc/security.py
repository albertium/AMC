
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


class MaskGenerator(abc.ABC):
    @abc.abstractmethod
    def __call__(self, time_slice: TimeSlice):
        pass


class CallMask(MaskGenerator):
    def __init__(self, asset: str, strike: float):
        self.asset = asset
        self.strike = strike

    def __call__(self, time_slice: TimeSlice):
        return time_slice.state(self.asset) > self.strike


class PutMask(MaskGenerator):
    def __init__(self, asset: str, strike: float):
        self.asset = asset
        self.strike = strike

    def __call__(self, time_slice: TimeSlice):
        return time_slice.state(self.asset) < self.strike


class Security(abc.ABC):
    def __init__(self, tenor: float, factors: List[str] = None, need_continuation: bool = False,
                 mask: MaskGenerator = None):
        self.tenor = tenor
        self.need_continuation = need_continuation
        self.mask = mask
        self.factors = factors

    @abc.abstractmethod
    def backprop(self, time_slice: TimeSlice, last_values: np.ndarray, continuation: np.ndarray) -> np.ndarray:
        pass


class EuropeanOptionBase(Security):
    def __init__(self, payoff: Payoff, tenor: float):
        super(EuropeanOptionBase, self).__init__(tenor)
        self.payoff = payoff

    def backprop(self, time_slice: TimeSlice, last_values: np.ndarray, continuation: np.ndarray) -> Union[None, np.ndarray]:
        if time_slice.time == self.tenor:
            return self.payoff(time_slice)
        return last_values


class EuropeanCall(EuropeanOptionBase):
    def __init__(self, asset: str, strike: float, tenor: float):
        payoff = CallPayoff(asset, strike)
        super(EuropeanCall, self).__init__(payoff, tenor)


class EuropeanPut(EuropeanOptionBase):
    def __init__(self, asset: str, strike: float, tenor: float):
        payoff = PutPayoff(asset, strike)
        super(EuropeanPut, self).__init__(payoff, tenor)


class AmericanOptionBase(Security):
    def __init__(self, tenor: float, payoff: Payoff, mask: MaskGenerator):
        super(AmericanOptionBase, self).__init__(tenor, factors=payoff.factors, need_continuation=True)
        self.payoff = payoff
        self.mask = mask

    def backprop(self, time_slice: TimeSlice, last_values: np.ndarray, continuation: np.ndarray) -> np.ndarray:
        intrinsic_values = self.payoff(time_slice)
        if time_slice.time == self.tenor:
            return intrinsic_values
        return np.where(continuation > intrinsic_values, last_values, intrinsic_values)


class AmericanCall(AmericanOptionBase):
    def __init__(self, asset: str, strike: float, tenor: float):
        payoff = CallPayoff(asset=asset, strike=strike)
        mask = CallMask(asset=asset, strike=strike)
        super(AmericanCall, self).__init__(tenor, payoff, mask)


class AmericanPut(AmericanOptionBase):
    def __init__(self, asset: str, strike: float, tenor: float):
        payoff = PutPayoff(asset=asset, strike=strike)
        mask = PutMask(asset=asset, strike=strike)
        super(AmericanPut, self).__init__(tenor, payoff, mask)