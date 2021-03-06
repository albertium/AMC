
import abc
from typing import List, Union, Tuple, Dict, Callable
import numpy as np

from amc.grid import MCSlice, GridSlice, TimeSlice
from amc.data import EquityFactor
from amc.const import Bound


# ================================= Payoff =================================
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


# ================================= Mask =================================
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


# ================================= Boundary =================================
class Boundary:
    def __init__(self, dim: str, bound: Bound):
        self.dim = dim  # asset or factor name
        self.bound = bound  # upper or lower

    def update_values(self, curr_slice: GridSlice) -> None:
        pass

    def update_differential_operator(self, dim:str, diff_op: np.ndarray, curr_slice: GridSlice) -> None:
        pass

    def post_process_values(self, curr_slice: GridSlice) -> None:
        pass


class DirichletBoundary(Boundary):
    """
    Simply set v[0] = f(x[0], t) and v[-1] = f(x[-1], t)
    """

    def __init__(self, dim: str, bound: Bound, func: Callable[[GridSlice], Union[float, np.ndarray]]):
        super(DirichletBoundary, self).__init__(dim=dim, bound=bound)
        self.func = func
        if self.bound == Bound.UPPER:
            self.idx = -1
        else:
            self.idx = 0

    def update_values(self, curr_slice: GridSlice):
        curr_slice.value(self.dim, self.idx)[:] = self.func(curr_slice)


class LinearBoundary(Boundary):
    """
    For general grid, the second derivative is h_i * v[2] - (h_i + h_p) * v[1] + h_p * v[0]
    Linear means setting this quantity to 0, which gives v[0] = (h_i + h_p) / h_p * v[1] - h_i / h_p * v[2]
    See Clark 7.54
    """

    def update_differential_operator(self, dim:str, diff_op: np.ndarray, curr_slice: GridSlice) -> None:
        # TODO: linear boundary doesn't seem correct now. Need to test
        if dim != self.dim:
            return

        x = curr_slice.state(self.dim)

        if self.bound == Bound.UPPER:
            xi = x[-2] - x[-3]
            xp = x[-1] - x[-2]
            diff_op[0, -1] += diff_op[2, -1] * (xi + xp) / xi
            diff_op[1, -1] -= diff_op[2, -1] * xp / xi

        else:
            xi = x[1] - x[0]
            xp = x[2] - x[1]
            diff_op[1, 0] += diff_op[0, 0] * (xi + xp) / xp
            diff_op[2, 0] -= diff_op[0, 0] * xi / xp

    def post_process_values(self, curr_slice: GridSlice) -> None:
        x = curr_slice.state(self.dim)

        # TODO: assume 1D for now
        if self.bound == Bound.UPPER:
            xi = x[-2] - x[-3]
            xp = x[-1] - x[-2]

            curr_slice.value(self.dim, -1)[:] = curr_slice.value(self.dim, -2) * (xi + xp) / xi - \
                                                curr_slice.value(self.dim, -3) * xp / xi

        else:
            xi = x[1] - x[0]
            xp = x[2] - x[1]

            curr_slice.value(self.dim, 0)[:] = curr_slice.value(self.dim, 1) * (xi + xp) / xp - \
                                               curr_slice.value(self.dim, 2) * xi / xp


# ================================= Security =================================
class Security(abc.ABC):
    def __init__(self, tenor: float, factors: List[str] = None, need_continuation: bool = False,
                 mask: MaskGenerator = None, boundaries: List[Boundary] = None):
        self.tenor = tenor
        self.need_continuation = need_continuation
        self.mask = mask
        self.factors = factors
        self.boundaries = boundaries if boundaries is not None else []

    @abc.abstractmethod
    def backprop(self, time_slice: TimeSlice, last_values: np.ndarray, continuation: np.ndarray) -> np.ndarray:
        pass

    def update_values(self, curr_slice: GridSlice) -> None:
        for boundary in self.boundaries:
            boundary.update_values(curr_slice)

    def update_differential_operator(self, dim: str, diff_op: np.ndarray, curr_slice: GridSlice) -> None:
        for boundary in self.boundaries:
            boundary.update_differential_operator(dim, diff_op, curr_slice)

    def post_process_values(self, curr_slice: GridSlice) -> None:
        for boundary in self.boundaries:
            boundary.post_process_values(curr_slice)


class HeatSecurity(Security):

    def __init__(self, tenor: float):
        self.asset = 'Heat'
        super(HeatSecurity, self).__init__(tenor, [self.asset])
        self.tenor = tenor

        # create boundary
        # TODO: not sure will work for 2D
        lb = DirichletBoundary(self.asset, Bound.LOWER, lambda gl: np.exp(gl.state(self.asset)[0] + self.tenor - gl.t))
        ub = DirichletBoundary(self.asset, Bound.UPPER, lambda gl: np.exp(gl.state(self.asset)[-1] + self.tenor - gl.t))
        self.boundaries = [lb, ub]  # a little bit anti-pattern

    def backprop(self, time_slice: TimeSlice, last_values: np.ndarray, continuation: np.ndarray) -> np.ndarray:
        if time_slice.time == self.tenor:
            return np.exp(time_slice.state('Heat'))
        return last_values


# ================================= European =================================
class EuropeanOptionBase(Security):
    def __init__(self, asset: str, payoff: Payoff, tenor: float, boundaries: List[Boundary] = None):
        super(EuropeanOptionBase, self).__init__(tenor=tenor, factors=[asset], boundaries=boundaries)
        self.payoff = payoff

    def backprop(self, time_slice: TimeSlice, last_values: np.ndarray, continuation: np.ndarray) -> Union[None, np.ndarray]:
        if time_slice.time == self.tenor:
            return self.payoff(time_slice)
        return last_values


class EuropeanCall(EuropeanOptionBase):
    def __init__(self, asset: str, strike: float, tenor: float):
        self.asset = asset
        self.strike = strike
        payoff = CallPayoff(asset, self.strike)
        lb = DirichletBoundary(asset, Bound.LOWER, lambda gl: 0)
        ub = DirichletBoundary(asset, Bound.UPPER, lambda gl: gl.state(asset)[-1] - strike)
        super(EuropeanCall, self).__init__(asset=asset, payoff=payoff, tenor=tenor, boundaries=[lb, ub])


class EuropeanPut(EuropeanOptionBase):
    def __init__(self, asset: str, strike: float, tenor: float):
        payoff = PutPayoff(asset, strike)
        super(EuropeanPut, self).__init__(asset=asset, payoff=payoff, tenor=tenor)


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


# ================================= Exotic =================================
class ExchangeOption(Security):

    def __init__(self, equity1: EquityFactor, equity2: EquityFactor, tenor):
        self.equities = [equity1.name, equity2.name]
        x_lb = DirichletBoundary(equity1.name, Bound.LOWER,
                                 lambda gs: np.maximum(gs.state(equity1.name)[0] - gs.state(equity2.name), 0))
        x_ub = DirichletBoundary(equity1.name, Bound.UPPER,
                                 lambda gs: np.maximum(gs.state(equity1.name)[-1] - gs.state(equity2.name), 0))
        y_lb = DirichletBoundary(equity2.name, Bound.LOWER,
                                 lambda gs: np.maximum(gs.state(equity1.name) - gs.state(equity2.name)[0], 0))
        y_ub = DirichletBoundary(equity2.name, Bound.UPPER,
                                 lambda gs: np.maximum(gs.state(equity1.name) - gs.state(equity2.name)[-1], 0))
        super(ExchangeOption, self).__init__(tenor=tenor, factors=[equity1.name, equity2.name],
                                             boundaries=[x_lb, x_ub, y_lb, y_ub])

    def backprop(self, time_slice: TimeSlice, last_values: np.ndarray, continuation: np.ndarray) -> np.ndarray:
        if time_slice.time == self.tenor:
            grid = time_slice.state(self.equities[0]).reshape(-1, 1) - time_slice.state(self.equities[1])
            return np.maximum(grid, 0)
        return last_values
