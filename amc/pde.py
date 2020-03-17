
import abc
import numpy as np
from typing import Union, List, Dict, Tuple

from amc.data import Factor, EquityFactor


# ================================= Base Class =================================
class PDE(abc.ABC):
    def __init__(self, factors: List[Factor], is_dynamic_pde: bool = False):
        self.factors = factors
        self.is_dynamic_pde = is_dynamic_pde
        self.diff_ops = {}

    @abc.abstractmethod
    def differential_operator(self, states: Dict[str, np.ndarray]) -> np.ndarray:
        pass

    @staticmethod
    def generate_diff_operator(convection: np.ndarray, diffusion: np.ndarray, reaction: np.ndarray,
                               dx: np.ndarray, d2x: np.ndarray, sizes: List[int]) -> np.ndarray:
        """
        get differential operator matrix (Lx + Lxx). Support non-uniform mesh
        See Iain Clark formula (7.56)
        """
        dxi = dx[1:]
        dxp = dx[: -1]
        lx = np.zeros([3] + sizes)  # use dx shape and y_dim because convection can be a scalar

        # if convection / diffusion / reaction is fully specified, ie a 2D array,
        # then its dimension should be (y_dim, x_dim). This is different from the convention (x_dim, y_dim).
        # we do this because differential operator is used in transpose form (A.T * b) and this convention is
        # better to work with
        lx[0, :] = ((2 * diffusion - convection * dxp) / dxi / d2x)
        lx[1, :] = ((convection * (dxp - dxi) - 2 * diffusion) / dxi / dxp - reaction)
        lx[2, :] = ((2 * diffusion + convection * dxi) / dxp / d2x)
        return lx

    # def differential_cross_operator(self, dim1: str, dim2: str, xs: np.ndarray, ys: np.ndarray):
    #     """
    #     Differential operator for cross term, ie, Lxy
    #     This function is meant to cache result. For example, in the case of Black Scholes
    #     """
    #
    #     if self.is_dynamic_pde or (dim1, dim2) not in self.diff_ops:
    #         dx = xs[1:] - xs[: -1]
    #         d2x = xs[2:] - xs[: -2]
    #         interior = xs[1: -1]
    #         convection = getattr(self, f'convection_{dim}')(interior)
    #         diffusion = getattr(self, f'diffusion_{dim}')(interior)
    #         reaction = getattr(self, f'reaction_{dim}')(interior)
    #         self.diff_ops[(dim1, dim2)] = self.generate_diff_operator(convection, diffusion, reaction, dx, d2x)


class PDE1D(PDE):
    def __init__(self, factor: Factor, is_dynamic_pde: bool = False):
        super(PDE1D, self).__init__([factor], is_dynamic_pde=is_dynamic_pde)
        self.x_name = factor.name

    @abc.abstractmethod
    def convection(self, x: np.ndarray) -> np.ndarray:
        """
        u_x
        """
        pass

    @abc.abstractmethod
    def diffusion(self, x: np.ndarray) -> np.ndarray:
        """
        u_xx
        """
        pass

    @abc.abstractmethod
    def reaction(self, x: np.ndarray) -> Union[float, np.ndarray]:
        """
        u
        """
        pass

    def differential_operator(self, states: Dict[str, np.ndarray]) -> Tuple:
        """
        Lx - differential operator. Equals Lx + Lxx
        This function is meant to cache result. For example, in the case of Black Scholes
        """
        if self.is_dynamic_pde or not self.diff_ops:
            xs = states[self.x_name]
            dx = xs[1:] - xs[: -1]
            d2x = xs[2:] - xs[: -2]
            interior = xs[1: -1]
            convection = self.convection(interior)
            diffusion = self.diffusion(interior)
            reaction = self.reaction(interior)
            self.diff_ops[self.x_name] = self.generate_diff_operator(convection, diffusion, reaction, dx, d2x, [len(xs) - 2])
        return self.diff_ops[self.x_name]


class PDE2D(PDE):
    def __init__(self, factors: List[Factor], is_dynamic_pde: bool = False):
        super(PDE2D, self).__init__(factors=factors, is_dynamic_pde=is_dynamic_pde)
        self.x_name = factors[0].name
        self.y_name = factors[1].name

    @abc.abstractmethod
    def convection_x(self, xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
        """
        u_x.
        This should be row oriented, meaning the first dimension is y and the second dimension is x
        We do this because when we apply diff_op, we do this diff_op[0 - 2, y_idx, :]
        Save u_x this way also ensure continuous memory space, which gives a bit of performance enhancement
        """
        pass

    @abc.abstractmethod
    def diffusion_x(self, xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
        """
        u_xx
        This should be row oriented, meaning the first dimension is y and the second dimension is x
        We do this because when we apply diff_op, we do this diff_op[0 - 2, y_idx, :]
        Save u_xx this way also ensure continuous memory space, which gives a bit of performance enhancement
        """
        pass

    @abc.abstractmethod
    def convection_y(self, xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
        """
        u_y
        This should be row oriented, meaning the first dimension is x and the second dimension is y
        We do this because when we apply diff_op, we do this diff_op[0 - 2, x_idx, :]
        Save u_y this way also ensure continuous memory space, which gives a bit of performance enhancement
        """
        pass

    @abc.abstractmethod
    def diffusion_y(self, xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
        """
        u_yy
        This should be row oriented, meaning the first dimension is x and the second dimension is y
        We do this because when we apply diff_op, we do this diff_op[0 - 2, x_idx, :]
        Save u_yy this way also ensure continuous memory space, which gives a bit of performance enhancement
        """
        pass

    @abc.abstractmethod
    def cross(self, xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
        """
        u_xy
        This assume usual convention that x as first dimension and y as second dimension
        """
        pass

    @abc.abstractmethod
    def reaction(self, xs: np.ndarray, ys: np.ndarray) -> Union[float, np.ndarray]:
        """
        coefficient of u
        """
        pass

    def differential_operator(self, states: Dict[str, np.ndarray]) -> Tuple:
        """
        Lx - differential operator. Equals Lx + Lxx
        This function is meant to cache result. For example, in the case of Black Scholes

        :param states: coordinates of the dimensions. Keys should be 'x' and 'y'
        """
        if self.is_dynamic_pde or not self.diff_ops:
            xs, ys = states[self.x_name], states[self.y_name]
            x_dim, y_dim = len(xs), len(ys)
            dx, d2x, x_int = xs[1:] - xs[:-1], xs[2:] - xs[:-2], xs[1: -1]
            dy, d2y, y_int = ys[1:] - ys[:-1], ys[2:] - ys[:-2], ys[1: -1]

            convection_x, diffusion_x = self.convection_x(x_int, y_int), self.diffusion_x(x_int, y_int)
            convection_y, diffusion_y = self.convection_y(x_int, y_int), self.diffusion_y(x_int, y_int)
            reaction = self.reaction(x_int, y_int) / 2  # reaction is split between x and y
            self.diff_ops[self.x_name] = self.generate_diff_operator(convection_x, diffusion_x, reaction,
                                                                     dx, d2x, [y_dim - 2, x_dim - 2])
            self.diff_ops[self.y_name] = self.generate_diff_operator(convection_y, diffusion_y, reaction,
                                                                     dy, d2y, [x_dim - 2, y_dim - 2])

        # TODO: should also return 'xy'
        return self.diff_ops[self.x_name], self.diff_ops[self.y_name]


# ================================= Template PDE =================================
class HeatPDE(PDE1D):
    def __init__(self):
        heat = Factor(name='Heat', spot=0, atm_vol=2, is_normal=True)
        super(HeatPDE, self).__init__(factor=heat, is_dynamic_pde=False)

    def convection(self, x: np.ndarray) -> np.ndarray:
        return 0

    def diffusion(self, x: np.ndarray) -> np.ndarray:
        return 1

    def reaction(self, x: np.ndarray) -> Union[float, np.ndarray]:
        return 0


# ================================= Black Scholes =================================
class BlackScholesPDE1D(PDE1D):
    def __init__(self, asset: str, spot: float, r: float, q: float, sig: float):
        super(BlackScholesPDE1D, self).__init__(Factor(name=asset, spot=spot, atm_vol=sig))
        self.r = r
        self.carry = r - q
        self.var = sig ** 2

    def convection(self, x: np.ndarray) -> np.ndarray:
        return self.carry * x

    def diffusion(self, x: np.ndarray) -> np.ndarray:
        return 0.5 * self.var * x * x

    def reaction(self, x: np.ndarray) -> Union[float, np.ndarray]:
        return self.r


class BlackScholesPDE2D(PDE2D):
    def __init__(self, equity1: EquityFactor, equity2: EquityFactor, r: float, rho: float):
        super(BlackScholesPDE2D, self).__init__(factors=[equity1, equity2], is_dynamic_pde=False)

        # common parameters
        self.r = r
        self.rho = rho

        # idiosyncratic parameters
        self.carry1 = r - equity1.q
        self.sig1 = equity1.atm_vol
        self.var1 = self.sig1 ** 2
        self.carry2 = r - equity2.q
        self.sig2 = equity2.atm_vol
        self.var2 = self.sig2 ** 2

    def convection_x(self, xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
        return self.carry1 * xs

    def diffusion_x(self, xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
        return 0.5 * self.var1 * xs * xs  # np.power is much slower

    def convection_y(self, xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
        return self.carry2 * ys

    def diffusion_y(self, xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
        return 0.5 * self.var2 * ys * ys

    def cross(self, xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
        return self.rho * self.sig1 * self.sig2 * np.outer(xs, ys)

    def reaction(self, xs: np.ndarray, ys: np.ndarray) -> Union[float, np.ndarray]:
        return self.r

