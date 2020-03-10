
import abc
import numpy as np
from typing import Union


class PDE1D(abc.ABC):
    def __init__(self, spot: float, atm_vol: float, is_normal: bool = False):
        self.spot = spot  # for anchoring the range of grids
        self.atm_vol = atm_vol  # for deciding the range of grids
        self.is_normal = is_normal

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

    @abc.abstractmethod
    def lx(self, x: np.ndarray, recalculate: bool = False) -> np.ndarray:
        """
        Lx - differentiation operator. Equals Lx + Lxx
        This function is meant to cache result. For example, in the case of Black Scholes
        """
        pass

    @staticmethod
    def get_lx(convection: np.ndarray, diffusion: np.ndarray, reaction: np.ndarray,
               dx: np.ndarray, d2x: np.ndarray) -> np.ndarray:
        """
        get differentiation operator matrix (Lx + Lxx). Support non-uniform mesh
        See Iain Clark formula (7.56)
        """
        dxi = dx[1:]
        dxp = dx[: -1]
        lx = np.zeros([3, dx.shape[0] - 1])
        lx[0, :] = (2 * diffusion - convection * dxp) / dxi / d2x
        lx[1, :] = (reaction + (convection * (dxp - dxi) - 2 * diffusion) / dxi / dxp)
        lx[2, :] = (2 * diffusion + convection * dxi) / dxp / d2x
        return lx


class PDE2D(abc.ABC):
    @abc.abstractmethod
    def convection_x(self, x: np.ndarray) -> np.ndarray:
        """
        u_x
        """
        pass

    @abc.abstractmethod
    def diffusion_x(self, x: np.ndarray) -> np.ndarray:
        """
        u_xx
        """
        pass

    @abc.abstractmethod
    def convection_y(self, y: np.ndarray) -> np.ndarray:
        """
        u_y
        """
        pass

    @abc.abstractmethod
    def diffusion_y(self, y: np.ndarray) -> np.ndarray:
        """
        u_yy
        """
        pass

    @abc.abstractmethod
    def cross(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        u_xy
        """
        pass

    @abc.abstractmethod
    def reaction(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        coefficient of u
        """
        pass

    @staticmethod
    def get_lxy(convection_x: np.ndarray, diffusion_x: np.ndarray, dx: np.ndarray, d2x: np.ndarray,
                convection_y: np.ndarray, diffusion_y: np.ndarray, dy: np.ndarray, d2y: np.ndarray,
                reaction: np.ndarray) -> np.ndarray:
        pass


class BlackScholesPDE1D(PDE1D):
    def __init__(self, s: float, r: float, q: float, sig: float):
        super(BlackScholesPDE1D, self).__init__(s, sig, is_normal=False)
        self.r = r
        self.carry = r - q
        self.var = sig ** 2
        self.lx = None

    def convection(self, x: np.ndarray) -> np.ndarray:
        return self.carry * x

    def diffusion(self, x: np.ndarray) -> np.ndarray:
        return 0.5 * self.var * x * x

    def reaction(self, x: np.ndarray) -> Union[float, np.ndarray]:
        return self.r

    def lx(self, x: np.ndarray, recalculate: bool = False) -> np.ndarray:
        if self.lx is None:
            dx = x[1:] - x[: -1]
            d2x = x[2:] - x[: -2]
            convection = self.convection(x)
            diffusion = self.diffusion(x)
            reaction = self.reaction(x)
            self.lx = PDE1D.get_lx(convection, diffusion, reaction, dx, d2x)
        return self.lx


class BlackScholesPDE2D(PDE2D):
    def __init__(self, r: float, q1: float, sig1: float, q2: float, sig2: float, rho: float):
        self.r = r
        self.carry1 = r - q1
        self.sig1 = sig1
        self.var1 = sig1 ** 2
        self.carry2 = r - q2
        self.sig2 = sig2
        self.var2 = sig2 ** 2
        self.rho = rho

    def convection_x(self, x: np.ndarray) -> np.ndarray:
        return self.carry1 * x

    def diffusion_x(self, x: np.ndarray) -> np.ndarray:
        return 0.5 * self.var1 * x * x  # np.power is much slower

    def convection_y(self, y: np.ndarray) -> np.ndarray:
        return self.carry2 * y

    def diffusion_y(self, y: np.ndarray) -> np.ndarray:
        return 0.5 * self.var2 * y * y

    def cross(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        return self.rho * self.sig1 * self.sig2 * np.outer(x, y)

    def reaction(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        return self.r
