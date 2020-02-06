
import abc
from typing import List
import numpy as np
from sklearn import preprocessing, linear_model

from .simulation import TimeSlice


class Fitter(abc.ABC):
    @abc.abstractmethod
    def _fit_predict(self, X, y) -> np.ndarray:
        pass

    def fit_predict(self, last_values: np.ndarray, time_slice: TimeSlice, factors: List[str]):
        X = np.hstack([time_slice.states(factor) for factor in factors])
        X = preprocessing.scale(X)
        return self._fit_predict(X, last_values)


class LASSOFitter(Fitter):
    def __init__(self, order: int = 3):
        self.transformer = preprocessing.PolynomialFeatures(order)
        self.fitter = linear_model.Lasso(alpha=0.001)

    def _fit_predict(self, X, y) -> np.ndarray:
        X = self.transformer.fit_transform(X)
        self.fitter.fit(X, y)
        return self.fitter.predict(X)
