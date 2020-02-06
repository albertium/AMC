
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
        X = np.vstack([time_slice.state(factor) for factor in factors]).T
        return self._fit_predict(X, last_values)


class LASSOFitter(Fitter):
    def __init__(self, order: int = 3):
        self.transformer = preprocessing.PolynomialFeatures(order)
        self.fitter = linear_model.Lasso(alpha=0.01)

    def _fit_predict(self, X, y) -> np.ndarray:
        preprocessing.scale(X, axis=0, copy=False)
        X = self.transformer.fit_transform(X)
        preprocessing.scale(X, axis=0, copy=False)

        X_subset = X[y > 0]
        y_subset = y[y > 0]

        self.fitter.fit(X_subset, y_subset)
        return self.fitter.predict(X)
