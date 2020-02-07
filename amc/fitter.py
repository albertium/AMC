
import abc
from typing import List, Union
import numpy as np
from sklearn import preprocessing, linear_model, pipeline

from .simulation import TimeSlice


class Fitter(abc.ABC):
    @abc.abstractmethod
    def _fit_predict(self, X, y) -> np.ndarray:
        pass

    def fit_predict(self, last_values: np.ndarray, time_slice: TimeSlice, factors: List[str],
                    mask: Union[None, np.ndarray] = None):
        if mask is not None:
            X = np.vstack([time_slice.state(factor) for factor in factors]).T[mask]
            y = last_values[mask]
            result = np.repeat(np.inf, len(last_values))
            result[mask] = self._fit_predict(X, y)
            return result

        else:
            X = np.vstack([time_slice.state(factor) for factor in factors]).T
            return self._fit_predict(X, last_values)


class LASSOFitter(Fitter):
    def __init__(self, order: int = 3):
        self.transformer = pipeline.make_pipeline(preprocessing.StandardScaler(),
                                                  preprocessing.PolynomialFeatures(order))
        self.fitter = linear_model.Lasso(alpha=0.01)

    def _fit_predict(self, X, y) -> np.ndarray:
        X = self.transformer.fit_transform(X)
        self.fitter.fit(X, y)
        return self.fitter.predict(X)
