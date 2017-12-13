import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin


class ScaledRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, scaler, model):
        self.model = model
        self.scaler = scaler

    @staticmethod
    def _to_matrix(vector):
        return np.reshape(vector, (-1, 1))

    @staticmethod
    def _to_vector(matrix):
        return np.reshape(matrix, -1)

    def fit(self, X, y):
        y_scaled = self.scaler.fit_transform(self._to_matrix(y))
        self.model.fit(X, self._to_vector(y_scaled))

    def predict(self, X):
        return self._to_vector(
            self.scaler.inverse_transform(
                self._to_matrix(
                    self.model.predict(X)
                )
            )
        )

