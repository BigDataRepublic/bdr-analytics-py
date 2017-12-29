import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin


class ScaledRegressor(BaseEstimator, RegressorMixin):
    """Allows a regressor to work with a scaled target if it does not allow scaling itself.

    When fitting, the `y` will be transform using the `scaler`, before being passed to the `model.fit`.
    When predicting, the predicted y will be inverse transformed to obtain a y_hat in the original range of values.

    For example, this allows your regressor to predict manipulated targets (ie `log(y)`), without additional pre and
    postprocessing outside your sklearn pipeline

    Parameters
    ----------
    scaler : TransformerMixin
        The transformer which will be applied on the target before it is passed to the `model`

    estimator : RegressorMixin
        The regressor which will work in transformed target space

    Attributes
    ----------

    Examples
    >>> from sklearn.linear_model import LinearRegression
    >>> from sklearn.preprocessing import StandardScaler
    >>> from sklearn.pipeline import Pipeline
    >>> n_rows = 10
    >>> X = np.random.rand(n_rows, 2)
    >>> y = np.random.rand(n_rows)
    >>> regressor = LinearRegression()
    >>> scaler = StandardScaler()
    >>> pipeline = Pipeline([("predict", ScaledRegressor(scaler, regressor))])
    >>> y_hat = pipeline.fit(X, y).predict(X)
    """

    def __init__(self, scaler, estimator):
        self.estimator = estimator
        self.scaler = scaler

    @staticmethod
    def _to_matrix(vector):
        return np.reshape(vector, (-1, 1))

    @staticmethod
    def _to_vector(matrix):
        return np.reshape(matrix, -1)

    def fit(self, X, y):
        y_scaled = self.scaler.fit_transform(self._to_matrix(y))
        self.estimator.fit(X, self._to_vector(y_scaled))

    def predict(self, X):
        return self._to_vector(
            self.scaler.inverse_transform(
                self._to_matrix(
                    self.estimator.predict(X)
                )
            )
        )

