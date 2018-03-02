import unittest

import numpy as np
from sklearn.dummy import DummyRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from bdranalytics.sklearn.preprocessing import ScaledRegressor


class TestPreprocessing(unittest.TestCase):

    @staticmethod
    def create_regression_dataset(n_rows=1000):
        """
        Creates a data set with only numerical data
        """
        X = np.random.rand(n_rows, 2)
        y = np.random.rand(n_rows)
        return X, y

    def test_dummy_pipeline(self):
        """
        Just checking setup of a dummy regressor in a pipeline
        :return: None
        """
        X, y = self.create_regression_dataset(n_rows=20)
        predictor_constant = 3
        predictor = DummyRegressor(
            strategy="constant", constant=predictor_constant)
        y_hat = Pipeline([("predict", predictor)]).fit(X, y).predict(X)
        np.allclose(y_hat, np.repeat(predictor_constant, len(y)))

    def test_scaled_target(self):
        X, y = self.create_regression_dataset(n_rows=20)
        y_mean = np.mean(y)
        predictor_constant = 0  # 0 will be multiplied by std , and then added to the mean
        predictor = DummyRegressor(
            strategy="constant", constant=predictor_constant)
        scaler = StandardScaler()
        y_hat = Pipeline([("predict", ScaledRegressor(scaler, predictor))]).fit(
            X, y).predict(X)
        np.allclose(y_hat, np.repeat(y_mean, len(y)))

    def test_scaled_target_with_set_params(self):
        X, y = self.create_regression_dataset(n_rows=20)
        y_mean = np.mean(y)
        predictor_constant = 10  # 0 will be multiplied by std , and then added to the mean
        predictor = DummyRegressor(
            strategy="constant", constant=predictor_constant)
        scaler = StandardScaler()
        pipeline = Pipeline([("predict", ScaledRegressor(scaler, predictor))])
        pipeline.set_params(predict__estimator__constant=0)
        y_hat = pipeline.fit(X, y).predict(X)
        np.allclose(y_hat, np.repeat(y_mean, len(y)))


if __name__ == '__main__':
    unittest.main()
