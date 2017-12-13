import random
import unittest

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.dummy import DummyRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from bdranalytics.sklearn.preprocessing import WeightOfEvidenceEncoder


class TestPreprocessing(unittest.TestCase):

    def verify_numeric(self, X_test):
        for dt in X_test.dtypes:
            numeric = False
            if np.issubdtype(dt, int) or np.issubdtype(dt, float):
                numeric = True

            self.assertTrue(numeric)

    def create_dataset(self, n_rows=1000):
        """
        Creates a data set with some categorical variables
        """
        ds = [[
            random.random(),
            random.random(),
            random.choice(['A', 'B', 'C']),
            random.choice(['A', 'B', 'C']),
            random.choice(['A', 'B', 'C', None]),
            random.choice(['A', 'B', 'C'])
        ] for _ in range(n_rows)]

        X = pd.DataFrame(ds, columns=['c1', 'c2', 'c3', 'c4', 'c5', 'c6'])
        y = np.random.randint(2, size=(n_rows,))

        return X, y

    def test_weight_of_evidence(self):
        """
        Unit test for WeightOfEvidenceEncoder class
        """
        # generate some training data
        cols = ['c3', 'c4', 'c5', 'c6']
        n_rows = 100
        X_train, y_train = self.create_dataset(n_rows=n_rows)

        # independent data set to-be-transformed
        X_test, _ = self.create_dataset(n_rows=10)

        # add unseen category to catch NaN filling behavior
        X_test.loc[0, 'c3'] = 'Z'

        # data frame case
        enc = WeightOfEvidenceEncoder(verbose=1, cols=cols)
        enc.fit(X_train, y_train)
        self.verify_numeric(enc.transform(X_test))

        # numpy array case
        enc_np = WeightOfEvidenceEncoder(verbose=0, return_df=False, cols=cols)
        enc_np.fit(X_train, y_train)
        output_array_enc_np = enc_np.transform(X_test)  # save for following tests
        self.assertTrue(isinstance(output_array_enc_np, np.ndarray))

        # external dep var, DIFFERENT from y_train
        enc_ext = WeightOfEvidenceEncoder(verbose=1, cols=cols, return_df=False,
                                          dependent_variable_values=np.random.randint(2, size=(n_rows,)))
        enc_ext.fit(X_train, y_train)
        self.assertTrue(np.array_equal(output_array_enc_np, enc_ext.transform(X_test)) is False)

        # external dep var, SAME y_train
        enc_ext = WeightOfEvidenceEncoder(verbose=1, cols=cols, return_df=False,
                                          dependent_variable_values=y_train)
        enc_ext.fit(X_train, y_train)
        self.assertTrue(np.array_equal(output_array_enc_np, enc_ext.transform(X_test)) is True)

    def create_regression_dataset(self, n_rows=1000):
        """
        Creates a data set with only numerical data
        """
        ds = np.random.rand(n_rows, 2)
        X = pd.DataFrame(ds, columns=['c1', 'c2'])
        y = np.random.rand(n_rows)
        return X, y

    def test_dummy_pipeline(self):
        """
        Just checking setup of a dummy regressor in a pipeline
        :return: None
        """
        X, y = self.create_regression_dataset(n_rows=20)
        predictor_constant = 3
        predictor = DummyRegressor(strategy="constant", constant=predictor_constant)
        y_hat = Pipeline([("predict", predictor)]).fit(X, y).predict(X)
        print(y_hat)
        np.allclose(y_hat, np.repeat(predictor_constant, len(y)))

    def test_scaled_target(self):
        X, y = self.create_regression_dataset(n_rows=20)
        y_mean = np.mean(y)
        predictor_constant = 0  # 0 will be multiplied by std , and then added to the mean
        predictor = DummyRegressor(strategy="constant", constant=predictor_constant)
        scaler = StandardScaler()
        y_hat = Pipeline([("predict", ScaledRegressor(scaler, predictor))]).fit(X, y).predict(X)
        print(y_hat)
        np.allclose(y_hat, np.repeat(y_mean, len(y)))


class ScaledRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, scaler, model, *args, **kwargs):
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
        print(y_scaled)
        self.model.fit(X, self._to_vector(y_scaled))

    def predict(self, X):
        return self._to_vector(
            self.scaler.inverse_transform(
                self._to_matrix(
                    self.model.predict(X)
                )
            )
        )


if __name__ == '__main__':
    unittest.main()
