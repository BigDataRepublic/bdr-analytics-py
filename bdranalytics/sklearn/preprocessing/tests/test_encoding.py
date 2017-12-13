import random
import unittest

import numpy as np
import pandas as pd

from bdranalytics.sklearn.preprocessing import WeightOfEvidenceEncoder


class TestEncoding(unittest.TestCase):

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


if __name__ == '__main__':
    unittest.main()
