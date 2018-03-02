import numpy as np
import pandas as pd
import unittest
from sklearn.pipeline import FeatureUnion, Pipeline

from bdranalytics.pdlearn.pipeline import PdFeatureUnion, PdFeatureChain
from bdranalytics.pdlearn.preprocessing import PdLagTransformer, PdWindowTransformer


class TestLagTransformer(unittest.TestCase):
    def test_lagtransformer(self):
        orig_data = pd.DataFrame(data=np.arange(15).reshape(
            5, 3), columns=["col1", "col2", "col3"])
        lagged = PdLagTransformer(1).fit_transform(orig_data)
        np.testing.assert_array_equal(
            lagged.columns, ["col1_lag1", "col2_lag1", "col3_lag1"])
        np.testing.assert_array_equal(lagged.iloc[1, :], orig_data.iloc[0, :])
        np.testing.assert_array_equal(lagged.iloc[0, :], np.repeat(np.nan, 3))

    def test_lagtransformer_on_numpy(self):
        orig_data = np.arange(15).reshape(5, 3)
        lagged = PdLagTransformer(1).fit_transform(orig_data)
        np.testing.assert_array_equal(
            lagged.columns, ["0_lag1", "1_lag1", "2_lag1"])
        np.testing.assert_array_equal(lagged.iloc[1, :], orig_data[0, :])
        np.testing.assert_array_equal(lagged.iloc[0, :], np.repeat(np.nan, 3))

    def test_windowtransformer(self):
        orig_data = pd.DataFrame(data=np.arange(
            14, -1, -1).reshape(5, 3), columns=["col1", "col2", "col3"])
        result = PdWindowTransformer(
            lambda window: window.max(), window=2).fit_transform(orig_data)
        np.testing.assert_array_equal(
            result.columns, ["col1_window2", "col2_window2", "col3_window2"])
        np.testing.assert_array_equal(result.iloc[0, :], np.repeat(np.nan, 3))
        # orig data is [ [14, 13, 12], [11, 10, 9],.., thus rolling max at row 1 should be values of row 0
        np.testing.assert_array_equal(result.iloc[1, :], orig_data.iloc[0, :])

    def test_windowtransformer_on_numpy(self):
        orig_data = np.arange(14, -1, -1).reshape(5, 3)
        result = PdWindowTransformer(
            lambda window: window.max(), window=2).fit_transform(orig_data)
        np.testing.assert_array_equal(
            result.columns, ["0_window2", "1_window2", "2_window2"])
        np.testing.assert_array_equal(result.iloc[0, :], np.repeat(np.nan, 3))
        # orig data is [ [14, 13, 12], [11, 10, 9],.., thus rolling max at row 1 should be values of row 0
        np.testing.assert_array_equal(result.iloc[1, :], orig_data[0, :])

    def test_featureunion(self):
        orig_data = pd.DataFrame(data=np.arange(15).reshape(
            5, 3), columns=["col1", "col2", "col3"])
        result = PdFeatureUnion([
            ('lag', PdLagTransformer(1)),
            ('window', PdWindowTransformer(lambda window: window.max(), window=2))]
        ).fit_transform(orig_data)
        np.testing.assert_array_equal(result.columns,
                                      ["col1_lag1", "col2_lag1", "col3_lag1", "col1_window2", "col2_window2",
                                       "col3_window2"])
        np.testing.assert_array_equal(
            result.iloc[:, 0:3],
            PdLagTransformer(1).fit_transform(orig_data))
        np.testing.assert_array_equal(
            result.iloc[:, 3:6],
            PdWindowTransformer(lambda window: window.max(), window=2).fit_transform(orig_data))
        np.testing.assert_array_equal(result,
                                      FeatureUnion([
                                          ("lag", PdLagTransformer(1)),
                                          ("window", PdWindowTransformer(
                                              lambda window: window.max(), window=2))
                                      ]).fit_transform(orig_data))

    def test_featurechain(self):
        orig_data = pd.DataFrame(data=np.arange(15).reshape(
            5, 3), columns=["col1", "col2", "col3"])
        result = PdFeatureChain([
            ('lag', PdLagTransformer(1)),
            ('window', PdWindowTransformer(lambda window: window.max(), window=2))]).fit_transform(orig_data)
        np.testing.assert_array_equal(result.columns,
                                      ["col1_lag1_window2", "col2_lag1_window2", "col3_lag1_window2"])
        np.testing.assert_array_equal(
            result,
            PdWindowTransformer(lambda window: window.max(), window=2).fit_transform(
                PdLagTransformer(1).fit_transform(orig_data)
            )
        )
        np.testing.assert_array_equal(result,
                                      Pipeline(steps=[
                                          ("lag", PdLagTransformer(1)),
                                          ("window", PdWindowTransformer(
                                              lambda window: window.max(), window=2))
                                      ]).fit_transform(orig_data))
