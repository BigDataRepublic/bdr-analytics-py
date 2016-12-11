from bdranalytics.pandaspipeline.transformers import PdLagTransformer, PdWindowTransformer, PdFeatureUnion, \
    PdFeatureChain
import unittest
import pandas as pd
import numpy as np


class TestLagTransformer(unittest.TestCase):

    def test_lagtransformer(self):
        origData = pd.DataFrame(data=np.arange(15).reshape(5, 3), columns=["col1", "col2", "col3"])
        lagged = PdLagTransformer(1).fit_transform(origData)
        np.testing.assert_array_equal(lagged.columns, ["col1_lag1", "col2_lag1", "col3_lag1"])
        np.testing.assert_array_equal(lagged.iloc[1, :], origData.iloc[0, :])
        np.testing.assert_array_equal(lagged.iloc[0, :], np.repeat(np.nan, 3))

