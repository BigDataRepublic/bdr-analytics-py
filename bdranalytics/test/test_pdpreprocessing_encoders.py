import unittest

import numpy as np
import pandas as pd

from bdranalytics.pdpreprocessing.encoders import DateCyclicalEncoding, DateOneHotEncoding
from bdranalytics.pdpreprocessing.encoders import date_to_dateparts, date_to_cyclical


class TestDatePartitioner(unittest.TestCase):
    def test_date_to_dateparts(self):
        orig_data = pd.DataFrame(data=np.arange(
            np.datetime64('2011-07-11'), np.datetime64('2011-07-18')
            ).reshape(7, 1), columns=["thedate"])
        splitted_data = date_to_dateparts(orig_data, 'thedate', new_col_name_prefix='prefix')

        expected_columns = ["prefix_{}".format(x) for x in ["DAY", "DAY_OF_WEEK", "HOUR", "MINUTE", "MONTH", "SECOND"]]
        # no additional columns
        np.testing.assert_array_equal(list(set(splitted_data.columns)-set(expected_columns)), list())
        # no missing columns
        np.testing.assert_array_equal(list(set(expected_columns)-set(splitted_data.columns)), list())
        monday = 0
        tuesday = 1
        np.testing.assert_array_equal(splitted_data.loc[0, expected_columns], [11, monday, 0, 0, 7, 0])
        np.testing.assert_array_equal(splitted_data.loc[1, expected_columns], [12, tuesday, 0, 0, 7, 0])

    def test_dateparts_to_circular(self):
        orig_data = pd.DataFrame(data=np.arange(
            np.datetime64('2011-07-11'), np.datetime64('2011-07-18')
        ).reshape(7, 1), columns=["thedate"])
        circular_data = date_to_cyclical(orig_data, 'thedate', new_col_name_prefix= 'prefix')

        intermediate_columns = ["prefix_{}".format(x) for x in ["DAY", "DAY_OF_WEEK", "HOUR", "MINUTE", "MONTH", "SECOND"]]
        expected_columns = ["{}_{}".format(x, y) for y in ["COS","SIN"] for x in intermediate_columns]
        # no additional columns
        np.testing.assert_array_equal(list(set(circular_data.columns)-set(expected_columns)), list())
        # no missing columns
        np.testing.assert_array_equal(list(set(expected_columns)-set(circular_data.columns)), list())
        # correct result compared to just splitting the columns
        splitted_data = date_to_dateparts(orig_data, 'thedate', new_col_name_prefix='prefix')
        sin_columns = ["{}_{}".format(x, y) for y in ["SIN"] for x in intermediate_columns]
        np.testing.assert_array_equal(circular_data.loc[:, sin_columns], np.sin(splitted_data.loc[:, intermediate_columns] / (2.0*np.pi*np.array([31, 7, 24, 60, 12, 60]))))
        cos_columns = ["{}_{}".format(x, y) for y in ["COS"] for x in intermediate_columns]
        np.testing.assert_array_equal(circular_data.loc[:, cos_columns], np.cos(splitted_data.loc[:, intermediate_columns] / (2.0*np.pi*np.array([31, 7, 24, 60, 12, 60]))))

    def test_dateonehotencoding(self):
        orig_data = pd.DataFrame(data=np.arange(
            np.datetime64('2011-07-11'), np.datetime64('2011-07-18')
        ).reshape(7, 1), columns=["thedate"])
        y = np.repeat(0, 7)
        onehot = DateOneHotEncoding(['thedate'], drop=True).fit_transform(orig_data, y)
        print onehot

    def test_datecyclicalencoding(self):
        orig_data = pd.DataFrame(data=np.arange(
            np.datetime64('2011-07-11'), np.datetime64('2011-07-18')
        ).reshape(7, 1), columns=["thedate"])
        y = np.repeat(0, 7)

        # create splitted to also be able to calculate values
        splitted_data = date_to_dateparts(orig_data, 'thedate')

        circular_data = DateCyclicalEncoding(['thedate'], drop=True).fit_transform(orig_data, y)
        intermediate_columns = ["thedate_{}".format(x) for x in ["DAY", "DAY_OF_WEEK", "HOUR", "MINUTE", "MONTH", "SECOND"]]
        expected_columns = ["{}_{}".format(x, y) for y in ["COS", "SIN"] for x in intermediate_columns]
        # no additional columns
        np.testing.assert_array_equal(list(set(circular_data.columns)-set(expected_columns)), list())
        # no missing columns
        np.testing.assert_array_equal(list(set(expected_columns)-set(circular_data.columns)), list())
        sin_columns = ["{}_{}".format(x, y) for y in ["SIN"] for x in intermediate_columns]
        np.testing.assert_array_equal(circular_data.loc[:, sin_columns], np.sin(splitted_data.loc[:, intermediate_columns] / (2.0*np.pi*np.array([31, 7, 24, 60, 12, 60]))))
        cos_columns = ["{}_{}".format(x, y) for y in ["COS"] for x in intermediate_columns]
        np.testing.assert_array_equal(circular_data.loc[:, cos_columns], np.cos(splitted_data.loc[:, intermediate_columns] / (2.0*np.pi*np.array([31, 7, 24, 60, 12, 60]))))
