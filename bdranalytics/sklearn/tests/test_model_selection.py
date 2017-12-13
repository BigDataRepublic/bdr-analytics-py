import numpy as np
import pandas as pd
import unittest

from bdranalytics.sklearn.model_selection import GrowingWindow, IntervalGrowingWindow


def create_time_series_data_set(start_date=pd.datetime(year=2000, month=1, day=1), n_rows=100):

    end_date = start_date + pd.Timedelta(days=n_rows-1)

    ds = np.random.rand(n_rows)

    X = pd.DataFrame(ds,
                     columns=['variable'],
                     index=pd.date_range(start_date, end_date))

    y = np.random.randint(2, size=(n_rows,))

    return X, y


class TestGrowingWindow(unittest.TestCase):

    def test_n_splits(self):
        assert GrowingWindow(4).get_n_splits(np.arange(15).reshape(3,5)) == 4

    def test_n_splits_returned(self):
        assert len(list(GrowingWindow(4).split(np.arange(15).reshape(3,5), np.arange(3)))) == 4

    def test_n_splits_testsize(self):
        for train, test in GrowingWindow(4).split(np.arange(15).reshape(5,3), np.arange(5)):
            assert len(test) == 1

    def test_n_splits_testsize2(self):
        for i, (train, test) in zip(range(4), GrowingWindow(4).split(np.arange(15).reshape(5,3), np.arange(5))):
            assert len(train) == i+1


class TestIntervalGrowingWindow(unittest.TestCase):

    def test_split_on_index(self):

        X, y = create_time_series_data_set()

        cv = IntervalGrowingWindow(
            test_start_date=pd.datetime(year=2000, month=2, day=1),
            test_end_date=pd.datetime(year=2000, month=3, day=1),
            test_size='7D')

        self.assertTrue(len(list(cv.split(X, y))) == 4)

    def test_split_on_array(self):

        X, y = create_time_series_data_set()

        test_size_in_days = 7

        cv = IntervalGrowingWindow(
            timestamps=X.index.values,
            test_start_date=pd.datetime(year=2000, month=2, day=1),
            test_end_date=pd.datetime(year=2000, month=3, day=1),
            test_size=pd.Timedelta(days=test_size_in_days))

        self.assertTrue(len(list(cv.split(X, y))) == 4)

    def test_split_test_size(self):

        X, y = create_time_series_data_set()

        test_size_in_days = 7

        cv = IntervalGrowingWindow(
            test_start_date=pd.datetime(year=2000, month=2, day=1),
            test_end_date=pd.datetime(year=2000, month=3, day=1),
            test_size=pd.Timedelta(days=test_size_in_days))

        for _, test in cv.split(X, y):
            self.assertTrue(len(test) == test_size_in_days)

    def test_split_with_train_size(self):

        X, y = create_time_series_data_set()

        train_size_in_days = 14

        cv = IntervalGrowingWindow(
            test_start_date=pd.datetime(year=2000, month=2, day=1),
            test_end_date=pd.datetime(year=2000, month=3, day=1),
            test_size=pd.Timedelta(days=7),
            train_size=pd.Timedelta(days=train_size_in_days))

        for train, _ in cv.split(X, y):
            self.assertTrue(len(train) == train_size_in_days)

    def test_n_splits(self):

        X, y = create_time_series_data_set()

        cv = IntervalGrowingWindow(
            test_start_date=pd.datetime(year=2000, month=2, day=1),
            test_end_date=pd.datetime(year=2000, month=3, day=1),
            test_size=pd.Timedelta(days=7))

        self.assertTrue(cv.get_n_splits(X) == 4)




