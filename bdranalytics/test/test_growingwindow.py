from bdranalytics.model_selection.growingwindow import GrowingWindow, IntervalGrowingWindow
import numpy as np
import unittest
import pandas as pd


def create_time_series_data_set(start_date, n_rows=1000):

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

    def test_n_splits_testsize(self):
        for i, (train, test) in zip(range(4), GrowingWindow(4).split(np.arange(15).reshape(5,3), np.arange(5))):
            assert len(train) == i+1


class TestIntervalGrowingWindow(unittest.TestCase):

    def test_split(self):

        X, y = create_time_series_data_set(start_date=pd.datetime(year=2000, month=1, day=1))

        cv = IntervalGrowingWindow(
            timestamps='index',
            test_start_date=pd.datetime(year=2000, month=2, day=1),
            test_end_date=pd.datetime(year=2000, month=3, day=1),
            test_size='7D',
            train_size='14D')

        for train, test in cv.split(X):
            print train, test

        self.assertTrue(1 == 6)

