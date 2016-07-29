from bdranalytics.model_selection.growingwindow import GrowingWindow
import numpy as np


def test_n_splits():
    assert GrowingWindow(4).get_n_splits(np.arange(15).reshape(3,5)) == 4


def test_n_splits_returned():
    assert len(list(GrowingWindow(4).split(np.arange(15).reshape(3,5), np.arange(3)))) == 4


def test_n_splits_testsize():
    for train, test in GrowingWindow(4).split(np.arange(15).reshape(5,3), np.arange(5)):
        assert len(test) == 1


def test_n_splits_testsize():
    for i, (train, test) in zip(range(4), GrowingWindow(4).split(np.arange(15).reshape(5,3), np.arange(5))):
        assert len(train) == i+1
