import numpy as np
from abc import ABCMeta
from sklearn.externals.six import with_metaclass
from sklearn.utils.validation import _num_samples
from sklearn.model_selection._split import _build_repr


class GrowingWindow(with_metaclass(ABCMeta)):
    """Growing Window cross validator

    Provides train/test indices to split data in train/test sets.
    Divides the data in n_folds+1 slices.
    For split i [1..n_folds], slices [0..i} are train, slice i is test

    Parameters:
        n_folds : int, default=3
            Number of folds. Must be at least 1
    """

    def __init__(self, n_folds=3):
        self.n_folds = n_folds

    def __repr__(self):
        return _build_repr(self)

    def split(self, X, y=None, labels=None):
        """Generate indices to split data into training and test set.
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training data, where n_samples is the number of samples
            and n_features is the number of features.
        y : array-like, of length n_samples
            The target variable for supervised learning problems.
            ignored
        labels : array-like, with shape (n_samples,), optional
            Group labels for the samples used while splitting the dataset into
            train/test set.
            ignored
        Returns
        -------
        train : ndarray
            The training set indices for that split.
        test : ndarray
            The testing set indices for that split.
        """
        n = _num_samples(X)
        n_slices = self.n_folds + 1
        # loop from the first 2 folds to the total number of folds
        for i in range(2, n_slices + 1):
            # the split is the percentage at which to split the folds into train
            # and test. For example when i = 2 we are taking the first 2 folds out
            # of the total available. In this specific case we have to split the
            # two of them in half (train on the first, test on the second),
            # so split = 1/2 = 0.5 = 50%. When i = 3 we are taking the first 3 folds
            # out of the total available, meaning that we have to split the three of them
            # in two at split = 2/3 = 0.66 = 66% (train on the first 2 and test on the
            # following)
            split = float(i - 1) / i
            # as we loop over the folds X and y are updated and increase in size.
            # This is the data that is going to be split and it increases in size
            # in the loop as we account for more folds. If k = 300, with i starting from 2
            # the result is the following in the loop
            # i = 2
            # X = X_train[:(600)]
            # y = y_train[:(600)]
            #
            # i = 3
            # X = X_train[:(900)]
            # y = y_train[:(900)]
            # ....
            n_sub = int(np.floor(float(n * i) / n_slices))
            subset = range(0, n_sub)
            # X and y contain both the folds to train and the fold to test.
            # index is the integer telling us where to split, according to the
            # split percentage we have set above
            n_train = int(np.floor(n_sub * split))
            train_index = np.arange(0, n_train)
            test_index = np.arange(n_train, n_sub)
            yield train_index, test_index

    def get_n_splits(self, X, y=None, labels=None):
        """Returns the number of splitting iterations in the cross-validator
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training data, where n_samples is the number of samples
            and n_features is the number of features.
        y : object
            Always ignored, exists for compatibility.
        labels : object
            Always ignored, exists for compatibility.
        Returns
        -------
        n_splits : int
            Returns the number of splitting iterations in the cross-validator.
        """
        if X is None:
            raise ValueError("The X parameter should not be None")
        return self.n_folds


