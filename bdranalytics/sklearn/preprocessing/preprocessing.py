from sklearn.base import BaseEstimator
from sklearn.base import TransformerMixin


class ColumnSelector(BaseEstimator, TransformerMixin):
    def __init__(self, columns):
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        try:
            return X[self.columns]
        except:
            print("Could not find selected columns {:s} in available columns {:s}".format(
                self.columns, X.columns))
            raise
