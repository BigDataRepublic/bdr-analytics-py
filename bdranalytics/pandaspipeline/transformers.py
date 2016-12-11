from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd


# like sklearn's transformers, but then on pandas DataFrame
class PdLagTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, lag):
        self.lag = lag

    def fit(self, X, y=None, **fit_params):
        return self

    def do_transform(self, dataframe):
        return (dataframe.shift(self.lag)
                .rename(columns=lambda c: "{}_lag{}".format(c, self.lag)))

    def transform(self, X):
        try:
            return self.do_transform(X)
        except AttributeError:
            return self.do_transform(pd.DataFrame(X))

    def fit_transform(self, X, y=None, **fit_params):
        return self.fit(X, y).transform(X)


class PdWindowTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, window, func):
        self.window = window
        self.func = func

    def fit(self, X, y=None, **fit_params):
        return self

    def do_transform(self, dataframe):
        return (self.func(dataframe.rolling(self.window))
                .rename(columns=lambda c: "{}_window{}".format(c, self.window)))

    def transform(self, X):
        try:
            return self.do_transform(X)
        except AttributeError:
            return self.do_transform(pd.DataFrame(X))

    def fit_transform(self, X, y=None, **fit_params):
        return self.fit(X, y).transform(X)


class PdFeatureUnion(BaseEstimator, TransformerMixin):
    def __init__(self, union):
        self.union = union

    def fit(self, X, y=None, **fit_params):
        return PdFeatureUnion([one.fit(X, y) for one in self.union])

    def transform(self, X):
        return pd.concat([one.transform(X) for one in self.union], axis=1, join_axes=[X.index])

    def fit_transform(self, X, y=None, **fit_params):
        return self.fit(X, y).transform(X)


class PdFeatureChain(BaseEstimator, TransformerMixin):
    def __init__(self, chain):
        self.chain = chain

    def fit(self, X, y=None, **fit_params):
        return self

    def transform(self, X, y=None, **fit_params):
        result = X
        for element in self.chain:
            result = element.fit_transform(result, y)
        return result

    def fit_transform(self, X, y=None, **fit_params):
        return self.fit(X, y).transform(X)