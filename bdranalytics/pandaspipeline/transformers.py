from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
import six


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
        return self.fit(X, y, **fit_params).transform(X)


class PdWindowTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, func, **rolling_params):
        self.func = func
        self.rolling_params = rolling_params

    def fit(self, X, y=None, **fit_params):
        return self

    def do_transform(self, dataframe):
        return (self.func(dataframe.rolling(**self.rolling_params))
                .rename(columns=lambda c: "{}_{}".format(c, "".join(
            ["{}{}".format(k, v) for k, v in self.rolling_params.items()]))))

    def transform(self, X):
        try:
            return self.do_transform(X)
        except AttributeError:
            return self.do_transform(pd.DataFrame(X))

    def fit_transform(self, X, y=None, **fit_params):
        return self.fit(X, y, **fit_params).transform(X)


class PdFeatureUnion(BaseEstimator, TransformerMixin):
    """Concatenates the result of multiple transformers"""

    def __init__(self, transformer_list, n_jobs=1, transformer_weights=None):
        self.transformer_list = transformer_list

    def fit(self, X, y=None, **fit_params):
        fit_params_steps = dict((name, {}) for name, step in self.transformer_list
                                if step is not None)
        for pname, pval in six.iteritems(fit_params):
            step, param = pname.split('__', 1)
            fit_params_steps[step][param] = pval

        for name, transform in self.transformer_list:
            if transform is None:
                pass
            transform.fit(X, y, **fit_params_steps[name])
        return self

    def transformgen(self, X):
        for name, transform in self.transformer_list:
            if transform is None:
                pass
            Xt = transform.transform(X)
            columns = Xt.columns if hasattr(Xt, "columns") else ["{}-{}".format(name, c) for c in xrange(Xt.shape[1])]
            Xt = pd.DataFrame(Xt, index=X.index, columns=columns)
            assert len(Xt) == len(X), "Transformer {} shouldn't change nr of rows. " \
                                      "Returned {} while original is {}".format(name, len(Xt), len(X))
            yield Xt

    def transform(self, X):
        xts = list(self.transformgen(X))
        return pd.concat(xts, axis=1, verify_integrity=True, join_axes=None)


class PdFeatureChain(BaseEstimator, TransformerMixin):
    """Passes a data set through a pipeline / chain of transformers.
    The output of the first transformer is fed into the next transformer.

    Similar to sklearn Pipeline, but does not work with predictor in final step."""

    def __init__(self, steps):
        self.steps = steps

    def fit(self, X, y=None, **fit_params):
        fit_params_steps = dict((name, {}) for name, step in self.steps
                                if step is not None)
        for pname, pval in six.iteritems(fit_params):
            step, param = pname.split('__', 1)
            fit_params_steps[step][param] = pval

        Xt = X
        for name, transform in self.steps:
            Xt = pd.DataFrame(Xt)
            if transform is None:
                pass
            elif hasattr(transform, "fit_transform"):
                Xt = transform.fit_transform(Xt, y, **fit_params_steps[name])
            else:
                Xt = transform.fit(Xt, y, **fit_params_steps[name]).transform(Xt)
            assert len(Xt) == len(X), "Transformer {} shouldn't change nr of rows. " \
                                      "Returned {} while original is {}".format(name, len(Xt), len(X))
        return self

    def transform(self, X):
        Xt = X
        for name, transform in self.steps:
            if transform is not None:
                Xt = pd.DataFrame(Xt)
                Xt = transform.transform(Xt)
                assert len(Xt) == len(X), "Transformer {} shouldn't change nr of rows. " \
                                          "Returned {} while original is {}".format(name, len(Xt), len(X))
        return pd.DataFrame(Xt)

    def fit_transform(self, X, y=None, **fit_params):
        return self.fit(X, y, **fit_params).transform(X)
