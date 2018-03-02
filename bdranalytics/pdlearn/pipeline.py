import pandas as pd
import six
from sklearn.base import BaseEstimator, TransformerMixin


class PdFeatureUnion(BaseEstimator, TransformerMixin):
    """Concatenates the result of multiple transformers"""

    def __init__(self, transformer_list, n_jobs=1, transformer_weights=None, debug=False):
        self.transformer_list = transformer_list
        self.debug = debug

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
            columns = Xt.columns if hasattr(Xt, "columns") else [
                "{}-{}".format(name, c) for c in range(Xt.shape[1])]
            Xt = pd.DataFrame(Xt, index=X.index, columns=columns)
            assert len(Xt) == len(X), "Transformer {} shouldn't change nr of rows. " \
                                      "Returned {} while original is {}".format(
                                          name, len(Xt), len(X))
            yield Xt

    def _print_columns(self, xts):
        for xt in xts:
            print(xt.columns)
            print("\r\n")

    def transform(self, X):
        xts = list(self.transformgen(X))
        if self.debug:
            self._print_columns(xts)
        try:
            return pd.concat(xts, axis=1, verify_integrity=True, join_axes=None)
        except:
            self._print_columns(xts)
            raise


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
                Xt = transform.fit(
                    Xt, y, **fit_params_steps[name]).transform(Xt)
            assert len(Xt) == len(X), "Transformer {} shouldn't change nr of rows. " \
                                      "Returned {} while original is {}".format(
                                          name, len(Xt), len(X))
        return self

    def transform(self, X):
        Xt = X
        for name, transform in self.steps:
            if transform is not None:
                Xt = pd.DataFrame(Xt)
                Xt = transform.transform(Xt)
                assert len(Xt) == len(X), "Transformer {} shouldn't change nr of rows. " \
                                          "Returned {} while original is {}".format(
                                              name, len(Xt), len(X))
        return pd.DataFrame(Xt)

    def fit_transform(self, X, y=None, **fit_params):
        return self.fit(X, y, **fit_params).transform(X)
