import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class WeightOfEvidenceEncoder(BaseEstimator, TransformerMixin):
    """
    Feature-engineering class that transforms a high-capacity categorical value
    into Weigh of Evidence scores. Can be used in sklearn pipelines.
    """

    def __init__(self, verbose=0, cols=None, return_df=True,
                 smooth=0.5, fillna=0, dependent_variable_values=None):
        """
        :param smooth: value for additive smoothing, to prevent divide by zero
        """
        # make sure cols is a list of strings
        if not isinstance(cols, list):
            cols = [cols]

        self.stat = {}
        self.return_df = return_df
        self.verbose = verbose
        self.cols = cols
        self.smooth = smooth
        self.fillna = fillna
        self.dependent_variable_values = dependent_variable_values

    def fit(self, X, y):

        if not isinstance(X, pd.DataFrame):
            raise TypeError(
                'Input should be an instance of pandas.DataFrame()')

        if self.dependent_variable_values is not None:
            y = self.dependent_variable_values

        df = X[self.cols].copy()
        y_col_index = len(df.columns) + 1
        df[y_col_index] = np.array(y)

        def get_totals(x):
            total = np.size(x)
            pos = max(float(np.sum(x)), self.smooth)
            neg = max(float(total - pos), self.smooth)
            return pos, neg

        # get the totals per class
        total_positive, total_negative = get_totals(y)
        if self.verbose:
            print("total positives {:.0f}, total negatives {:.0f}".format(
                total_positive, total_negative))

        def compute_bucket_woe(x):
            bucket_positive, bucket_negative = get_totals(x)
            return np.log(bucket_positive / bucket_negative)

        # compute WoE scores per bucket (category)
        stat = {}
        for col in self.cols:

            if self.verbose:
                print(
                    "computing weight of evidence for column {:s}".format(col))

            stat[col] = ((df.groupby(col)[y_col_index].agg(compute_bucket_woe)
                          + np.log(total_negative / total_positive)).to_dict())

        self.stat = stat

        return self

    def transform(self, X, y=None):

        if not isinstance(X, pd.DataFrame):
            raise TypeError(
                'Input should be an instance of pandas.DataFrame()')

        df = X.copy()

        # join the WoE stats with the data
        for col in self.cols:

            if self.verbose:
                print("transforming categorical column {:s}".format(col))

            stat = pd.DataFrame.from_dict(self.stat[col], orient='index')

            ser = (pd.merge(df, stat, left_on=col, right_index=True, how='left')
                   .sort_index()
                   .reindex(df.index))[0]

            # fill missing values with
            if self.verbose:
                print("{:.0f} NaNs in transformed data".format(
                    ser.isnull().sum()))
                print("{:.4f} mean weight of evidence".format(ser.mean()))

            df[col] = np.array(ser.fillna(self.fillna))

        if not self.return_df:
            out = np.array(df)
        else:
            out = df

        return out


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


class StringIndexer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.dictionaries = dict()
        self.columns = list()

    def fit(self, X, y=None):
        self.columns = X.columns.values
        for col in self.columns:
            categories = np.unique(X[col])
            self.dictionaries[col] = dict(
                zip(categories, range(len(categories))))
        return self

    def transform(self, X):
        column_array = []
        for col in self.columns:
            dictionary = self.dictionaries[col]
            na_value = len(dictionary) + 1
            transformed_column = X[col].apply(
                lambda x: dictionary.get(x, na_value))
            column_array.append(transformed_column.values.reshape(-1, 1))
        return np.hstack(column_array)


class LeaveOneOutEncoder(BaseEstimator, TransformerMixin):
    """
    Leave one out transformation for high-capacity categorical variables.
    """

    def __init__(self, with_stdevs=True):

        self.with_stdevs = with_stdevs
        self.means = {}
        self.stdevs = {}

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        df = X.copy()
        for col in self.means.keys():

            mean_col_name = "{:s}_MEAN".format(col)
            df[mean_col_name] = df.merge(pd.DataFrame(self.means[col]),
                                         how='left', left_on=[col], right_index=True)['y']
            if self.with_stdevs:
                std_col_name = "{:s}_STD".format(col)
                df[std_col_name] = df.merge(pd.DataFrame(self.stdevs[col]),
                                            how='left', left_on=[col], right_index=True)['y']

            df.drop(col, axis=1, inplace=True)

        return df

    def fit_transform(self, X, y):
        """will be used during pipeline fit"""
        df = X.copy()
        df['y'] = y
        for col in df.columns.difference(['y']):

            mean_col_name = "{:s}_MEAN".format(col)

            grouped = df.groupby(col)['y']

            self.means[col] = grouped.mean()
            df[mean_col_name] = grouped.transform(self._loo_means)

            if self.with_stdevs:
                std_col_name = "{:s}_STD".format(col)
                self.stdevs[col] = grouped.std()
                df[std_col_name] = grouped.transform(self._loo_stdevs)

            df.drop(col, axis=1, inplace=True)

        df.drop('y', axis=1, inplace=True)
        return df

    def _loo_means(self, s):
        n = len(s)
        loo_means = (s.sum() - s) / (n - 1)
        return loo_means * np.random.normal(loc=1.0, scale=0.01, size=n)

    def _loo_stdevs(self, s):
        n = len(s)
        if n > 1:
            loo_means = self._loo_means(s)
            sum_of_sq = n * s.std() ** 2
            loo_stdevs = np.sqrt(
                abs((sum_of_sq - (s - s.mean()) * (s - loo_means))) / (n - 1))
        else:
            loo_stdevs = np.array([0])

        return loo_stdevs * np.random.normal(loc=1.0, scale=0.01, size=n)
