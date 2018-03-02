import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from bdranalytics.sklearn.preprocessing import StringIndexer


def format_colname(prefix, suffix):
    return "{:s}_{:s}".format(prefix, suffix)


"""A dictionary to get a date part cardinality given a general name"""
__date_part_cardinality = {
    "MONTH": 12,
    "DAY": 31,
    "DAY_OF_WEEK": 7,
    "HOUR": 24,
    "MINUTE": 60,
    "SECOND": 60
}

"""A dictionary to get a date part extractor given a general name"""
__date_part_funcs = {
    "MONTH": lambda x: x.month,
    "DAY": lambda x: x.day,
    "DAY_OF_WEEK": lambda x: x.dayofweek,
    "HOUR": lambda x: x.hour,
    "MINUTE": lambda x: x.minute,
    "SECOND": lambda x: x.second
}


def date_to_dateparts(df, col_name, parts=list(__date_part_funcs.keys()), new_col_name_prefix=None):
    if new_col_name_prefix is None:
        new_col_name_prefix = col_name
    for part in parts:
        assert part in list(__date_part_funcs.keys()), \
            "part '{}' is not known. Available are {}".format(
                part, ", ".join(list(__date_part_funcs.keys())))
    return pd.DataFrame({
        format_colname(new_col_name_prefix, part):
        df[col_name].apply(__date_part_funcs.get(part))
        for part in parts}, index=df.index)


def date_to_cyclical(df, col_name, parts=list(__date_part_funcs.keys()), new_col_name_prefix=None):
    if new_col_name_prefix is None:
        new_col_name_prefix = col_name
    for part in parts:
        assert part in list(__date_part_funcs.keys()), \
            "part '{}' is not known. Available are {}".format(
                part, ", ".join(list(__date_part_funcs.keys())))
    names = [format_colname(new_col_name_prefix, part) for part in parts]
    names_sin = ["{:s}_SIN".format(name) for name in names]
    names_cos = ["{:s}_COS".format(name) for name in names]
    values = [df[col_name].apply(__date_part_funcs.get(part)) /
              (2.0 * np.pi * __date_part_cardinality.get(part)) for part in parts]
    values_sin = [col.apply(np.sin) for col in values]
    values_cos = [col.apply(np.cos) for col in values]
    result = pd.concat(values_sin + values_cos, axis=1)
    result.columns = names_sin + names_cos
    return result


def to_circular_variable(df, col_name, cardinality):
    return pd.DataFrame({
        # note that np.sin(df[col_name] / float(cardinalilty...)) gives different values, probably rounding
        "{:s}_SIN".format(col_name): df[col_name].apply(lambda x: np.sin(x / float(cardinality * 2 * np.pi))),
        "{:s}_COS".format(col_name): df[col_name].apply(lambda x: np.cos(x / float(cardinality * 2 * np.pi)))
    }, index=df.index)


class DateOneHotEncoding(BaseEstimator, TransformerMixin):
    """
    Feature-engineering class that transforms date columns into one hot encoding of the parts (day, hour, ..).
    The original date column will be removed.
    To be used by sklearn pipelines
    """

    def __init__(self, date_columns, parts=list(["DAY", "DAY_OF_WEEK", "HOUR", "MINUTE", "MONTH", "SECOND"]),
                 new_column_names=None, drop=True):
        """
        :param date_columns: the column names of the date columns to be expanded in one hot encodings
        :param new_column_names: the names to use as prefix for the generated column names
        :param drop: whether or not to drop the original column
        :param parts: the parts to extract from the date columns, and to then transform into one-hot encodings
        """
        self.drop = drop
        self.parts = parts
        if new_column_names is None:
            self.new_column_names = date_columns
        else:
            self.new_column_names = new_column_names
        self.date_columns = date_columns
        self.one_hot_encoding_model = OneHotEncoder(sparse=False, handle_unknown='ignore'
                                                    # , n_values=datepart_maxvalue
                                                    )
        self.encoding_pipeline = Pipeline([
            ('labeler', StringIndexer()),
            ('encoder', self.one_hot_encoding_model)
        ])
        assert (len(self.date_columns) == len(self.new_column_names)), \
            "length of new column names is not equal to given column names"

    def all_to_parts(self, X):
        parts = [date_to_dateparts(X, old_name, self.parts, new_name)
                 for old_name, new_name in zip(self.date_columns, self.new_column_names)]
        result = pd.concat(parts, axis=1, join_axes=[X.index])
        return result

    def fit(self, X, y):
        parts = self.all_to_parts(X)
        self.encoding_pipeline.fit(parts)
        # original column i is mapped to values in range resulting_indices[i] .. resulting_indices[i+1]
        resulting_indices = self.one_hot_encoding_model.feature_indices_
        active_features = self.one_hot_encoding_model.active_features_
        new_names = [''] * (np.max(resulting_indices) + 1)
        for i, item in enumerate(parts.columns):
            for j in range(resulting_indices[i], resulting_indices[i + 1]):
                new_names[j] = "{}-{}".format(item, j)
        self.fitted_names = [new_names[i] for i in active_features]
        return self

    def transform_one_hots(self, X):
        np_frame = self.encoding_pipeline.transform(self.all_to_parts(X))
        return pd.DataFrame(np_frame, columns=self.fitted_names)

    def transform(self, X):
        new_columns = self.transform_one_hots(X)
        old_columns = X.drop(self.date_columns, axis=1,
                             inplace=False) if self.drop else X

        return pd.concat([old_columns, new_columns], axis=1, join_axes=[X.index])


class DateCyclicalEncoding(BaseEstimator, TransformerMixin):
    """
    Feature-engineering class that transforms date columns into cyclical numerical columns.
    The original date column will be removed.
    To be used by sklearn pipelines
    """

    def __init__(self, date_columns, parts=list(["DAY", "DAY_OF_WEEK", "HOUR", "MINUTE", "MONTH", "SECOND"]),
                 new_column_names=None, drop=True):
        """
        :param date_columns: the column names of the date columns to be expanded in one hot encodings
        :param new_column_names: the names to use as prefix for the generated column names
        :param drop: whether or not to drop the original column
        :param parts: the parts to extract from the date columns, and to then transform into one-hot encodings
        """
        self.parts = parts
        self.drop = drop
        if new_column_names is None:
            self.new_column_names = date_columns
        else:
            self.new_column_names = new_column_names
        self.date_columns = date_columns
        assert (len(self.date_columns) == len(self.new_column_names))

    def all_to_cyclical_parts(self, X):
        parts = [date_to_cyclical(X, old_name, self.parts, new_name)
                 for old_name, new_name in zip(self.date_columns, self.new_column_names)]
        return pd.concat(parts, axis=1, join_axes=[X.index])

    def fit(self, X, y):
        return self

    def transform(self, X):
        new_columns = self.all_to_cyclical_parts(X)
        old_columns = X.drop(self.date_columns, axis=1,
                             inplace=False) if self.drop else X
        return pd.concat([old_columns, new_columns], axis=1, join_axes=[X.index])


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
