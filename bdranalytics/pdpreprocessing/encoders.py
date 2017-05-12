import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline, FeatureUnion, make_pipeline
from bdranalytics.pipeline.encoders import ColumnSelector, StringIndexer

def format_colname(prefix, suffix):
    return "{:s}_{:s}".format(prefix, suffix)


def date_to_dateparts(df, col_name, new_col_name_prefix=None):
    if new_col_name_prefix is None:
        new_col_name_prefix = col_name
    return pd.DataFrame({format_colname(new_col_name_prefix, "MONTH"): df[col_name].apply(lambda x: x.month),
                         format_colname(new_col_name_prefix, "DAY"): df[col_name].apply(lambda x: x.day),
                         format_colname(new_col_name_prefix, "DAY_OF_WEEK"): df[col_name].apply(lambda x: x.dayofweek),
                         format_colname(new_col_name_prefix, "HOUR"): df[col_name].apply(lambda x: x.hour),
                         format_colname(new_col_name_prefix, "MINUTE"): df[col_name].apply(lambda x: x.minute),
                         format_colname(new_col_name_prefix, "SECOND"): df[col_name].apply(lambda x: x.second)
                         }, index=df.index)


def to_circular_variable(df, col_name, cardinality):
    return pd.DataFrame({
            # note that np.sin(df[col_name] / float(cardinalilty...)) gives different values, probably rounding
            "{:s}_SIN".format(col_name) : df[col_name].apply(lambda x: np.sin(x / float(cardinality * 2 * np.pi))),
            "{:s}_COS".format(col_name) : df[col_name].apply(lambda x: np.cos(x / float(cardinality * 2 * np.pi)))
        }, index=df.index)


def dateparts_to_circular(df, col_prefix):
    intermediate_columns = [format_colname(col_prefix, x) for x in ["DAY", "DAY_OF_WEEK", "HOUR", "MINUTE", "MONTH", "SECOND"]]
    radial = df.loc[:, intermediate_columns] / (2.0*np.pi*np.array([31, 7, 24, 60, 12, 60]))
    df_sin = np.sin(radial)
    df_sin.columns = ["{}_{}".format(x, y) for y in ["SIN"] for x in intermediate_columns]
    df_cos = np.cos(radial)
    df_cos.columns = ["{}_{}".format(x, y) for y in ["COS"] for x in intermediate_columns]
    return pd.concat(
        [
            df_sin,
            df_cos
        ], axis=1)


class DateOneHotEncoding(BaseEstimator, TransformerMixin):
    """
    Feature-engineering class that transforms date columns into one hot encoding of the parts (day, hour, ..).
    The original date column will be removed.
    To be used by sklearn pipelines
    """

    def __init__(self, date_columns, new_column_names=None, drop=True):
        self.drop_date_columns = drop
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
        assert (len(self.date_columns) == len(self.new_column_names))

    def all_to_parts(self, X):
        parts = [date_to_dateparts(X, old_name, new_name)
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
        old_columns = X.drop(self.date_columns, axis=1, inplace=False) if self.drop_date_columns else X

        return pd.concat([old_columns, new_columns], axis=1, join_axes=[X.index])


class DateCyclicalEncoding(BaseEstimator, TransformerMixin):
    """
    Feature-engineering class that transforms date columns into cyclical numerical columns.
    The original date column will be removed.
    To be used by sklearn pipelines
    """

    def __init__(self, date_columns, new_column_names=None, drop=True):
        self.drop_date_columns = drop
        if new_column_names is None:
            self.new_column_names = date_columns
        else:
            self.new_column_names = new_column_names
        self.date_columns = date_columns
        assert (len(self.date_columns) == len(self.new_column_names))

    def all_to_cyclical_parts(self, X):
        parts = [dateparts_to_circular(
            date_to_dateparts(X, old_name, new_name),
            new_name)
                 for old_name, new_name in zip(self.date_columns, self.new_column_names)]
        return pd.concat(parts, axis=1, join_axes=[X.index])

    def fit(self, X, y):
        return self

    def transform(self, X):
        new_columns = self.all_to_cyclical_parts(X)
        old_columns = X.drop(self.date_columns, axis=1, inplace=False) if self.drop_date_columns else X
        return pd.concat([old_columns, new_columns], axis=1, join_axes=[X.index])
