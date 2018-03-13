"""
The :mod:`bdranalytics.pdlearn` module contains adapters that allows you
to put :class:`pandas.DataFrame` instances into :mod:`sklearn` without
losing the column names.
:mod:`sklearn` already allows you to provide instances of :class:`pandas.DataFrame`,
but as it internally works with :class:`numpy.array`, column names are lost during transformation.
Here we provide adapters, which re-add the column names after the :mod:`sklearn` modifications.
"""
