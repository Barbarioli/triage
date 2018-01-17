import functools
import io
import logging
import operator

import pandas


def feature_list(feature_dictionary):
    """Convert a feature dictionary to a sorted list.

    Args: feature_dictionary (dict)

    Returns: sorted list of feature names

    """
    return sorted(
        functools.reduce(
            operator.concat,
            (feature_dictionary[key] for key in feature_dictionary.keys())
        )
    )


def str_in_sql(values):
    return ','.join("'{}'".format(value) for value in values)


def df_from_query(db_engine, query_string, index, header='HEADER', read_csv_args=None):
    """Given a query, write the requested data to a DataFrame.

    Uses a COPY command into a StringIO and pandas.read_csv

    :param query_string: query to send
    :header: text to include in query indicating if a header should be saved in output
    :type query_string: str
    :type file_name: str
    :type header: str

    :return: none
    :rtype: none
    """
    copy_target = io.StringIO()
    logging.debug('Loading into dataframe query %s', query_string)
    copy_sql = 'COPY ({query}) TO STDOUT WITH CSV {head}'.format(
        query=query_string,
        head=header
    )
    conn = db_engine.raw_connection()
    cur = conn.cursor()
    cur.copy_expert(copy_sql, copy_target)
    copy_target.seek(0)
    read_csv_args = read_csv_args or {}
    dataframe = pandas.read_csv(copy_target, **read_csv_args)
    dataframe.set_index(index, inplace=True)
    return dataframe


def verify_dataframe_no_nulls(dataframe):
    columns_with_nulls = [
        column
        for column in dataframe.columns
        if dataframe[column].isnull().values.any()
    ]
    if len(columns_with_nulls) > 0:
        raise ValueError(
            "Imputation failed for the following features: %s" %
            columns_with_nulls
        )
