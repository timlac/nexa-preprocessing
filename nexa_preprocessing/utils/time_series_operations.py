from itertools import compress

import numpy as np
import pandas as pd
import os

from typing import List


def pad_time_series(ts_list: List[np.ndarray], padding_value: int = -1000) -> np.ndarray:
    """
    :param ts_list: a list of arrays where every element has shape (number of frames, number of features)
    :param padding_value: value to pad the time series with to make them of equal length
                            /this value should NOT be already existing in the time series
    :return: array where every element in the series has been padded with zeros and then transformed into a matrix
                with shape (number of observations, number of frames, number of features)
    """
    # obtain the longest element in the list
    max_length = max(map(len, ts_list))

    padded_list = []
    for series in ts_list:
        diff = max_length - len(series)

        # create an empty np array of appropriate size
        pad = np.full(shape=(diff, series.shape[1]), fill_value=padding_value)

        # concat with series and transpose in order to
        series = np.concatenate((series, pad))

        padded_list.append(series)

    return np.asarray(padded_list)


def get_cols_as_arrays(slices: List[pd.DataFrame], cols: List[str]) -> List[np.ndarray]:
    """
    :param slices: List of dataframes
    :param cols: list of column names to return
    :return: List of arrays with only selected columns
    """
    ret = []
    for df in slices:
        array = df[cols].values
        ret.append(array)
    return ret


def get_cols(slices: List[pd.DataFrame], cols: List[str]) -> List[pd.DataFrame]:
    """
    :param slices: List of dataframes
    :param cols: list of column names to return
    :return: List of arrays with only selected columns
    """
    ret = []
    for df in slices:
        ret.append(df[cols])
    return ret


def get_identifier_vals_as_array(slices: List[pd.DataFrame], identifier_column: str) -> np.ndarray:
    """
    :param slices: list of dataframes
    :param identifier_column: str, column name
    :return: np array with the extracted column value, one for each dataframe
    """
    ret = []
    for df in slices:
        array = df[identifier_column].values
        # assert that we have only extracted one identifier per dataframe
        if len(np.unique(array)) != 1:
            raise ValueError("something went wrong, more than one {} found for time series".format(identifier_column))
        else:
            ret.append(array[0])
    return np.asarray(ret)


def slice_by(df: pd.DataFrame, column_name_to_slice_by: str) -> List[pd.DataFrame]:
    """
    :param df: dataframe with multiple time series
    :param column_name_to_slice_by: str, column name to identify unique time series
    :return: list of dataframes.
    """
    ret = []
    for _, group in df.groupby(column_name_to_slice_by, sort=False):
        ret.append(group)
    return ret
