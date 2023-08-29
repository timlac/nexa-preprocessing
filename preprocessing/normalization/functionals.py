from copy import copy

import numpy as np
import pandas as pd

from typing import Union

from preprocessing.normalization.methods import select_scaler


def within_subject_functional_normalization(x: Union[np.ndarray, pd.DataFrame],
                                            subject_ids: np.ndarray,
                                            method: str) -> Union[np.ndarray, pd.DataFrame]:
    """
    Normalization function whereby each subject is normalized separately

    :param x: matrix with shape (observations, features)
    :param subject_ids: np Array with subject_ids
    :param method: normalization method
    :return: normalized matrix
    """

    # make a copy of x to not modify input variable
    ret_x = copy(x)

    # iterate over all subject ids
    for subject_id in np.unique(subject_ids):

        # select normalization method and create a scaler for the specific subject
        scaler = select_scaler(method)

        # normalize the rows corresponding for the specific subject
        rows = np.where(subject_ids == subject_id)[0]

        if isinstance(ret_x, pd.DataFrame):
            ret_x.iloc[rows] = scaler.fit_transform(ret_x.iloc[rows])
        elif isinstance(x, np.ndarray):
            ret_x[rows] = scaler.fit_transform(ret_x[rows])
        else:
            raise ValueError("input array x need to be either a pd.Dataframe or np.ndarray")

    return ret_x
