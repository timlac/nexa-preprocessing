import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from itertools import compress
from typing import Union
import pandas as pd
from typing import List

from preprocessing.normalization.methods import select_scaler

from preprocessing.normalization.check_list_object_types import check_list_objects_type


def within_subject_low_level_normalization(slices: List[Union[np.ndarray, pd.DataFrame]],
                                           subject_ids: np.ndarray,
                                           method: str) -> List[Union[np.ndarray, pd.DataFrame]]:
    """
    This function takes a list of arrays, groups them according subject ids, specified in another array,
    and normalized the data in these groups.
    While still preserving the original list of arrays.

    :param slices: list of arrays with observational data
    :param subject_ids: array with all subject ids
    :param method: normalization method
    :return: normalized slices
    """
    for subject_array, subject_indices in get_subject_specific_chunks(slices, subject_ids):

        scaler = select_scaler(method)
        scaler.fit(subject_array)

        for idx in subject_indices:

            if isinstance(slices[idx], np.ndarray):
                slices[idx] = scaler.transform(slices[idx])

            elif isinstance(slices[idx], pd.DataFrame):
                slices[idx] = pd.DataFrame(scaler.transform(slices[idx]),
                                           columns=slices[idx].columns,
                                           index=slices[idx].index)
            else:
                raise ValueError("input array needs to be either a pd.Dataframe or np.ndarray")

    return slices


def get_subject_specific_chunks(slices: List[Union[np.ndarray, pd.DataFrame]],
                                subject_ids: np.ndarray) -> (List[Union[np.ndarray, pd.DataFrame]], np.ndarray):
    """
    :param slices: list of arrays with observational data
    :param subject_ids: array with all subject ids
    :return: a generator that yields pairs of
    (1) a single array of data belonging to a single subject id
    (2) the indices of data belonging to a single subject id
    """

    for subject_id in np.unique(subject_ids):
        subject_boolean_indices = (subject_ids == subject_id)

        # choose videos at subject_id index
        subject_array_list = list(compress(slices, subject_boolean_indices))

        try:
            object_types = check_list_objects_type(slices)
            if object_types == np.ndarray:
                subject_array = np.vstack(subject_array_list)
            elif object_types == pd.DataFrame:
                subject_array = pd.concat(subject_array_list)
        except ValueError as e:
            # Raise the error to terminate the generator
            raise e

        subject_indices = np.where(subject_boolean_indices)[0]

        yield subject_array, subject_indices