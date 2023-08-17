import numpy as np

from itertools import compress

from typing import List

from preprocessing.normalization.methods import select_scaler


def within_subject_low_level_normalization(slices: List[np.ndarray],
                                           subject_ids: np.ndarray,
                                           method: str) -> List[np.ndarray]:
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
            slices[idx] = scaler.transform(slices[idx])
    return slices


def get_subject_specific_chunks(slices: List[np.ndarray],
                                subject_ids: np.ndarray):
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

        # stack the list to get a single array
        subject_array = np.vstack(subject_array_list)

        subject_indices = np.where(subject_boolean_indices)[0]

        print(type(subject_indices))

        yield subject_array, subject_indices

