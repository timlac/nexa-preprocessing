import numpy as np

from itertools import compress

import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from typing import List

from normalization.methods import Methods


def within_subject_low_level_normalization(slices: List[pd.DataFrame],
                                           subject_ids: np.ndarray,
                                           method: str) -> List[np.ndarray]:
    """
    :param slices: list of dataframes
    :param subject_ids: np array with all subject ids
    :param method: normalization method
    :return: normalized slices
    """

    for slice_chunk, indices in chunk_data(subject_ids, slices):
        slices_as_array = np.vstack(slice_chunk)

        if method == Methods.standard:
            scaler = StandardScaler()
        elif method == Methods.min_max:
            scaler = MinMaxScaler()
        else:
            raise RuntimeError("Something went wrong, no scaling method chosen")

        # fit on all videos in chunk
        scaler.fit(slices_as_array)

        for idx in indices:
            # transform every video in chunk indices
            slices[idx] = scaler.transform(slices[idx])
    return slices


def chunk_data(subject_ids: np.ndarray,
               slices: List[pd.DataFrame]) -> (List[np.ndarray], np.ndarray):
    """
    :param subject_ids: list of subject ids
    :param slices: list of numpy arrays
    :return: list of dataframes corresponding to subject ids, indices of these items
    """

    for subject_id in np.unique(subject_ids):
        boolean_indices = (subject_ids == subject_id)
        indices = np.where(boolean_indices)[0]

        # get slices corresponding to chunk indices
        slice_chunk = list(compress(slices, boolean_indices))

        yield slice_chunk, indices