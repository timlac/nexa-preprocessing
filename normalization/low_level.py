import numpy as np

from itertools import compress

import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from typing import List

from normalization.methods import select_scaler


def within_subject_low_level_normalization(slices: List[np.ndarray],
                                           subject_ids: np.ndarray,
                                           method: str) -> List[np.ndarray]:
    """
    :param slices: list of arrays
    :param subject_ids: np array with all subject ids
    :param method: normalization method
    :return: normalized slices
    """

    for subject_id in np.unique(subject_ids):
        boolean_indices = (subject_ids == subject_id)

        # fit on all videos in chunk
        slice_chunk = list(compress(slices, boolean_indices))
        slices_as_array = np.vstack(slice_chunk)
        scaler = select_scaler(method)
        scaler.fit(slices_as_array)

        # transform every video in chunk indices
        indices = np.where(boolean_indices)[0]
        for idx in indices:
            slices[idx] = scaler.transform(slices[idx])
    return slices
