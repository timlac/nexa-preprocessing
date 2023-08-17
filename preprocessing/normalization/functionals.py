import numpy as np

from preprocessing.normalization.methods import select_scaler


def within_subject_functional_normalization(x: np.ndarray,
                                            subject_ids: np.ndarray,
                                            method: str) -> np.ndarray:
    """
    Normalization function whereby each subject is normalized separately

    :param x: matrix with shape (observations, features)
    :param subject_ids: np Array with subject_ids
    :param method: normalization method
    :return: normalized matrix
    """
    # iterate over all subject ids
    for subject_id in np.unique(subject_ids):

        # select normalization method and create a scaler for the specific subject
        scaler = select_scaler(method)

        # normalize the rows corresponding for the specific subject
        rows = np.where(subject_ids == subject_id)
        x[rows] = scaler.fit_transform(x[rows])
    return x
