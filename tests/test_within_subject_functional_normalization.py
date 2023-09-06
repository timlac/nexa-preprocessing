import unittest
from copy import copy

import pandas as pd
import numpy as np
from nexa_preprocessing.normalization.functionals import within_subject_functional_normalization


class TestLowLevelNormalization(unittest.TestCase):

    def setUp(self):
        # Setup example data for testing
        self.x = np.array([[1, 2, 3],
                           [4, 5, 6],
                           [7, 8, 9],
                           [10, 11, 12]])

        self.subject_ids = np.array([1, 1, 2, 2])

        self.df = pd.DataFrame(self.x, columns=['feat1', 'feat2', 'feat3'])

    def test_standard_normalization(self):
        normalized_x = within_subject_functional_normalization(self.x, self.subject_ids, "standard")

        for subject_id in np.unique(self.subject_ids):
            rows = np.where(self.subject_ids == subject_id)
            mean = np.mean(normalized_x[rows], axis=0)
            std = np.std(normalized_x[rows], axis=0)

            # Assert that the mean is close to 0 (within a certain tolerance)
            self.assertTrue(np.allclose(mean, 0, atol=0.01))

            # Assert that the standard deviation is close to 1 (within a certain tolerance)
            self.assertTrue(np.allclose(std, 1, atol=0.01))

    def test_min_max_normalization(self):
        normalized_x = within_subject_functional_normalization(
            self.x, self.subject_ids, "min_max"
        )
        for subject_id in np.unique(self.subject_ids):
            rows = np.where(self.subject_ids == subject_id)

            # Check if all values are in the range [0, 1]
            all_in_range = np.all((0 <= normalized_x[rows]) & (normalized_x[rows] <= 1))
            self.assertTrue(all_in_range)

    def test_standard_normalization_df(self):
        normalized_df = within_subject_functional_normalization(self.df, self.subject_ids, "standard")

        for subject_id in np.unique(self.subject_ids):
            rows = np.where(self.subject_ids == subject_id)
            mean = np.mean(normalized_df.iloc[rows], axis=0)
            std = np.std(normalized_df.iloc[rows], axis=0)

            # Assert that the mean is close to 0 (within a certain tolerance)
            self.assertTrue(np.allclose(mean, 0, atol=0.01))

            # Assert that the standard deviation is close to 1 (within a certain tolerance)
            self.assertTrue(np.allclose(std, 1, atol=0.01))

    def test_min_max_normalization_df(self):
        normalized_df = within_subject_functional_normalization(
            self.df, self.subject_ids, "min_max"
        )
        for subject_id in np.unique(self.subject_ids):
            rows = np.where(self.subject_ids == subject_id)

            # Check if all values are in the range [0, 1]
            all_in_range = np.all((0 <= normalized_df.iloc[rows]) & (normalized_df.iloc[rows] <= 1))
            self.assertTrue(all_in_range)


if __name__ == '__main__':
    unittest.main()
