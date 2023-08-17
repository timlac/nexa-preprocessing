import unittest
from copy import copy

import numpy as np
from preprocessing.normalization.low_level import within_subject_low_level_normalization, get_subject_specific_chunks


class TestLowLevelNormalization(unittest.TestCase):

    def setUp(self):
        # Setup example data for testing
        self.slices = [
            np.array([[1, 2, 3],
                      [4, 5, 6]]),

            np.array([[7, 8, 9],
                      [10, 11, 12]]),

            np.array([[13, 14, 15],
                      [16, 17, 18]]),

            np.array([[19, 20, 21],
                      [22, 23, 24]])
        ]

        self.subject_ids = np.array([1, 1, 2, 2])

    def test_standard_normalization(self):
        normalized_slices = within_subject_low_level_normalization(
            copy(self.slices), self.subject_ids, "standard"
        )
        for subject_array, _ in get_subject_specific_chunks(normalized_slices, self.subject_ids):

            mean = np.mean(subject_array)
            std = np.std(subject_array)

            # Assert that the mean is close to 0 (within a certain tolerance)
            self.assertAlmostEqual(mean, 0, delta=0.01)

            # Assert that the standard deviation is close to 1 (within a certain tolerance)
            self.assertAlmostEqual(std, 1, delta=0.01)

    def test_min_max_normalization(self):
        normalized_slices = within_subject_low_level_normalization(
            copy(self.slices), self.subject_ids, "min_max"
        )
        for subject_array, _ in get_subject_specific_chunks(normalized_slices, self.subject_ids):

            # Check if all values are in the range [0, 1]
            all_in_range = np.all((0 <= subject_array) & (subject_array <= 1))
            self.assertTrue(all_in_range)


if __name__ == '__main__':
    unittest.main()
