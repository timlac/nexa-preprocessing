import unittest
from copy import copy

import numpy as np
from normalization.low_level import within_subject_low_level_normalization


class TestWithinSubjectNormalization(unittest.TestCase):

    def setUp(self):
        # Setup example data for testing
        self.slices = [
            np.array([[1, 2, 3], [4, 5, 6]]),  # Example data, adjust as needed
            np.array([[7, 8, 9], [10, 11, 12]]),
            np.array([[13, 14, 15], [16, 17, 18]]),
            np.array([[19, 20, 21], [22, 23, 24]])
        ]

        self.subject_ids = np.array([1, 1, 2, 2])
        self.method = "standard"

    def test_within_subject_low_level_normalization(self):
        normalized_slices = within_subject_low_level_normalization(
            copy(self.slices), self.subject_ids, self.method
        )

        i = 0
        # Loop through the normalized_slices and corresponding original slices

        for idx, original_slice in enumerate(self.slices):
            normalized_slice = normalized_slices[idx]
            i += 1
            # if i == 1:
            #     continue

            print(normalized_slice)
            print(original_slice)
            mean = np.mean(normalized_slice)
            std = np.std(normalized_slice)

            print(mean)
            print(std)

            # Assert that the mean is close to 0 (within a certain tolerance)
            self.assertAlmostEqual(mean, 0, delta=0.01)

            # Assert that the standard deviation is close to 1 (within a certain tolerance)
            self.assertAlmostEqual(std, 1, delta=0.01)


if __name__ == '__main__':
    unittest.main()
