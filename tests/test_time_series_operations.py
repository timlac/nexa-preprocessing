import unittest
import numpy as np
import pandas as pd
from nexa_preprocessing.utils.time_series_operations import (pad_time_series,
                                                             get_cols_as_arrays,
                                                             get_identifier_vals_as_array,
                                                             slice_by
                                                             )


class TestTimeSeriesOperations(unittest.TestCase):

    def test_pad_time_series(self):
        ts_list = [
            np.array([[1, 2], [3, 4]]),
            np.array([[5, 6]])
        ]

        expected_result = np.array(
            [
                [[1, 2], [3, 4]],
                [[5, 6], [-1000, -1000]]
            ]
        )

        result = pad_time_series(ts_list)
        self.assertTrue(np.array_equal(result, expected_result))

    def test_get_cols(self):
        data1 = {'A': [1, 2, 3], 'B': [4, 5, 6]}
        data2 = {'A': [7, 8, 9], 'B': [10, 11, 12]}

        df1 = pd.DataFrame(data1)
        df2 = pd.DataFrame(data2)

        slices = [df1, df2]
        COLS = ['A']
        expected_result = [np.array([[1], [2], [3]]), np.array([[7], [8], [9]])]

        result = get_cols_as_arrays(slices, COLS)
        self.assertTrue(np.array_equal(result, expected_result))

    def test_get_fixed_col(self):
        data1 = {'A': [1, 1, 1], 'B': [4, 5, 6]}
        data2 = {'A': [1, 1, 1], 'B': [10, 11, 12]}

        df1 = pd.DataFrame(data1)
        df2 = pd.DataFrame(data2)

        slices = [df1, df2]
        COL_NAME = 'A'
        expected_result = np.array([1, 1])

        result = get_identifier_vals_as_array(slices, COL_NAME)
        self.assertTrue(np.array_equal(result, expected_result))

    def assert_df_lists_equal(self, df_list1, df_list2):
        set1 = set()
        for df in df_list1:
            for _, row in df.iterrows():
                set1.add(frozenset(row))

        set2 = set()
        for df in df_list2:
            for _, row in df.iterrows():
                set2.add(frozenset(row))

        self.assertTrue(set1 == set2, "Assertion failed: The df lists are not equal")

    def test_slice_by(self):
        data = {'ID': [1, 1, 2, 2, 4, 4, 4], 'Value': [10, 20, 30, 40, 50, 60, 80]}
        df = pd.DataFrame(data)

        identifier = 'ID'
        expected_result = [pd.DataFrame({'ID': [1, 1], 'Value': [10, 20]}),
                           pd.DataFrame({'ID': [2, 2], 'Value': [30, 40]}),
                           pd.DataFrame({'ID': [4, 4, 4], 'Value': [50, 60, 80]})]

        result = slice_by(df, identifier)
        self.assert_df_lists_equal(result, expected_result)


if __name__ == '__main__':
    unittest.main()
    