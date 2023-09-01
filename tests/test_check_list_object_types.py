import unittest
import numpy as np
import pandas as pd

from preprocessing.normalization.check_list_object_types import check_list_objects_type


class TestCheckListObjectsType(unittest.TestCase):
    def test_all_numpy_arrays(self):
        lst = [np.array([1, 2, 3]), np.array([4, 5, 6])]
        result = check_list_objects_type(lst)
        self.assertEqual(result, np.ndarray)

    def test_all_pandas_dataframes(self):
        lst = [pd.DataFrame({'A': [1, 2, 3]}), pd.DataFrame({'B': [4, 5, 6]})]
        result = check_list_objects_type(lst)
        self.assertEqual(result, pd.DataFrame)

    def test_mixed_types(self):
        lst = [np.array([1, 2, 3]), pd.DataFrame({'A': [4, 5, 6]})]
        with self.assertRaises(ValueError):
            check_list_objects_type(lst)

    def test_other_types(self):
        lst = [1, 2, 3]
        with self.assertRaises(ValueError):
            check_list_objects_type(lst)

    def test_empty_list(self):
        lst = []
        with self.assertRaises(ValueError):
            check_list_objects_type(lst)


if __name__ == '__main__':
    unittest.main()
