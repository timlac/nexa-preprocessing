import numpy as np
import pandas as pd


def check_list_objects_type(input_list):
    numpy_array_flag = False
    dataframe_flag = False

    for item in input_list:
        if isinstance(item, np.ndarray):
            if dataframe_flag:
                raise ValueError("Mixed types")  # Raise an error for mixed types
            numpy_array_flag = True
        elif isinstance(item, pd.DataFrame):
            if numpy_array_flag:
                raise ValueError("Mixed types")  # Raise an error for mixed types
            dataframe_flag = True
        else:
            raise ValueError("Unknown type")  # Raise an error for other types

    if numpy_array_flag:
        return np.ndarray
    elif dataframe_flag:
        return pd.DataFrame
    else:
        raise ValueError("Empty list")  # Raise an error if the list is empty