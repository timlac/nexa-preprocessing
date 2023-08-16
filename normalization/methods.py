from enum import Enum
from sklearn.preprocessing import StandardScaler, MinMaxScaler


class Methods(str, Enum):
    min_max = "min_max"
    standard = "standard"


def select_scaler(method: str):
    if method == Methods.standard:
        scaler = StandardScaler()
    elif method == Methods.min_max:
        scaler = MinMaxScaler()
    else:
        raise RuntimeError("Something went wrong, no scaling method chosen")
    return scaler

