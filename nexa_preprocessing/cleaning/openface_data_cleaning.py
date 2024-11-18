import numpy as np


class IndexIndicator:
    perfect_indices = []
    interpolatable_indices = []
    bad_indices = []


class OpenfaceDataCleaner:
    MIN_RATIO_GOOD_FRAMES = 0.85

    # openface parameters
    CONFIDENCE_THRESHOLD = 0.98
    SUCCESS_INDICATOR = 1

    # column names
    CONFIDENCE = "confidence"
    SUCCESS = "success"

    @classmethod
    def examine_data_quality_batch(cls, slices):
        index_indicator = IndexIndicator()

        for idx, df in enumerate(slices):
            ratio_successful, ratio_high_conf = cls.examine_data_quality(df)
            if ratio_successful < 1 or ratio_high_conf < 1:
                if ratio_successful < cls.MIN_RATIO_GOOD_FRAMES or ratio_high_conf < cls.MIN_RATIO_GOOD_FRAMES:
                    index_indicator.bad_indices.append(idx)
                else:
                    index_indicator.interpolatable_indices.append(idx)
            else:
                index_indicator.perfect_indices.append(idx)

        return index_indicator

    @classmethod
    def examine_data_quality(cls, df):
        confidence = df[cls.CONFIDENCE].values
        success = df[cls.SUCCESS].values

        n_rows = df.shape[0]
        ratio_high_conf = (confidence >= cls.CONFIDENCE_THRESHOLD).sum() / n_rows
        ratio_successful = (success == cls.SUCCESS_INDICATOR).sum() / n_rows
        return ratio_successful, ratio_high_conf

    @classmethod
    def interpolate_single_slice(cls, df, x_cols):
        """
        :param df: openface data
        :param x_cols: observational data columns
        :return: openface data with interpolated values
        """
        # iterate over X_cols and set cells with bad values to NaN
        for x in x_cols:
            df.loc[(df[cls.SUCCESS] != 1) | (df[cls.CONFIDENCE] != 1), x] = np.NaN

        # interpolate
        df[x_cols] = df[x_cols].interpolate(method="linear")

        # drop rows that couldn't be interpolated
        df = df.dropna()

        return df
