"""
2021 Simon Bing, ETHZ, MPI IS
"""
import numpy as np
from healthgen.data_access.preprocessing.base_processor import BaseProcessor

class MissingnessDeltaT(BaseProcessor):
    def __init__(self):
        super().__init__()


    def _get_missingness_mask(self, x):
        """
        Returns:
            m_mask: Mask of missing data of x (0: missing, 1: present)
        """
        m_mask = (~np.isnan(x)).astype(int)
        return m_mask


    def _get_delta_t(self, mask):
        """
        Args:
            mask: mask of missing values
        Returns:
            delta_t: Array with times since last observation
        """
        N, features, time_len = mask.shape

        delta_t = np.empty(mask.shape)
        for t in range(time_len):
            if not t: # only for first step
                delta_t[:,:,t] = 0
            else:
                for n in range(N):
                    for d in range(features):
                        if mask[n,d,t-1] == 0: # missing value at previous step
                            delta_t[n,d,t] = 1 + delta_t[n,d,t-1]
                        else: # no missing value in previous step
                            delta_t[n,d,t] = 1

        return delta_t


    def transform(self, x):
        """
        Creates the missingness mask and the time delta array from the input data.

        Args:
            x: array of input time series [N_patients, features, time_len]

        Returns:
            X_dict_tf: contains input data as well as missingness masks and delta_t array.
        """
        missingness_mask = self._get_missingness_mask(x)

        delta_t = self._get_delta_t(missingness_mask)

        X_dict_tf = {
            'X': np.nan_to_num(x),
            'm': missingness_mask,
            'delta_t': delta_t
        }

        return X_dict_tf
