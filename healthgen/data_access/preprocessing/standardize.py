"""
2021 Simon Bing, ETHZ, MPI IS
"""
import numpy as np
from sklearn.preprocessing import StandardScaler
from healthgen.data_access.preprocessing.base_processor import BaseProcessor

class Standardize(BaseProcessor):
    def __init__(self):
        super().__init__()


    def transform(self, x):
        """
        Standardize input features.

        Args:
            x: dict with X_train, X_val, X_test arrays of dim.: [N, features, time_len]

        Returns:
            X_dict_tf: dict of transformed inputs.
        """
        # Check if missing values mask exists and replace zeros with nans
        if 'm_train' in x.keys():
            X_train = np.where(x['m_train'], x['X_train'], np.nan)
            X_val = np.where(x['m_val'], x['X_val'], np.nan)
            X_test = np.where(x['m_test'], x['X_test'], np.nan)
        else:
            X_train = x['X_train']
            X_val = x['X_val']
            X_test = x['X_test']

        n_features = X_train.shape[1]
        time_len = X_train.shape[2]

        # Flatten time dimension
        X_train_reshape = np.reshape(X_train.transpose(0,2,1), (-1, n_features))
        X_val_reshape = np.reshape(X_val.transpose(0, 2, 1), (-1, n_features))
        X_test_reshape = np.reshape(X_test.transpose(0, 2, 1), (-1, n_features))

        # Fit scaler
        scaler = StandardScaler()
        scaler.fit(X_train_reshape)

        # Transform data
        X_train_reshape_tf = scaler.transform(X_train_reshape)
        X_val_reshape_tf = scaler.transform(X_val_reshape)
        X_test_reshape_tf = scaler.transform(X_test_reshape)

        # Recover original dimensions
        X_train_tf = np.reshape(X_train_reshape_tf, X_train.transpose(0,2,1).shape).transpose(0,2,1)
        X_val_tf = np.reshape(X_val_reshape_tf, X_val.transpose(0, 2, 1).shape).transpose(0,2,1)
        X_test_tf = np.reshape(X_test_reshape_tf, X_test.transpose(0, 2, 1).shape).transpose(0,2,1)

        X_dict_tf = x.copy()
        X_dict_tf['X_train'] = np.nan_to_num(X_train_tf)
        X_dict_tf['X_val'] = np.nan_to_num(X_val_tf)
        X_dict_tf['X_test'] = np.nan_to_num(X_test_tf)


        return X_dict_tf