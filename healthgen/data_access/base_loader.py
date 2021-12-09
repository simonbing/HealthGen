"""
2021 Simon Bing, ETHZ, MPI IS
"""
import six
import numpy as np
import healthgen.apps.global_parameters
from abc import ABCMeta, abstractmethod
from absl import flags
from sklearn.model_selection import train_test_split
import logging

FLAGS = flags.FLAGS
flags.DEFINE_string('extracted_intermed_features_path', None, 'Path to previously extracted '
                                                              'and saved intermediate X.')
flags.DEFINE_string('extracted_intermed_labels_path', None, 'Path to previously extracted '
                                                            'and saved intermediate y.')
flags.DEFINE_string('extracted_intermed_static_path', None, 'Path to previously extracted '
                                                            'and saved intermediate c.')
flags.DEFINE_string('processed_features_path', None, 'Path to previously processed and saved X.')
flags.DEFINE_string('processed_labels_path', None, 'Path to previously processed and saved y.')
flags.DEFINE_string('task', None, 'Prediction task')
flags.DEFINE_float('val_set_fraction', 0.15, 'Fraction of data for validation.')
flags.DEFINE_float('test_set_fraction', 0.15, 'Fraction of data for testing.')
flags.DEFINE_bool('save_intermed_data', False, 'Whether or not to save the intermediate extracted data.')
flags.DEFINE_bool('save_processed_data', False, 'Whether or not to save the processed data.')
flags.DEFINE_string('processed_output_path', '', 'Path to save processed data set.')
flags.DEFINE_list('features', None, 'List of features to keep')

@six.add_metaclass(ABCMeta)
class BaseLoader(object):
    def __init__(self, seed, task_name, extracted_intermed_features_path,
                 extracted_intermed_labels_path, extracted_intermed_static_path,
                 processed_features_path, processed_labels_path):
        self.randomstate = np.random.RandomState(seed)
        self.task_name = task_name
        self.extracted_intermed_features_path = extracted_intermed_features_path
        self.extracted_intermed_labels_path = extracted_intermed_labels_path
        self.extracted_intermed_static_path = extracted_intermed_static_path
        self.processed_features_path = processed_features_path
        self.processed_labels_path = processed_labels_path

    @abstractmethod
    def get_timeseries(self):
        """
        Loads raw data from some source and returns formatted patient time series.
        Returns:
            patients_ts: [N_patients, features, time_len]
            feature_names: [features]
        """
        raise NotImplementedError

    @abstractmethod
    def get_outcomes(self):
        """
        Loads the labels for various outcomes/prediction tasks.
        Returns:
            outcomes: [N_patients, outcomes]
        """
        raise NotImplementedError

    @abstractmethod
    def get_static_data(self):
        """
        Loads static data for patients.
        Returns:
            static_data: [N_patients, static_data]
            feature_names: [static_data]
        """
        raise NotImplementedError

    @abstractmethod
    def select_task(self, outcomes, task):
        """
        Select task and return respective labels.
        """
        raise NotImplementedError

    @abstractmethod
    def get_intermed_preprocessors(self):
        raise NotImplementedError

    @abstractmethod
    def get_input_preprocessors(self):
        raise NotImplementedError

    @abstractmethod
    def select_features(self, X, features):
        raise NotImplementedError

    @abstractmethod
    def save_intermed_data(self, X, y, out_path):
        raise NotImplementedError

    @abstractmethod
    def save_processed_data(self, X, y, out_path):
        raise NotImplementedError

    # @staticmethod
    # def _train_val_test_split(X, y):


    def split_dataset(self, X_in, y, c, val_fraction, test_fraction):
        """
        Splits dataset into stratified train/val/test.

        Args:
            X_in: dict with where entries are asuumed to share the first dimension.
               Includes at least 'X' and 'feature_names' keys.

        Returns:
            X_dict: dictionary with X_train, X_val, X_test
            y_dict: dictionary with y_train, y_val, y_test
        """
        key_list = []
        input_list = []
        for key, value in X_in.items():
            if key != 'feature_names':
                key_list.append(key)
                input_list.append(value)

        X = np.stack(input_list, axis=1)
        # TODO: MOVE THIS TO UTIL FUNCTION
        train_fraction = 1.0 - val_fraction - test_fraction

        y = np.concatenate((np.expand_dims(y,1), c['c']), axis=1)

        if len(y.shape) == 1: # stratify only if we have a single task
            y_strat = y
        else:
            y_strat = y[:,0]

        X_train, X_intermed, y_train, y_intermed = train_test_split(X, y,
                                                                    test_size=1 - train_fraction,
                                                                    random_state=self.randomstate,
                                                                    stratify=y_strat)
        if y_strat is not None:
            y_intermed_strat = y_intermed[:,0]
        else:
            y_intermed_strat = None
        X_val, X_test, y_val, y_test = train_test_split(X_intermed, y_intermed,
                                                        test_size=test_fraction / (test_fraction + val_fraction),
                                                        random_state=self.randomstate,
                                                        stratify=y_intermed_strat)

        X_dict = {}
        for idx, key in enumerate(key_list):
            X_dict[F'{key}_train'] = X_train[:,idx,...]
            X_dict[F'{key}_val'] = X_val[:, idx, ...]
            X_dict[F'{key}_test'] = X_test[:, idx, ...]
        X_dict['feature_names'] = X_in['feature_names']

        y_dict = {
            'y_train': y_train[:,0],
            'y_val': y_val[:,0],
            'y_test': y_test[:,0]
        }

        c_dict = {
            'c_train': y_train[:,1:],
            'c_val': y_val[:,1:],
            'c_test': y_test[:,1:],
            'feature_names': c['feature_names']
        }

        y_dict.update(c_dict)

        return X_dict, y_dict


    def preprocess_variables(self, x, preprocessors):
        """
        Args:
            x: variables to preprocess. may be an array or dict, depending on
               preprocessors that are called
            preprocessors: ordered list of preprocessing methods

        Returns:
            x: preprocessed variables
        """
        # Preprocess with steps defined in preprocessors
        for process_step in preprocessors:
            x = process_step.transform(x)

        return x


    def get_data(self):
        """
        The main method of the loader. Grabs raw (extracted) data and returns
        split data set ready for downstream training.

        Returns:
            X_dict_tf: dictionary of preprocessed input features for training
            y_dict: dictionary of labels for training
        """
        if self.processed_features_path is not None and \
           self.processed_labels_path is not None:
            X_dict_tf = np.load(self.processed_features_path)
            y_dict = np.load(self.processed_labels_path)
        else:
            if self.extracted_intermed_features_path is not None and \
               self.extracted_intermed_labels_path is not None and \
               self.extracted_intermed_static_path is not None:
                # dict with at least 'X': [N_patients, features, time_len]
                # and 'feature_names': [features]
                patients_intermed = np.load(self.extracted_intermed_features_path)
                # array: [N_patients, labels]
                outcomes_intermed = np.load(self.extracted_intermed_labels_path)
                static_intermed = np.load(self.extracted_intermed_static_path)
                if self.N_patients is None:
                    self.N_patients = len(patients_intermed['X'])
            else: # Extract patients and outcomes (ALL patients, features and labels)
                # array: [N_patients, features, time_len], array: [features]
                patients, feature_names = self.get_timeseries()
                # array: [N_patients, labels]
                outcomes_intermed = self.get_outcomes()
                static_intermed = self.get_static_data()

                # Preprocessing that is independent of split of dataset
                # dict with at least 'X': [N_patients, features, time_len]
                patients_intermed = self.preprocess_variables(patients, self.get_intermed_preprocessors())
                patients_intermed['feature_names'] = feature_names

                if FLAGS.save_intermed_data:
                    self.save_intermed_data(patients_intermed, outcomes_intermed,
                                            static_intermed, FLAGS.processed_output_path)
                    logging.info('Saved intermediate data!')

            patients_truncated = self.select_features(patients_intermed, FLAGS.features)
            y = self.select_task(outcomes_intermed, FLAGS.task)
            # y = outcomes # EXPERIMENTAL: skipping task extraction and doing it later

            X_dict, y_dict = self.split_dataset(patients_truncated, y, static_intermed, FLAGS.val_set_fraction, FLAGS.test_set_fraction)

            # Modular preprocessing of SPLIT data
            X_dict_tf = self.preprocess_variables(X_dict, self.get_input_preprocessors())

            if FLAGS.save_processed_data:
                self.save_processed_data(X_dict_tf, y_dict, FLAGS.processed_output_path)
                logging.info('Saved processed data!')

            # TODO: comment with expected shape of X and y everywhere

        return X_dict_tf, y_dict

