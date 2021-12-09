"""
2021 Simon Bing, ETHZ, MPI IS
"""
import six
import numpy as np
import healthgen.apps.global_parameters
from abc import ABCMeta, abstractmethod
from absl import flags
import logging
import wandb

FLAGS = flags.FLAGS
flags.DEFINE_string('X_test_path', '', 'Path to test data for evaluation.')
flags.DEFINE_string('y_test_path', '', 'Path to test labels for evaluation.')
flags.DEFINE_enum('eval_model', None, ['grud'], 'Generative model to use.')
flags.DEFINE_enum('eval_type', None, ['real', 'synth'], 'Whether the evaluation is on real'
                                                        'or synthetic data.')
flags.DEFINE_bool('bagging', False, 'Whether or not to perform bootstrap resampling of the training data.')
# TODO: add flag to supress saving of evaluator


@six.add_metaclass(ABCMeta)
class BaseEvaluator(object):
    def __init__(self, seed, eval_mode):
        self.seed = seed
        self.randomstate = np.random.RandomState(self.seed)
        self.eval_mode = eval_mode

    @abstractmethod
    def prepare_data(self, X_dict, y_dict):
        return NotImplementedError

    @abstractmethod
    def train_evaluator(self, X_train, X_val, y_train, y_val):
        raise NotImplementedError

    @abstractmethod
    def load_evaluator(self, saved_eval_model_path):
        raise NotImplementedError

    # @abstractmethod
    # def do_eval(self, X, y):
    #     raise NotImplementedError

    @abstractmethod
    def evaluate(self, X_test, y_test):
        raise NotImplementedError

    def get_data(self, X_dict=None, y_dict=None):
        """
        Returns:
            X_dict, y_dict: dictionaries of test data and labels
        """
        if X_dict is None and y_dict is None:
            X_dict = np.load(FLAGS.X_test_path)
            y_dict = np.load(FLAGS.y_test_path)

        X_train, X_val, X_test, y_train, y_val, y_test = self.prepare_data(X_dict, y_dict)

        if FLAGS.bagging:
            N_train = len(X_train)
            # Fix seed for reproducibility
            np.random.seed(self.seed)
            resample_idxs = np.random.choice(N_train, size=N_train, replace=True)
            X_train = X_train[resample_idxs, ...]
            y_train = y_train[resample_idxs]

            logging.info('Resampled training data for bootstrapping!')

        return X_train, X_val, X_test, y_train, y_val, y_test
