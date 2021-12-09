"""
2021 Simon Bing, ETHZ, MPI IS
"""
from healthgen.data_access import MimicLoader
import six
import logging
import torch
import wandb
from abc import ABCMeta, abstractmethod
from absl import flags, app

FLAGS = flags.FLAGS
flags.DEFINE_enum('dataset', 'mimic', ['mimic'], 'Which data to use for training and evaluation.')
flags.DEFINE_enum('evaluation', 'grud', ['grud'], 'Evaluation method to use.')

@six.add_metaclass(ABCMeta)
class BaseApplication(object):
    def __init__(self):
        self.data_loader = self._get_data_loader()

    def _get_data_loader(self):
        if FLAGS.dataset == 'mimic':
            return MimicLoader(seed=FLAGS.seed, task_name=FLAGS.task,
                               processed_features_path=FLAGS.processed_features_path,
                               processed_labels_path=FLAGS.processed_labels_path,
                               extracted_intermed_features_path=FLAGS.extracted_intermed_features_path,
                               extracted_intermed_labels_path=FLAGS.extracted_intermed_labels_path,
                               extracted_intermed_static_path=FLAGS.extracted_intermed_static_path,
                               vitals_path=FLAGS.input_vitals, outcomes_path=FLAGS.input_outcomes,
                               static_path=FLAGS.input_static,
                               N_patients=FLAGS.N_patients, time_resolution=FLAGS.time_resolution,
                               time_length=FLAGS.time_length, prediction_gap=FLAGS.prediction_gap,
                               prediction_window=FLAGS.prediction_window)

    @abstractmethod
    def run(self):
        raise NotImplementedError
