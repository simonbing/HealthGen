"""
2021 Simon Bing, ETHZ, MPI IS

Globally shared parameters. Local parameters can be defined in the respective modules.
"""
# from argparse import ArgumentParser
from absl import flags

FLAGS = flags.FLAGS
# Shared training parameters
flags.DEFINE_integer('seed', 0, 'Seed for random number generator.')
flags.DEFINE_string('out_path', '', 'Base directory where to save all outputs.')
flags.DEFINE_string('group', 'General', 'Group for wandb logging.')
flags.DEFINE_string('subgroup', 'General', 'Subroup for wandb logging of multiple seeds.')
flags.DEFINE_string('run_name', None, 'Run name for wandb logging.')
flags.DEFINE_bool('debug', False, 'Flag for additional debugging logging.')

# Shared generative model parameters
