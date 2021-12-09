"""
2021 Simon Bing, ETHZ, MPI IS
"""
import numpy as np
from absl import flags

class BaseProcessor(object):
    def __init__(self):
        self.name = None

    def transform(self, x):
        raise NotImplementedError