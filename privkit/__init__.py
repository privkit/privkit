"""
Privkit
========

Privkit is a privacy toolkit that provides methods for privacy analysis. It includes different data types, privacy-preserving mechanisms, attacks, and metrics. The current version is focused on location data and facial data. The Python Package is designed in a modular manner and can be easily extended to include new mechanisms. Privkit can be used to process data, configure privacy-preserving mechanisms, apply attacks, and also evaluate the privacy/utility trade-off through suitable metrics.

See privkit.fc.up.pt for a complete documentation.
"""
# This ignores the Info and Warning logs from Tensorflow
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

name = 'privkit'

__version__ = '0.5.0'

from .attacks import *
from .data import *
from .datasets import *
from .metrics import *
from .ppms import *
from .utils import *

__all__ = [
    "attacks",
    "data",
    "datasets",
    "metrics",
    "ppms",
    "utils"
]