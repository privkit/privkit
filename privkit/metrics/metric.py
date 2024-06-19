import numpy as np

from typing import List
from abc import ABC, abstractmethod


class Metric(ABC):
    """
    Metric is an abstract class for a generic metric. Defines a series of methods common to all metrics.
    Provides a general method to execute, to plot and to print statistics of the metric.
    Requires the definition of a METRIC_ID, METRIC_NAME, METRIC_INFO, METRIC_REF, and DATA_TYPE_ID (of the data types
    this metric can be applied to).
    """
    @property
    def METRIC_ID(self) -> str:
        """Identifier of the metric"""
        raise NotImplementedError

    @property
    def METRIC_NAME(self) -> str:
        """Name of the metric"""
        raise NotImplementedError

    @property
    def METRIC_INFO(self) -> str:
        """Information about the metric and how it works"""
        raise NotImplementedError

    @property
    def DATA_TYPE_ID(self) -> List[str]:
        """Identifiers of the data types that the metric is applied to"""
        raise NotImplementedError

    def __init__(self):
        self.values = []

    @abstractmethod
    def execute(self, *args):
        """Executes the metric. This is specific to the metric."""
        pass

    @abstractmethod
    def plot(self, *args):
        """Plots the results of the metric. This is specific to the metric."""
        pass

    def _print_statistics(self):
        """Prints statistics of the metric results."""
        print("Statistics of {}:\n".format(self.METRIC_NAME))
        print("Median: {}\n".format(np.median(self.values)))
        print("Mean: {}\n".format(np.mean(self.values)))
