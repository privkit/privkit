from typing import List
from abc import ABC, abstractmethod


class Dataset(ABC):
    """
    Abstract class for a generic dataset. Defines a series of functions common to process different datasets.
    Provides basic functions to load, process, and save data.
    Requires the definition of a DATASET_ID, DATASET_NAME, DATASET_INFO, and DATA_TYPE_ID.
    """
    @property
    def DATASET_ID(self) -> str:
        """Identifier of the dataset"""
        raise NotImplementedError

    @property
    def DATASET_NAME(self) -> str:
        """Name of the dataset"""
        raise NotImplementedError

    @property
    def DATASET_INFO(self) -> str:
        """Information of the dataset, specifying the reference for the dataset (if it exists)"""
        raise NotImplementedError

    @property
    def DATA_TYPE_ID(self) -> List[str]:
        """Identifier of the data type of this dataset"""
        raise NotImplementedError

    def __init__(self):
        self.data = None

    @abstractmethod
    def load_dataset(self, *args):
        """Loads dataset. This is specific to the dataset"""
        pass

    @abstractmethod
    def process_dataset(self, *args):
        """Performs dataset processing or returns dataset processing methods. This is specific to the dataset"""
        pass

    @abstractmethod
    def save_dataset(self, *args):
        """Saves the dataset to a file. This is specific to the dataset"""
        pass
