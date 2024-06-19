from abc import ABC, abstractmethod


class DataType(ABC):
    """
    DataType is an abstract class for a generic type of data. Defines a series of methods common to all data types.
    Provides basic functions to load, process, and save data.
    Requires the definition of the DATA_TYPE_ID, DATA_TYPE_NAME, and DATA_TYPE_INFO.
    """
    @property
    def DATA_TYPE_ID(self) -> str:
        """Identifier of the data type"""
        raise NotImplementedError

    @property
    def DATA_TYPE_NAME(self) -> str:
        """Name of the data type"""
        raise NotImplementedError

    @property
    def DATA_TYPE_INFO(self) -> str:
        """Information of the data type, specifically the format of the files to be read"""
        raise NotImplementedError

    @abstractmethod
    def load_data(self, *args):
        """Loads data. This is specific to the data type"""
        pass

    @abstractmethod
    def process_data(self, *args):
        """Performs data processing or returns data processing methods. This is specific to the data type"""
        pass

    @abstractmethod
    def save_data(self, *args):
        """Saves data to a file. This is specific to the data type"""
        pass
