import numpy as np
import pandas as pd

from pathlib import Path
from pandas.core.frame import DataFrame
from sklearn.model_selection import train_test_split

from privkit.data import DataType
from privkit.utils import (
    constants,
    io_utils,
    dev_utils as du
)


class TabularData(DataType):
    """
    TabularData is a privkit.DataType to handle tabular data. Tabular data is stored as a Pandas DataFrame.
    """
    DATA_TYPE_ID = "tabular_data"
    DATA_TYPE_NAME = "Tabular Data"
    DATA_TYPE_INFO = "Tabular data can be imported through a Pandas dataframe or a read by a delimited file, " \
                     "a file-like object or an object. "

    def __init__(self, id_name: str = None):
        super().__init__()
        self.id_name = id_name or self.DATA_TYPE_ID
        """Identifier name of this tabular data instance"""
        self.original_data = None
        """Original tabular data is stored as a pd.DataFrame"""
        self.data = None
        """Tabular data is stored as a pd.DataFrame"""

    def load_data(self, data_to_load: DataFrame or str or Path or object,
                  save: bool = False,
                  **kwargs):
        """
        Loads tabular data from a pd.DataFrame or a file that can be read by a Pandas read() method.
        The definition of the parameters came from the parameters of Pandas.

        :param DataFrame or str or Path or object data_to_load: either a Pandas Dataframe, a path to a file (str or Path) or any object that can be read from pandas read() methods.
        :param bool save: if `True`, data is saved to a file. The default is `False`.
        :param kwargs: parameters for the Pandas read methods.
        """
        if isinstance(data_to_load, pd.DataFrame):
            tdf = data_to_load
        else:
            tdf = io_utils.read_dataframe(data_to_load, unique=False, **(kwargs or {}))

        try:
            self.data = tdf
            self.original_data = tdf

            filename = 'original_{}'.format(self.id_name)
            if save:
                self.save_data(filename=filename)  # To save the original data

            if save:
                self.save_data()  # To save the processed data
        except Exception as e:
            du.log("Exception while saving data: {}".format(self.DATA_ID, e))


    def save_data(self, filepath: str = constants.data_folder, filename: str = None, extension: str = 'pkl'):
        """
        Saves data to a file.

        :param str filepath: path where data should be saved.
        :param str filename: name of the file to be saved.
        :param str extension: extension of the format of how the file should be saved. The default value is `'pkl'`.
        """
        if filename is None:
            filename = self.id_name
        io_utils.write_dataframe(self.data, filepath, filename, extension)

    def process_data(self):
        """
        Performs tabular data processing.
        """
        du.log("Starting data processing.")
        du.log("Data processing done.")

    # ========================= ML-related methods =========================

    def divide_data(self, test_size: float = 0.2):
        """
        Divides data into train and test
        :param test_size: size of test data
        """
        du.log(f"Dividing data into train and test data with a {test_size} ratio for testing.")
        _, _, train_indexes, test_indexes = train_test_split(self.data, self.data.index, test_size=test_size)
        self.train_indexes = np.sort(train_indexes)
        self.test_indexes = np.sort(test_indexes)

    def get_train_data(self):
        """
        Returns train data
        :return: train data
        """
        if not hasattr(self, "train_indexes"):
            du.warn("Data should be divided first through divide_data method.")
            self.divide_data()
        return self.data.loc[self.train_indexes]

    def get_test_data(self):
        """
        Returns test data
        :return: test data
        """
        if not hasattr(self, "test_indexes"):
            du.warn("Data should be divided first through divide_data method.")
            self.divide_data()
        return self.data.loc[self.test_indexes]

    # ========================= Statistics methods =========================

    def print_data_summary(self, dataset_name: str = None):
        """
        Prints data summary

        :param str dataset_name: dataset name (optional). The default value is `None`.
        """
        print("\n=======================================")
        if dataset_name is None:
            print("{} contains:".format(self.DATA_TYPE_NAME))
        else:
            print("Dataset {} contains:".format(dataset_name))

        print("#records: {}".format(len(self.data)))

        print("=======================================\n")
