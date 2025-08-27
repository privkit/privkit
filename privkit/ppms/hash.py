from typing import List

from privkit.data import TabularData
from privkit.ppms import PPM
from privkit.utils import constants


class Hash(PPM):
    """
    Hash class to apply the hash mechanism from Python built-in functions

    References
    ----------
    https://docs.python.org/3/library/functions.html#hash
    """
    PPM_ID = "hash"
    PPM_NAME = "Hash"
    PPM_INFO = "The Hash mechanism returns the hash value of data."
    PPM_REF = "https://docs.python.org/3/library/functions.html#hash"
    DATA_TYPE_ID = [TabularData.DATA_TYPE_ID]
    METRIC_ID = []

    def __init__(self):
        """
        Initializes the Hash mechanism
        """
        super().__init__()

    def execute(self, tabular_data: TabularData, features: List = []):
        """
        Executes the Hash mechanism to the data given as parameter

        :param privkit.TabularData tabular_data: tabular data where the hash should be executed
        :param List features: list of feature(s) to hash. If no feature is selected, try all features.
        :return: tabular data with obfuscated data
        """
        try:
            if len(features) == 0:
                featured_columns = tabular_data.data.columns
            else:
                featured_columns = tabular_data.data[features]

            for column in featured_columns.columns:
                tabular_data.data[f"{constants.OBFUSCATED}_{column}"] = tabular_data.data[column].apply(hash)
        except Exception as e:
            raise Exception(f"Error while executing the {self.PPM_ID} mechanism: {e}")

        return tabular_data

    @staticmethod
    def get_obfuscated_data(data: object) -> int:
        """
        Returns obfuscated data

        :param object data: original data
        :return: obfuscated data through hash function
        """
        return hash(data)
