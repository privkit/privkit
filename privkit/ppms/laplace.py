import numpy as np
import pandas as pd

from typing import List

from privkit.data import TabularData
from privkit.ppms import PPM
from privkit.metrics import QualityLoss
from privkit.utils import constants


class Laplace(PPM):
    """
    Laplace class to apply the mechanism

    References
    ----------
    Dwork, C., McSherry, F., Nissim, K., & Smith, A. (2006). Calibrating noise to sensitivity in private data analysis.
    In Theory of Cryptography: Third Theory of Cryptography Conference, TCC 2006, New York, NY, USA, March 4-7, 2006.
    Proceedings 3 (pp. 265-284). Springer Berlin Heidelberg.
    """
    PPM_ID = "laplace"
    PPM_NAME = "Laplace"
    PPM_INFO = "The Laplace mechanism consists of adding Laplacian noise to satisfy the Differential Privacy notion. The noise is added according to the Laplacian distribution and depends on a privacy parameter epsilon and on a sensitivity parameter, both defined by the user."
    PPM_REF = "Dwork, C., McSherry, F., Nissim, K., & Smith, A. (2006). Calibrating noise to sensitivity in private data analysis. In Theory of Cryptography: Third Theory of Cryptography Conference, TCC 2006, New York, NY, USA, March 4-7, 2006. Proceedings 3 (pp. 265-284). Springer Berlin Heidelberg."
    DATA_TYPE_ID = [TabularData.DATA_TYPE_ID]
    METRIC_ID = [QualityLoss.METRIC_ID]

    def __init__(self, epsilon: float, sensitivity: float = 1):
        """
        Initializes the Laplace mechanism by defining the privacy parameter epsilon and the sensitivity

        :param float epsilon: privacy parameter
        :param float sensitivity: sensitivity parameter. Default value = 1.
        """
        super().__init__()
        self.epsilon = epsilon
        self.sensitivity = sensitivity

    def execute(self, tabular_data: TabularData, features: List = []):
        """
        Executes the Laplace mechanism to the data given as parameter

        :param privkit.TabularData tabular_data: tabular data where the laplace should be executed
        :param List features: feature(s) to apply Laplacian noise. If no feature is selected, try all float features.
        :return: tabular data with obfuscated data and the quality loss metric
        """
        try:
            if len(features) == 0:
                featured_columns = tabular_data.data.select_dtypes(include=[float])
            else:
                featured_columns = tabular_data.data[features]

            for column in featured_columns.columns:
                results = tabular_data.data[column].apply(self.get_obfuscated_point)

                results_df = pd.DataFrame(results.tolist(), index=tabular_data.data.index)

                new_column_names = [f"{constants.OBFUSCATED}_{column}", f"{constants.QUALITY_LOSS}_{column}"]

                tabular_data.data[new_column_names] = results_df
        except Exception as e:
            raise Exception(f"Error while executing the {self.PPM_ID} mechanism: {e}")

        return tabular_data

    def get_obfuscated_point(self, data: float) -> [float, float]:
        """
        Returns an obfuscated single point

        :param float data: original data
        :return: obfuscated point and distance between original and obfuscation point
        """
        scale = self.sensitivity / self.epsilon
        noise = np.random.laplace(loc=0, scale=scale)
        obfuscated_point = data + noise

        return obfuscated_point, abs(noise)
