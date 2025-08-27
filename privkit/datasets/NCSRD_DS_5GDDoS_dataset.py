import os
import io
import glob
import requests
import numpy as np
import pandas as pd

from pathlib import Path
from zipfile import ZipFile

from privkit.data import TabularData
from privkit.datasets import Dataset
from privkit.ppms import Hash, Laplace
from privkit.utils import constants, dev_utils as du

class NCSRD_DS_5GDDoS_Dataset(Dataset):
    """
    Class to handle the NCSRD-DS-5GDDoS dataset.
    """
    DATASET_ID = 'NCSRD_DS_5GDDoS'
    DATASET_NAME = 'NCSRD-DS-5GDDoS Dataset'
    DATASET_INFO = ("NCSRD-DS-5GDDoS dataset is a real mobility dataset collected in a period of over three year from GPS "
                    "devices. The dataset contains data from 182 users, 17621 trajectories and roughly 25 million "
                    "reports.")
    DATA_TYPE_ID = [TabularData.DATA_TYPE_ID]

    def __init__(self):
        super().__init__()
        self.data = TabularData(self.DATASET_ID)

    @staticmethod
    def _get_url_by_filename(filename: str = None) -> str:
        """
        Gets url from Zenodo (https://zenodo.org/records/13900057) given the filename

        :param str filename: filename from the available files in Zenodo (https://zenodo.org/records/13900057)
        """
        available_files = ["amari_ue_data_classic_tabular.csv",
                           "amari_ue_data_merged_with_attack_number.csv",
                           "amari_ue_data_mini_tabular.csv",
                           "enb_counters_data_classic_tabular.csv",
                           "enb_counters_data_mini_tabular.csv",
                           "mme_counters.csv"
                        ]
        if filename in available_files:
            return f"https://zenodo.org/records/13900057/files/{filename}"
        else:
            return "https://zenodo.org/records/13900057/files/amari_ue_data_merged_with_attack_number.csv"

    def download(self, savepath: str = constants.datasets_folder, filename: str = "NCSRD_DS_5GDDoS"):
        """
        Downloads NCSRD-DS-5GDDoS dataset.

        :param str savepath: path where dataset should be saved. The default is `constants.datasets_folder`.
        :param str filename: filename from the available files in Zenodo (https://zenodo.org/records/13900057)
        """
        try:
            du.log("Downloading the {} dataset".format(self.DATASET_ID))
            url = self._get_url_by_filename(filename)
            data = pd.read_csv(url)
            filepath = "{}{}.csv".format(savepath, self.DATASET_ID)
            Path(filepath).parent.mkdir(parents=True, exist_ok=True)
            data.to_csv(filepath)
            du.log("The download of {} dataset was saved at {}".format(self.DATASET_ID, filepath))
        except Exception as e:
            du.log("Error while downloading the {} dataset: {}".format(self.DATASET_ID, e))

    def load_dataset(self):
        """Loads NCSRD-DS-5GDDoS dataset"""
        try:
            du.log(f"Loading the {self.DATASET_ID} dataset")
            load_path = constants.datasets_folder + self.DATASET_ID + ".csv"
            self.data.load_data(load_path)
            du.log("Dataset loaded from {}".format(load_path))
        except FileNotFoundError:
            self.download()
            load_path = constants.datasets_folder + self.DATASET_ID + ".csv"
            du.log("Loading dataset from files at {}".format(load_path))
            self.data.load_data(load_path)
            self.save_dataset()

    def anonymize_dataset(self, epsilon: float = 0.01, sensitivity: float = 1):
        """Apply anonymization to sensitive fields

        :param float epsilon: privacy parameter for the Laplace mechanism. Default value = 0.01.
        :param float sensitivity: sensitivity parameter. Default value = 1.
        """

        anonymized = self.data.data.copy()

        # Hash device identifier
        if 'imeisv' in anonymized:
            original_imeisv = anonymized['imeisv']

            anonymized['_original_device_id'] = original_imeisv  # Keep for buffering

            anonymized['imeisv'] = anonymized['imeisv'].apply(lambda x: f"{Hash.get_obfuscated_data(x) % 10000}")

        # Mask IP addresses
        for field in ['ip', 'bearer_0_ip', 'bearer_1_ip']:
            if field in anonymized:
                anonymized[field] = "xxx.xxx.xxx.xxx"

        # Hash other identifiers
        for field in ['amf_ue_id', '5g_tmsi']:
            if field in anonymized:
                anonymized[field] = anonymized[field].apply(lambda x: f"{Hash.get_obfuscated_data(x) % 10000}")

        # Obfuscate sensitive features with Laplace
        for field in ['dl_bitrate', 'ul_bitrate']:
            if field in anonymized:
                results = anonymized[field].apply(Laplace(epsilon=epsilon, sensitivity=sensitivity).get_obfuscated_point)
                obf_data = pd.DataFrame(results.tolist(), index=anonymized.index)
                anonymized[[f"{field}", f"{constants.QUALITY_LOSS}_{field}"]] = obf_data

        return anonymized

    def process_dataset(self, anonymize: bool = True):
        """Processes dataset"""
        if anonymize:
            self.anonymize_dataset()

    def save_dataset(self):
        """Saves dataset"""
        self.data.save_data(filename=self.DATASET_ID, extension='csv')
