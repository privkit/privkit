import glob
import tarfile
import requests
import numpy as np
import pandas as pd

from typing import List
from pathlib import Path

from privkit.data import LocationData
from privkit.datasets import Dataset
from privkit.utils import constants, dev_utils as du


class CabspottingDataset(Dataset):
    """
    Class to handle the Cabspotting dataset.
    """
    DATASET_ID = 'cabspotting'
    DATASET_NAME = 'Cabspotting Dataset'
    DATASET_INFO = ("Cabspotting dataset is a dataset of taxi trajectories over the city of San Francisco, California, "
                    "USA. The trajectories belong to 536 taxis and were collected over a period of 30 days, "
                    "containing not only the GPS position and timestamp, but also whether the cab had a costumer at "
                    "each time.")
    DATA_TYPE_ID = [LocationData.DATA_TYPE_ID]

    def __init__(self):
        super().__init__()
        self.data = LocationData(self.DATASET_ID)

    def download(self, url: str, savepath: str = constants.datasets_folder):
        """
        Downloads Cabspotting dataset from IEEE Data Port if the authorized URL is provided.
        To proceed with the download through the Privkit, the user is required to login in the website https://ieee-dataport.org/open-access/crawdad-epflmobility,
        click on the "cabspottingdata.tar.gz" button and copy the url.

        :param str url: url that results from clicking on the "cabspottingdata.tar.gz" button after logging in the website https://ieee-dataport.org/open-access/crawdad-epflmobility.
        :param str savepath: path where dataset should be saved. The default is `constants.datasets_folder`.
        """
        try:
            du.log(f"Downloading the {self.DATASET_ID} dataset")
            response = requests.get(url, stream=True)
            file = tarfile.open(fileobj=response.raw, mode="r|gz")
            Path(savepath).parent.mkdir(parents=True, exist_ok=True)
            file.extractall(path=savepath)
            du.log(f"The download of {self.DATASET_ID} dataset was saved at {savepath}")
            self.load_from_files()
        except Exception as e:
            raise Exception(f"Error while downloading the {self.DATASET_ID} dataset: {e}")

    def load_dataset(self):
        """Loads cabspotting dataset"""
        try:
            du.log(f"Loading the {self.DATASET_ID} dataset")
            load_path = constants.data_folder + self.DATASET_ID + ".pkl"
            self.data.load_data(load_path)
            du.log("Dataset loaded from {}".format(load_path))
        except FileNotFoundError:
            self.load_from_files()

    def load_from_files(self):
        try:
            load_path = constants.datasets_folder + "cabspottingdata/"
            du.log("Loading dataset from files at {}".format(load_path))
            user_id = 0
            taxi_dataframes = []
            for file in glob.glob(load_path + "new_*.txt"):
                f = open(file, "r")
                latitudes, longitudes, occupancies, datetime = [], [], [], []
                for line in f:
                    fields = line.replace("\n", "").split(" ")
                    latitudes.append(np.float64(fields[0]))
                    longitudes.append(np.float64(fields[1]))
                    occupancies.append(np.int64(fields[2]))
                    datetime.append(np.datetime64(int(fields[3]), "s"))

                taxi_dataframes.append(pd.DataFrame({constants.LATITUDE: latitudes,
                                                     constants.LONGITUDE: longitudes,
                                                     constants.DATETIME: datetime,
                                                     constants.OCCUPANCY: occupancies,
                                                     constants.UID: [user_id] * len(latitudes),
                                                     constants.TID: self._convert_occupancy_to_trajectory(
                                                         occupancies)}))
                user_id += 1
                f.close()
            dataframe = pd.concat(taxi_dataframes, ignore_index=True)
            self.data.load_data(dataframe)
            self.save_dataset()
        except FileNotFoundError:
            raise FileNotFoundError("Download the dataset files first")

    @staticmethod
    def _convert_occupancy_to_trajectory(occupancies: List) -> List:
        """
        Computes trajectories based on the taxi occupancy

        :param List occupancies: list of taxi occupancy, where 1 is occupied and 0 is free
        :return: list of trajectory identifiers
        """
        start_idx = 0
        trajectory_id = 0
        trajectory_length = len(occupancies)
        trajectory_occupancy = occupancies[start_idx]
        tid = [None] * trajectory_length
        for idx, current_occupancy in enumerate(occupancies):
            if current_occupancy != trajectory_occupancy:
                tid[start_idx:idx] = [trajectory_id] * (idx - start_idx)
                start_idx = idx
                trajectory_occupancy = current_occupancy
                trajectory_id += 1
            if idx == trajectory_length - 1:
                tid[start_idx:trajectory_length] = [trajectory_id] * (trajectory_length - start_idx)
        return tid

    def filter_by_occupancy(self, occupancy: int = 1):
        """
        Filters trajectories by occupancy.

        :param int occupancy: 1 is occupied and 0 is free. The default value is `1`, that is, it filters trajectories with occupancy.
        """
        self.data.data = self.data.data[self.data.data[constants.OCCUPANCY] == occupancy]

    def process_dataset(self):
        pass

    def save_dataset(self):
        """Saves dataset"""
        self.data.save_data(filename=self.DATASET_ID)
