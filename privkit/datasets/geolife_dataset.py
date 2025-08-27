import os
import io
import glob
import requests
import numpy as np
import pandas as pd

from zipfile import ZipFile

from privkit import LocationData
from privkit.datasets import Dataset
from privkit.utils import constants, io_utils, dev_utils as du


class GeolifeDataset(Dataset):
    """
    Class to handle the Geolife dataset.
    """
    DATASET_ID = 'geolife'
    DATASET_NAME = 'Geolife Dataset'
    DATASET_INFO = ("Geolife dataset is a real mobility dataset collected in a period of over three year from GPS "
                    "devices. The dataset contains data from 182 users, 17621 trajectories and roughly 25 million "
                    "reports.")
    DATA_TYPE_ID = [LocationData.DATA_TYPE_ID]

    def __init__(self):
        super().__init__()
        self.data = LocationData(self.DATASET_ID)

    def download(self, savepath: str = constants.datasets_folder):
        """
        Downloads Geolife dataset.

        :param str savepath: path where dataset should be saved. The default is `constants.datasets_folder`.
        """
        try:
            du.log("Downloading the {} dataset".format(self.DATASET_ID))
            url = "https://download.microsoft.com/download/F/4/8/F4894AA5-FDBC-481E-9285-D5F8C4C4F039/Geolife%20Trajectories%201.3.zip"
            response = requests.get(url, stream=True)
            ZipFile(io.BytesIO(response.content)).extractall(savepath)
            os.rename(savepath + "Geolife Trajectories 1.3", savepath + self.DATASET_ID + "data")
            du.log("The download of {} dataset was saved at {}".format(self.DATASET_ID, savepath))
        except Exception as e:
            du.log("Error while downloading the {} dataset: {}".format(self.DATASET_ID, e))

    def load_dataset(self):
        """Loads Geolife dataset"""
        try:
            du.log(f"Loading the {self.DATASET_ID} dataset")
            load_path = constants.data_folder + self.DATASET_ID + ".pkl"
            self.data.load_data(load_path)
            du.log("Dataset loaded from {}".format(load_path))
        except FileNotFoundError:
            self.download()
            load_path = constants.datasets_folder + self.DATASET_ID + "data/Data"
            du.log("Loading dataset from files at {}".format(load_path))

            dataframes = []
            sub_folders = os.listdir(load_path)

            for i, sub_folder in enumerate(sub_folders, start=1):
                du.log("[{}/{}] processing user {}".format(i, len(sub_folders), sub_folder))

                user_folder = os.path.join(load_path, sub_folder)
                plt_files = glob.glob(os.path.join(user_folder, "Trajectory", "*.plt"))

                df = pd.concat([self.load_plt(file, i) for i, file in enumerate(plt_files, start=1)])
                df[constants.UID] = int(sub_folder)

                labels_file = os.path.join(user_folder, 'labels.txt')
                if os.path.exists(labels_file):
                    labels = self.load_labels(labels_file)
                    self.apply_labels(df, labels)
                else:
                    df['label'] = np.nan

                dataframes.append(df)

            dataframe = pd.concat(dataframes)
            self.data.load_data(dataframe)
            self.save_dataset()

    @staticmethod
    def load_labels(labels_file: str) -> pd.DataFrame:
        """
        Loads the labels from a given file

        :param str labels_file: filename of the file that should be loaded
        :return: returns the loaded file as a Pandas Dataframe
        """
        labels_df = pd.read_csv(labels_file, skiprows=1, header=None,
                                parse_dates=[[0, 1], [2, 3]],
                                infer_datetime_format=True, delim_whitespace=True)

        # for clarity rename columns
        labels_df.columns = ['start_time', 'end_time', 'label']

        return labels_df

    @staticmethod
    def load_plt(plt_file: str, trajectory_id: int) -> pd.DataFrame:
        """
        Loads the trajectory data given the filename

        :param str plt_file: filename of the file that should be loaded
        :param int trajectory_id: corresponding trajectory number of the loading file
        :return: returns the loaded file as a Pandas Dataframe
        """
        trajectory_df = pd.read_csv(plt_file, skiprows=6, header=None,
                                    parse_dates=[[5, 6]], infer_datetime_format=True)

        # for clarity rename columns
        trajectory_df.rename(inplace=True, columns={'5_6': constants.DATETIME,
                                                    0: constants.LATITUDE,
                                                    1: constants.LONGITUDE,
                                                    3: constants.ALTITUDE})

        # remove unused columns
        trajectory_df.drop(inplace=True, columns=[2, 4])
        trajectory_df[constants.TID] = trajectory_id

        return trajectory_df

    @staticmethod
    def apply_labels(trajectory_df: pd.DataFrame, labels_df: pd.DataFrame):
        """
        Applies the labels from the loaded file to the trajectories dataframe

        :param pd.DataFrame trajectory_df: trajectory dataframe previously loaded from dataset files
        :param pd.DataFrame labels_df: labels data previously loaded from dataset files
        """
        indices = labels_df['start_time'].searchsorted(trajectory_df[constants.DATETIME], side='right') - 1
        no_label = (indices < 0) | (trajectory_df[constants.DATETIME].values >= labels_df['end_time'].iloc[indices].values)
        trajectory_df['label'] = labels_df['label'].iloc[indices].values
        trajectory_df.loc[no_label, 'label'] = np.nan

    def process_dataset(self):
        pass

    def save_dataset(self):
        """Saves dataset"""
        self.data.save_data(filename=self.DATASET_ID)
