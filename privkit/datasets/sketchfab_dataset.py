import io
import requests

from pathlib import Path
from zipfile import ZipFile

from privkit.data import FacialData
from privkit.datasets import Dataset
from privkit.utils import constants, dev_utils as du


class SketchfabDataset(Dataset):
    """
    Class to download facial models from the Sketchfab dataset.
    """
    DATASET_ID = 'sketchfab'
    DATASET_NAME = 'Sketchfab Dataset'
    DATASET_INFO = ("Sketchfab is a 3D asset website used to publish, share, discover, buy and sell 3D, VR and AR "
                    "content. The Sketchfab dataset provided in the Privkit allows to download models from Sketchfab, "
                    "but also provides some model as examples.")
    DATA_TYPE_ID = [FacialData.DATA_TYPE_ID]

    def __init__(self, api_token: str):
        """
        Initializes the SketchfabDataset class with the given API token.

        :param str api_token: API token for the Sketchfab website
        """
        super().__init__()
        self.data = FacialData(self.DATASET_ID)
        """Data is stored as an a FacialData structure"""
        self.api_token = api_token
        """API token for the Sketchfab website"""

    def load_dataset(self, model_id: int = 1):
        """
        Loads an example of facial model from the available.
        """
        models = {1: {"url": "https://skfb.ly/6Ut7w", "uid": "63553c2ce90c4a92b3bc42658eacd0e8"},
                  2: {"url": "https://skfb.ly/6XYVq", "uid": "d01569e2602b40a3a3b2f2a7b0f6400f"}}

        try:
            du.log(f"Loading the {self.DATASET_ID} dataset")
            load_path = constants.data_folder + self.DATASET_ID + str(model_id) + ".ply"

            file_to_load = Path(load_path)

            if file_to_load.is_file():
                self.data.load_data(load_path)
                du.log("Dataset loaded from {}".format(load_path))
            else:
                self.download_model(models[model_id]["uid"], filename=str(model_id))
        except Exception as e:
            du.log("Error while loading the {} dataset: {}".format(self.DATASET_ID, e))

    def download_model(self, uid: str, filename: str = None, savepath: str = constants.datasets_folder):
        """
        Downloads the model from the Sketchfab website given the model UID.

        :param str uid: model UID
        :param str filename: file name where the model should be saved
        :param str savepath: path where dataset should be saved. The default is `constants.datasets_folder`.
        """
        du.log(f"Downloading the model with uid {uid}")
        url = f"https://api.sketchfab.com/v3/models/{uid}/download"

        try:
            if filename is None:
                filename_to_save = f"{self.DATASET_ID}"
                filename = f"{self.DATASET_ID}.ply"
            else:
                filename_to_save = f"{self.DATASET_ID}{filename}"
                filename = f"{self.DATASET_ID}{filename}.ply"

            response = requests.get(
                url,
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Token {self.api_token}",
                },
                stream=True
            )
            url_to_download = response.json()['source']['url']

            response = requests.get(url_to_download, stream=True)
            with ZipFile(io.BytesIO(response.content), 'r') as zfile:
                for name in zfile.namelist():
                    if 'source' in name:
                        zfile_data = io.BytesIO(zfile.read(name))
                        with ZipFile(zfile_data) as zfile2:
                            if len(zfile2.infolist()) == 1:
                                file = zfile2.infolist()[0]
                                if 'ply' in file.filename:
                                    file.filename = filename
                                    Path(savepath).parent.mkdir(parents=True, exist_ok=True)
                                    zfile2.extract(file, path=savepath)

            filepath = f"{savepath}{filename}"

            du.log(f"The download of model {uid} was saved at {filepath}")

            self.data.load_data(filepath)
            self.save_dataset(filename=filename_to_save)
        except Exception as e:
            du.log("Error while downloading the {} dataset: {}".format(self.DATASET_ID, e))

    def process_dataset(self):
        pass

    def save_dataset(self, filename: str = None):
        """
        Saves dataset

        :param str filename: file name where the model should be saved
        """
        if filename is None:
            self.data.save_data(filename=self.DATASET_ID)
        else:
            self.data.save_data(filename=filename)
