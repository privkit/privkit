from pathlib import Path
from deepface import DeepFace
from matplotlib import pyplot as plt

from privkit.data import FacialData
from privkit.metrics import Metric
from privkit.utils import face_utils as fu


class CMC(Metric):
    METRIC_ID = "cmc_curve"
    METRIC_NAME = "Cumulative Matching Characteristics Curve"
    METRIC_INFO = "CMC quantifies the likelihood of successful identification by indicating the proportion " \
                  "of queries where the correct match is found within the top N ranked results, providing insights " \
                  "into the system's overall recognition accuracy across different rank levels. " \
                  "It is assumed that both probe and gallery images associated with the same identity share " \
                  "identical names, and each identity is represented by only a single image in both the probe and " \
                  "gallery datasets. "
    DATA_TYPE_ID = [FacialData.DATA_TYPE_ID]

    def __init__(self, probe_path: str, gallery_path: str, model_name: str = 'ArcFace',
                 detector_backend: str = 'retinaface'):
        """
        Initializes the Cumulative Matching Characteristics (CMC) metric with the specified parameters

        :param str probe_path: directory path where the probe identities images are located
        :param str gallery_path: directory path where the gallery identity images are located
        :param str model_name: face recognition model (default is 'ArcFace')
        :param str detector_backend: face detection model for face recognition (default is 'RetinFace')
        """
        super().__init__()
        self.probe_path = probe_path
        self.gallery_path = gallery_path
        self.model_name = model_name
        self.detector_backend = detector_backend
        self.cmc_curve = None

    def execute(self):
        """
        Executes the CMC metric

        :return: dictionary with the computed metric
        """
        id_rank = dict()
        for identity_img in fu.list_files(self.probe_path, ('.png')):
            df = DeepFace.find(img_path=identity_img, db_path=self.gallery_path,
                               model_name=self.model_name, detector_backend=self.detector_backend,
                               enforce_detection=False, threshold=10)[0]

            image_id = Path(identity_img).stem
            rank_id = df['identity'].str.contains(image_id).idxmax() + 1
            id_rank[f'{image_id}'] = rank_id

        cmc_curve = dict()
        ranks = list(id_rank.values())
        for rank_k in range(1, len(ranks) + 1):
            num = sum(i <= rank_k for i in ranks)
            cmc_curve[rank_k] = num / len(ranks)

        self.cmc_curve = cmc_curve
        return cmc_curve

    def plot(self):
        if self.cmc_curve is not None:
            plt.plot(list(self.cmc_curve.keys()), list(self.cmc_curve.values()))
            plt.title('Cumulative Match Characteristic Curve')
            plt.xlabel('Rank')
            plt.ylabel('Recognition Rate')
            plt.show()
