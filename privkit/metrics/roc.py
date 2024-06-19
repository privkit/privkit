import numpy as np
import pandas as pd

from pathlib import Path
from sklearn import metrics
from deepface import DeepFace
from matplotlib import pyplot as plt
from pandas.core.frame import DataFrame

from privkit.data import FacialData
from privkit.metrics import Metric
from privkit.utils import face_utils as fu


class ROC_verify(Metric):
    METRIC_ID = "roc_verify"
    METRIC_NAME = "Receiver Operating Characteristic (ROC) Curve"
    METRIC_INFO = "ROC curve visually represents the trade-off between a face recognition (for verification) " \
                  "binary classification model's ability to correctly identify positive instances (true matches) " \
                  "and its tendency to make false positive errors."
    DATA_TYPE_ID = [FacialData.DATA_TYPE_ID]

    def __init__(self, df_pairs: DataFrame, model_name: str = 'ArcFace', detector_backend: str = 'retinaface'):
        """
        Initializes the ROC curve metric with the specified parameters

        :param DataFrame df_pairs: identity pairs to be compared
        :param str model_name: face recognition model (default is 'ArcFace')
        :param str detector_backend: face detection model for face recognition (default is 'RetinaFace')
        """
        super().__init__()
        self.df_pairs = df_pairs
        self.model_name = model_name
        self.detector_backend = detector_backend
        self.roc_curve = None

    def execute(self):
        """
        Executes the Receiver Operating Characteristic (ROC) curve and calculates the Area Under the Curve (AUC) value

        :return: A dictionary containing the True Positive Rate (TPR), False Positive Rate (FPR), and corresponding thresholds values for the ROC curve along with the AUC value
        """
        y_score = [DeepFace.verify(img1_path=self.df_pairs['pair1'][i],
                                   img2_path=self.df_pairs['pair2'][i],
                                   model_name=self.model_name, detector_backend=self.detector_backend,
                                   enforce_detection=False)['distance'] for i in range(self.df_pairs.shape[0])]

        fpr, tpr, thresholds = metrics.roc_curve(self.df_pairs.y_true, y_score, pos_label=1)
        auc_score = metrics.auc(tpr, fpr)

        self.roc_curve = {'fpr': tpr, 'tpr': fpr, 'thresholds': thresholds[::-1], 'auc': auc_score}
        return self.roc_curve

    def plot(self):
        if self.roc_curve is not None:
            auc_value = np.round(self.roc_curve['auc'], 3)
            plt.plot(self.roc_curve['fpr'], self.roc_curve['tpr'])
            plt.title('ROC curve')
            plt.xlabel('FPR')
            plt.ylabel('TPR')
            plt.legend([f'AUC= {auc_value}'])
            plt.show()


class ROC_find(Metric):
    METRIC_ID = "roc_find"
    METRIC_NAME = "Receiver Operating Characteristic (ROC) curve"
    METRIC_INFO = "ROC curve visually represents the trade-off between a face recognition (for verification) " \
                  "binary classification model's ability to correctly identify positive instances (true matches) " \
                  "and its tendency to make false positive errors. Faster than the ROC_verify implementation for a " \
                  "large number of identities. "
    DATA_TYPE_ID = [FacialData.DATA_TYPE_ID]

    def __init__(self, df_pairs: DataFrame):
        """
        Initializes the ROC curve metric with the specified parameters

        :param DataFrame df_pairs: identity pairs to be compared. The last two columns must represent predicted scores and true scores, respectively
        """
        super().__init__()
        self.df_pairs = df_pairs
        self.roc_find = None

    def execute(self):
        """
        Executes the Receiver Operating Characteristic (ROC) curve and calculates the Area Under the Curve (AUC) value

        :return: A dictionary containing the True Positive Rate (TPR), False Positive Rate (FPR), and corresponding thresholds values for the ROC curve along with the AUC value
        """
        tpr, fpr, thresholds = metrics.roc_curve(self.df_pairs[self.df_pairs.columns[-1]],
                                                 self.df_pairs[self.df_pairs.columns[-2]])
        auc_score = metrics.auc(fpr, tpr)

        self.roc_find = {'tpr': tpr, 'fpr': fpr, 'thresholds': thresholds[::-1], 'auc': auc_score}
        return self.roc_find

    def plot(self):
        if self.roc_find is not None:
            auc_value = np.round(self.roc_find['auc'], 3)
            plt.plot(self.roc_find['fpr'], self.roc_find['tpr'])
            plt.title('ROC curve')
            plt.xlabel('FPR')
            plt.ylabel('TPR')
            plt.legend([f'AUC= {auc_value}'])
            plt.show()


class MatchingScores(Metric):
    METRIC_ID = "matching_scores"
    METRIC_NAME = "Matching Scores"
    METRIC_INFO = "Matching scores are the numerical values assigned to pairs of facial images to " \
                  "quantify the similarity between them."
    DATA_TYPE_ID = [FacialData.DATA_TYPE_ID]

    def __init__(self, probe_path: str, gallery_path: str, model_name: str = 'ArcFace',
                 detector_backend: str = 'retinaface'):
        """
        Initializes the ROC metric with the specified parameters

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

    def execute(self):
        """
        Executes the ROC curve and AUC value.

        :param privkit.FacialData facial_data: data where ROC curve and AUC will be computed
        :param pd.DataFrame df_pairs: data frame with the identities (#####)
        :return: data with the computed metric
        """
        id_scores = pd.DataFrame()
        gallery_size = len(fu.list_files(self.gallery_path, ('.png')))
        for identity_img in fu.list_files(self.probe_path, ('.png')):
            result = DeepFace.find(img_path=identity_img, db_path=self.gallery_path,
                                   model_name=self.model_name, detector_backend=self.detector_backend,
                                   enforce_detection=False, threshold=10)[0]

            df = pd.DataFrame()
            df['id1'] = [Path(identity_img).stem] * gallery_size
            df['id2'] = result['identity'].apply(lambda x: Path(x).stem)
            df[result.columns[-1]] = result[result.columns[-1]]

            id_scores = pd.concat([id_scores, df], ignore_index=True)

        id_scores = id_scores.sort_values(by=result.columns[-1], ignore_index=True)
        id_scores['match'] = (id_scores['id1'] == id_scores['id2']).astype(int)

        return id_scores

    def plot(self):
        pass
