import os
import torch
import numpy as np

from torch import tensor
from pathlib import Path
from retinaface import RetinaFace

from privkit.metrics import Metric
from privkit.data import FacialData
from privkit.utils import face_utils as fu


class AccuracyFD(Metric):
    METRIC_ID = "accuracy_face_detection"
    METRIC_NAME = "Face Detection Accuracy"
    METRIC_INFO = "The face detection accuracy measures the ratio of correctly detected faces " \
                  "to the total number of faces of a face detection model."
    DATA_TYPE_ID = [FacialData.DATA_TYPE_ID]

    def __init__(self, probe_path: str, baseline_path: str, iou_threshold: float = 0.5):
        """
        Initializes the Face Detection Accuracy metric with the specified parameters

        :param str probe_path: directory path where the probe identities images are located
        :param str baseline_path: directory path where the baseline identity images are located
        """
        super().__init__()
        self.probe_path = probe_path
        self.baseline_path = baseline_path
        self.iou_threshold = iou_threshold

    def execute(self):
        """
        Executes the Face Detection Accuracy metric

        :return: dictionary with the detection accuracy and the Intersection Over Union (IoU) of the detection
        """
        iou = list()
        for identity_img in fu.list_files(self.probe_path, ('.png')):

            image_id = Path(identity_img).stem
            file_extension = os.path.splitext(identity_img)[1]
            resp_clear = RetinaFace.detect_faces(f'{self.baseline_path}/{image_id}{file_extension}')
            resp_anonymized = RetinaFace.detect_faces(identity_img)

            if len(resp_anonymized) == 1:

                # auxiliary values for iou 
                xmin, ymin, xmax, ymax = resp_clear['face_1']['facial_area']
                ground_truth_bbox = torch.tensor([xmin, ymin, xmax, ymax], dtype=torch.float)
                xmin_anon, ymin_anon, xmax_anon, ymax_anon = resp_anonymized['face_1']['facial_area']
                prediction_bbox = torch.tensor([xmin_anon, ymin_anon, xmax_anon, ymax_anon],
                                               dtype=torch.float)  # ground truth BB
                iou_tensor = self.get_iou(ground_truth_bbox, prediction_bbox)
                iou_value = float(iou_tensor.numpy())

                if iou_value > self.iou_threshold:
                    iou.append(iou_value)

        accuracy_fd = len(iou) / len(fu.list_files(self.probe_path, ('.png')))
        return {'accuracy_fd': accuracy_fd, 'iou': iou}

    @staticmethod
    def get_iou(ground_truth: tensor, pred: tensor):
        """
        Returns the IoU value of a bounding box prediction

        :param tensor ground_truth: a tensor with the ground truth bounding box coordinates
        :param tensor pred: a tensor with the predicted bounding box coordinates
        :return: IoU value 
        """
        # coordinates of the area of intersection.
        ix1 = np.maximum(ground_truth[0], pred[0])
        iy1 = np.maximum(ground_truth[1], pred[1])
        ix2 = np.minimum(ground_truth[2], pred[2])
        iy2 = np.minimum(ground_truth[3], pred[3])

        # Intersection height and width.
        i_height = np.maximum(iy2 - iy1 + 1, np.array(0.))
        i_width = np.maximum(ix2 - ix1 + 1, np.array(0.))

        area_of_intersection = i_height * i_width

        # Ground Truth dimensions.
        gt_height = ground_truth[3] - ground_truth[1] + 1
        gt_width = ground_truth[2] - ground_truth[0] + 1

        # Prediction dimensions.
        pd_height = pred[3] - pred[1] + 1
        pd_width = pred[2] - pred[0] + 1

        area_of_union = gt_height * gt_width + pd_height * pd_width - area_of_intersection
        iou = area_of_intersection / area_of_union

        return iou

    def plot(self):
        pass
