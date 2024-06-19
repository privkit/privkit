import numpy as np
import open3d as o3d
from typing import Callable

from privkit.ppms import PPM
from privkit.data import FacialData


class Tapering(PPM):
    """
    Tapering class to apply the mechanism

    References
    ----------
    Ricardo Andrade. 2023 
    Privacy-Preserving Face Detection: A Comprehensive Analysis of Face Anonymization Techniques.
    Master's Thesis. Universidade do Porto.
    """
    PPM_ID = "tapering"
    PPM_NAME = "Tapering"
    PPM_INFO = "Tapering applies a non-linear tapering deformation to facial points using a transformation " \
               "matrix guided by a tapering function. The shape and intensity of the deformation are regulated " \
               "by both the tapering function and its restricted domain. The outcome is linked to the coordinate " \
               "values of the points designated for anonymization."
    PPM_REF = "Ricardo Andrade. 2023 " \
              "Privacy-Preserving Face Detection: A Comprehensive Analysis of Face Anonymization Techniques. " \
              "Master's Thesis. Universidade do Porto."
    DATA_TYPE_ID = [FacialData.DATA_TYPE_ID]

    def __init__(self, k_min: float, k_max: float, f_tapering: Callable[[float], float] = None):
        """
        Initializes the Tapering mechanism by defining the required parameters 
        
        :param float k_min: lower bound of the restricted domain for the function
        :param float k_max: upper bound of the restricted domain for the function
        :param Callable[[float], float] f_tapering: tapering function. The default function is: def f_tapering(x): return 0.05*np.sin(x)**2 + np.cos(x)*0.5
        """
        super().__init__()
        self.k_min = k_min
        self.k_max = k_max
        if f_tapering is None:
            self.f_tapering = self.tapering_function
        else:
            self.f_tapering = f_tapering

    def execute(self, facial_data: FacialData):
        """
        Executes the Tapering mechanism to the data given as a parameter

        :param privkit.FacialData facial_data: facial data where the Tapering should be executed
        :return: obfuscated facial data 
        """
        pcd = facial_data.data

        pcd_points = np.array(pcd.points)

        sort_indices = np.argsort(pcd_points[:, 2])
        pcd_points = pcd_points[sort_indices]

        x_values = np.linspace(self.k_min, self.k_max, pcd_points.shape[0])
        y_values = self.f_tapering(x_values)

        pcd_array_taper = []
        s = 0
        if bool(facial_data.data.colors):
            pcd_colors = np.array(pcd.colors)
            pcd_colors = pcd_colors[sort_indices]

            for i, point in enumerate(pcd_points):
                M = np.array([[s, 0, 0], [0, s, 0], [0, 0, 1]])
                s = y_values[i]
                pcd_array_taper.append((point @ M).tolist())

            pcd_anon = o3d.geometry.PointCloud()
            pcd_anon.points = o3d.utility.Vector3dVector(pcd_array_taper)
            pcd_anon.colors = o3d.utility.Vector3dVector(pcd_colors)

        else:

            for i, point in enumerate(pcd_points):
                M = np.array([[s, 0, 0], [0, s, 0], [0, 0, 1]])
                s = y_values[i]
                pcd_array_taper.append((point @ M).tolist())

            pcd_anon = o3d.geometry.PointCloud()
            pcd_anon.points = o3d.utility.Vector3dVector(pcd_array_taper)

        return pcd_anon

    @staticmethod
    def tapering_function(x): return 0.05 * np.sin(x) ** 2 + np.cos(x) * 0.5
