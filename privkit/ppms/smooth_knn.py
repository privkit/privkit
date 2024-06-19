import numpy as np
import open3d as o3d

from privkit.ppms import PPM
from privkit.data import FacialData


class SmoothKNN(PPM):
    """
    Smooth k-nearest neighbors class to apply the mechanism

    References
    ----------
    Ricardo Andrade. 2023 
    Privacy-Preserving Face Detection: A Comprehensive Analysis of Face Anonymization Techniques.
    Master's Thesis. Universidade do Porto.
    """
    PPM_ID = "smooth_knn"
    PPM_NAME = "SmoothKNN"
    PPM_INFO = "The SmoothKNN operation involves replacing each point in the facial point cloud with the " \
               "mean coordinates and color attributes of its k-nearest neighbors. The degree of smoothing " \
               "is controlled by the number of neighboring points considered."
    PPM_REF = "Ricardo Andrade. 2023 " \
              "Privacy-Preserving Face Detection: A Comprehensive Analysis of Face Anonymization Techniques. " \
              "Master's Thesis. Universidade do Porto."
    DATA_TYPE_ID = [FacialData.DATA_TYPE_ID]

    def __init__(self, k: int):
        """
        Initializes the SmoothKNN mechanism by defining the required parameters

        :param float k: number of nearest neighbors considered for averaging 
        """
        super().__init__()
        self.k = k

    def execute(self, facial_data: FacialData):
        """
        Executes the SmoothKNN mechanism to the data given as a parameter

        :param privkit.FacialData facial_data: facial data where the SmoothKNN should be executed
        :return: obfuscated facial data 
        """
        pcd = facial_data.data

        pcd_points = np.array(pcd.points)

        pcd_tree = o3d.geometry.KDTreeFlann(pcd)

        neighbors_indices = [pcd_tree.search_knn_vector_3d(point, self.k)[1] for point in pcd_points]
        pcd_points_neighbours = [np.mean(pcd_points[idx], axis=0, dtype=np.float64) for idx in neighbors_indices]

        pcd_anon = o3d.geometry.PointCloud()
        pcd_anon.points = o3d.utility.Vector3dVector(pcd_points_neighbours)
        if bool(facial_data.data.colors):
            pcd_colors = np.array(pcd.colors)
            pcd_colors_neighbours = [np.mean(pcd_colors[idx], axis=0, dtype=np.float64) for idx in neighbors_indices]
            pcd_anon.colors = o3d.utility.Vector3dVector(pcd_colors_neighbours)

        return pcd_anon
