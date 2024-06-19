import numpy as np
import open3d as o3d

from privkit.ppms import PPM
from privkit.data import FacialData


class CentroidVoxel(PPM):
    """
    CentroidVoxel class to apply the mechanism

    References
    ----------
    Ricardo Andrade. 2023 
    Privacy-Preserving Face Detection: A Comprehensive Analysis of Face Anonymization Techniques.
    Master's Thesis. Universidade do Porto.
    """
    PPM_ID = "centroid_voxel"
    PPM_NAME = "CentroidVoxel"
    PPM_INFO = "The CentroidVoxel operation voxelizes facial points into regularly spaced cubic " \
               "elements, known as voxels. Each voxel is represented by a single point with centroid " \
               "coordinates, and its color is determined by averaging the colors of all-encompassed points. " \
               "The voxel size governs information compression on each voxel point, influencing data loss."
    PPM_REF = "Ricardo Andrade. 2023 " \
              "Privacy-Preserving Face Detection: A Comprehensive Analysis of Face Anonymization Techniques. " \
              "Master's Thesis. Universidade do Porto."
    DATA_TYPE_ID = [FacialData.DATA_TYPE_ID]

    def __init__(self, voxel_size: float):
        """
        Initializes the CentroidVoxel mechanism by defining the required parameters

        :param float voxel_size: size of each voxel in the voxel grid
        """
        super().__init__()
        self.voxel_size = voxel_size

    def execute(self, facial_data: FacialData):
        """
        Executes the CentroidVoxel mechanism to the data given as a parameter

        :param privkit.FacialData facial_data: facial data where the CentroidVoxel should be executed
        :return: obfuscated facial data 
        """
        pcd = facial_data.data

        voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd, voxel_size=self.voxel_size)
        pcd_np = np.asarray(
            [voxel_grid.origin + pt.grid_index * voxel_grid.voxel_size for pt in voxel_grid.get_voxels()])

        pcd_anon = o3d.geometry.PointCloud()
        pcd_anon.points = o3d.utility.Vector3dVector(pcd_np)
        if bool(facial_data.data.colors):
            pcd_colors = np.asarray([pt.color for pt in voxel_grid.get_voxels()])
            pcd_anon.colors = o3d.utility.Vector3dVector(pcd_colors)

        return pcd_anon
