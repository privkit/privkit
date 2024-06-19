import open3d as o3d

from privkit.ppms import PPM
from privkit.data import FacialData


class PointMeshPoint(PPM):
    """
    Point-mesh-point class to apply the mechanism

    References
    ----------
    Ricardo Andrade. 2023 
    Privacy-Preserving Face Detection: A Comprehensive Analysis of Face Anonymization Techniques.
    Master's Thesis. Universidade do Porto.
    """
    PPM_ID = "point_mesh_point"
    PPM_NAME = "Point-Mesh-Point"
    PPM_INFO = "The Point-Mesh-Point algorithm involves two main steps: it first transforms the " \
               "point cloud into an alpha-shape mesh, and subsequently, it reverts this alpha-shape mesh " \
               "back into a point cloud. The alpha value serves as a parameter controlling the level of detail " \
               "in facial features, and this is further influenced by the number of points defined in the final cloud."
    PPM_REF = "Ricardo Andrade. 2023 " \
              "Privacy-Preserving Face Detection: A Comprehensive Analysis of Face Anonymization Techniques. " \
              "Master's Thesis. Universidade do Porto."
    DATA_TYPE_ID = [FacialData.DATA_TYPE_ID]

    def __init__(self, alpha: float, n: float):
        """
        Initializes the Point-mesh-point mechanism by defining the required parameters

        :param float alpha: level of detail, determining proximity to the convex hull
        :param float n: number of points in the final point cloud
        """
        super().__init__()
        self.alpha = alpha
        self.n = n

    def execute(self, facial_data: FacialData):
        """
        Executes the Point-mesh-point mechanism to the data given as a parameter

        :param privkit.FacialData facial_data: facial data where the Point-mesh-point should be executed
        :return: obfuscated facial data 
        """
        mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(facial_data.data, self.alpha)
        pcd_anon = mesh.sample_points_uniformly(number_of_points=self.n)

        return pcd_anon
