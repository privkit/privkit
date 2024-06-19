import numpy as np
import open3d as o3d

from typing import Union
from numpy import ndarray
from open3d.cpu.pybind.geometry import PointCloud
from open3d.cpu.pybind.pipelines.registration import Feature, RegistrationResult

from privkit.ppms import PPM
from privkit.data import FacialData


class Merge2Faces(PPM):
    """
    Merge2Faces class to apply the mechanism

    References
    ----------
    Ricardo Andrade. 2023 
    Privacy-Preserving Face Detection: A Comprehensive Analysis of Face Anonymization Techniques.
    Master's Thesis. Universidade do Porto.
    """
    PPM_ID = "merge_two_faces"
    PPM_NAME = "Merge2Faces"
    PPM_INFO = "The Merge2Faces method combines attributes from two facial point clouds through a two-stage process. " \
               "Initially, the original and new faces undergo registration, starting with a global registration " \
               "using the Random Sample Consensus (RANSAC) algorithm, followed by refinement using a local registration " \
               "method called Point-to-Plane ICP. The second stage involves weighted averaging of coordinates and colors " \
               "for each point on the original face with their corresponding closest point on the new face. " \
               "Dissimilarity to the original point cloud is regulated by both the new face and the assigned weight. " \
               "The registration process code is based on the open3d's documentation for global_registration " \
               "[https://www.open3d.org/docs/release/tutorial/pipelines/global_registration.html]."
    PPM_REF = "Ricardo Andrade. 2023 " \
              "Privacy-Preserving Face Detection: A Comprehensive Analysis of Face Anonymization Techniques. " \
              "Master's Thesis. Universidade do Porto."
    DATA_TYPE_ID = [FacialData.DATA_TYPE_ID]

    def __init__(self, pcd_new: Union[PointCloud, str, ndarray], weight: float, voxel_size: float):
        """
        Initializes the Merge2Faces mechanism by defining the required parameters 
        
        :param PointCloud or str or ndarray pcd_new: either a PointCloud instance, a file path to a point cloud, or an array with the points coordinates
        :param voxel_size: voxel size for point cloud voxel downsampling upon registration (a rule of thumb is yielding 4k points)
        :param float weight: weight assigned to the new identity during neighboring averaging
        """
        super().__init__()
        self.pcd_new = pcd_new
        self.weight = weight
        self.voxel_size = voxel_size

    def execute(self, facial_data: FacialData):
        """
        Executes the Merge2Faces mechanism to the data given as a parameter

        :param privkit.FacialData facial_data: facial data where the Merge2Faces should be executed
        :return: obfuscated facial data 
        """
        pcd_target = facial_data.data

        facial_data = FacialData('source_pcd')
        facial_data.load_data(self.pcd_new)
        pcd_source = facial_data.data

        # global registration 
        source, target, source_down, target_down, source_fpfh, target_fpfh = self.prepare_dataset(pcd_source,
                                                                                                  pcd_target,
                                                                                                  self.voxel_size)
        result_ransac = self.execute_global_registration(source_down, target_down,
                                                         source_fpfh, target_fpfh,
                                                         self.voxel_size)
        # source.transform(result_ransac.transformation)
        # refine registration
        pcd_target.estimate_normals()
        source.estimate_normals()
        result_icp = self.refine_registration(source, target, self.voxel_size, result_ransac)
        source.transform(result_icp.transformation)

        # second stage of the method
        np_points_source = np.array(source.points)
        np_points_target = np.array(target.points)

        pcd_tree = o3d.geometry.KDTreeFlann(target)

        # calculate the coordinates and colors mean of a point considering its k_neighbours 
        neighbors_indices = [pcd_tree.search_knn_vector_3d(point, 1)[1] for point in np_points_source]
        np_average_points = [
            np.average(np.vstack((np_points_target[idx], np_points_source[i])), weights=[1 - self.weight, self.weight],
                       axis=0) for i, idx in enumerate(neighbors_indices)]

        pcd_anon = o3d.geometry.PointCloud()
        pcd_anon.points = o3d.utility.Vector3dVector(np_average_points)

        if all((bool(pcd_target.colors), bool(pcd_source.colors))):
            np_colors_source = np.array(source.colors)
            np_colors_target = np.array(target.colors)
            np_average_colors = [np.average(np.vstack((np_colors_target[idx], np_colors_source[i])),
                                            weights=[1 - self.weight, self.weight], axis=0) for i, idx in
                                 enumerate(neighbors_indices)]
            pcd_anon.colors = o3d.utility.Vector3dVector(np_average_colors)

        return pcd_anon

    @staticmethod
    def preprocess_point_cloud(pcd: PointCloud, voxel_size: float):
        """
        Returns a voxel-downsampled point cloud and the FPFH feature for each point.

        :param PointCloud pcd: point cloud to be downsampled
        :param float voxel_size: size of each downsampling voxel
        :return: point cloud with voxel downsampling and the FPFH feature for each point.
        """
        pcd_down = pcd.voxel_down_sample(voxel_size)

        radius_normal = voxel_size * 2
        pcd_down.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))

        radius_feature = voxel_size * 5
        pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
            pcd_down,
            o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
        return pcd_down, pcd_fpfh

    @staticmethod
    def execute_global_registration(source_down: PointCloud, target_down: PointCloud,
                                    source_fpfh: Feature, target_fpfh: Feature, voxel_size: float):
        """
        Return the global registration result using RANSANC

        :param PointCloud source_down: downsampled source point cloud
        :param PointCloud target_down: downsampled target point cloud
        :param open3d.cpu.pybind.pipelines.registration.Feature source_fpfh: FPFH feature of downsampled source point cloud
        :param open3d.cpu.pybind.pipelines.registration.Feature target_fpfh: FPFH feature of downsampled target point cloud
        :param float voxel_size: voxel size
        :return: RANSANC global registration result
        """
        distance_threshold = voxel_size * 1.5
        result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
            source_down, target_down, source_fpfh, target_fpfh, True,
            distance_threshold,
            o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
            3, [
                o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(
                    0.9),
                o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(
                    distance_threshold)
            ], o3d.pipelines.registration.RANSACConvergenceCriteria(100000, 0.999))
        return result

    def prepare_dataset(self, source: PointCloud, target: PointCloud, voxel_size: float):
        """
        Return the voxel downsapled point cloud and its FPFH feature of the point clouds source and target
        
        :param PointCloud source: a point cloud
        :param PointCloud target: a point cloud
        :param float voxel_size: voxel size
        :return: point cloud, voxel downsampled point cloud, FPFH feature of the two point clouds
        """
        source_down, source_fpfh = self.preprocess_point_cloud(source, voxel_size)
        target_down, target_fpfh = self.preprocess_point_cloud(target, voxel_size)
        return source, target, source_down, target_down, source_fpfh, target_fpfh

    @staticmethod
    def refine_registration(source: PointCloud, target: PointCloud, voxel_size: float, result_ransac: RegistrationResult):
        """
        Return the local registration result using Point-to-Plane ICP

        :param PointCloud source: a point cloud
        :param PointCloud target: a point cloud
        :param float voxel_size: voxel size
        :param open3d.cpu.pybind.pipelines.registration.RegistrationResult result_ransac: global registration result
        :return: Point-to-Plane ICP local registration result
        """
        distance_threshold = voxel_size * 0.0001
        result = o3d.pipelines.registration.registration_icp(
            source, target, distance_threshold, result_ransac.transformation,
            o3d.pipelines.registration.TransformationEstimationPointToPlane())
        return result
