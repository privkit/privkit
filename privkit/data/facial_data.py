import itertools
import numpy as np
import open3d as o3d

from pathlib import Path
from numpy import ndarray
from open3d.cpu.pybind.geometry import PointCloud

from privkit.data import DataType
from privkit.utils import constants


class FacialData(DataType):
    """
    FacialData is a privkit.DataType to handle facial data. Facial data is defined as a collection of points in
    3D space, where each point is represented by its coordinates (x, y, z) and, optionally, additional attributes 
    like color or normal vectors. It is stored as an Open3D data structure with the PointCloud class.
    """
    DATA_TYPE_ID = "facial_data"
    DATA_TYPE_NAME = "Facial Data"
    DATA_TYPE_INFO = "Facial data can be imported through an Open3D data structure or read by a PLY file. " \
                     "To be supported, data should contain at least one point (x, y, z)."

    def __init__(self, id_name: str = None):
        super().__init__()
        self.id_name = id_name or self.DATA_TYPE_ID
        """Identifier name of this facial data instance"""
        self.data = None
        """Facial data is stored as an Open3D data structure"""

    def load_data(self, pcd_or_filepath: PointCloud or str or Path):
        """
        Loads facial data from a PointCloud, an array with dimensions (#dim, 3) or a file that can be read using Open3D's read_point_cloud() method.

        :param DataFrame or str or Path pcd_or_filepath: either a PointCloud instance, a file path to a point cloud, or an array with the points coordinates
        """

        if isinstance(pcd_or_filepath, PointCloud):
            self.data = pcd_or_filepath
        elif isinstance(pcd_or_filepath, ndarray):
            self.data = o3d.geometry.PointCloud()
            self.data.points = o3d.utility.Vector3dVector(pcd_or_filepath)
        else:
            self.data = o3d.io.read_point_cloud(pcd_or_filepath)

    def process_data(self):
        pass

    def save_data(self, filepath: str = constants.data_folder, filename: str = None, extension: str = "ply"):
        """
        Saves data to a file.

        :param str filepath: path where data should be saved.
        :param str filename: name of the file to be saved.
        :param str extension: extension of the format of how the file should be saved. The default value is `'ply'`.
        """
        if filename is None:
            filename = self.id_name
        store_path = filepath + '/' + filename + '.' + extension
        o3d.io.write_point_cloud(filename=store_path, pointcloud=self.data)

    def crop_pcd(self, xmin: float, xmax: float, ymin: float, ymax: float, zmin: float, zmax: float):
        """
        Segment the point cloud with a bounding box 

        :param float xmin: minimum x-coordinate of the bounding box
        :param float xmax: maximum x-coordinate of the bounding box
        :param float ymin: minimum y-coordinate of the bounding box
        :param float ymax: maximum y-coordinate of the bounding box
        :param float zmin: minimum z-coordinate of the bounding box
        :param float zmax: maximum z-coordinate of the bounding box
        """
        # set the bounds, create the limit points and crate a bounding box object
        bounds = [[xmin, xmax], [ymin, ymax], [zmin, zmax]]
        bounding_box_points = list(itertools.product(*bounds))
        bounding_box = o3d.geometry.AxisAlignedBoundingBox.create_from_points(
            o3d.utility.Vector3dVector(bounding_box_points))

        # crop the point cloud using the bounding box
        pcd_data = self.data.crop(bounding_box)
        self.data = pcd_data

    def remove_points_outside_sphere(self, center: ndarray, radius: float):
        """
        Segment the point cloud with a sphere 

        :param ndarray center: coordinates of the center of the sphere
        :param float radius: radius of the sphere
        """
        points = np.asarray(self.data.points)

        # Calculate the distances between the points and the center of the sphere
        distances = np.linalg.norm(points - center, axis=1)

        # Keep only the points that are inside the sphere (distance <= radius)
        mask = distances <= radius
        np_points = points[mask]

        filtered_pcd = o3d.geometry.PointCloud()
        filtered_pcd.points = o3d.utility.Vector3dVector(np_points)

        if bool(self.data.colors):
            colors = np.array(self.data.colors)
            np_colors = colors[mask]
            filtered_pcd.colors = o3d.utility.Vector3dVector(np_colors)

        self.data = filtered_pcd

    def remove_outliers_statistical(self, nb_neighbors: float, std_ratio: float):
        """
        Remove outlier from the point cloud based on neighboring distance

        :param float nb_neighbors: number of neighbors for outlier detection
        :param float std_ratio: standard deviation for threshold computation
        """
        filtered_pcd, ind = self.data.remove_statistical_outlier(nb_neighbors, std_ratio)
        self.data = filtered_pcd

    def fp_downsample(self, N: int):
        """
        Downsample the point cloud with the Farthest Point Sampling technique

        :param int N: number of point of the sampled point cloud
        """
        self.data = self.data.farthest_point_down_sample(num_samples=N)

    # ========================= Statistics methods =========================

    def get_number_of_points(self):
        """
        Returns the number of points of the point cloud
        :return: number of points of the point cloud
        """
        return np.shape(self.data.points)[0]

    def get_color(self):
        """
        Returns a boolean value indicating whether the point cloud has color
        :return: True if the point cloud has color, False otherwise
        """
        return bool(self.data.colors)

    def get_point_median(self):
        """
        Returns the median coordinate of the point cloud points
        :return: median point cloud coordinate
        """
        pcd_array = np.array(self.data.points)
        x_median = np.median(pcd_array[:, 0])
        y_median = np.median(pcd_array[:, 1])
        z_median = np.median(pcd_array[:, 2])

        return round(x_median, 3), round(y_median, 3), round(z_median, 3)

    def get_point_mean_std(self):
        """
        Returns the average coordinate of the point cloud points along with the standard deviation
        :return: Average point cloud coordinate and standard deviation
        """
        pcd_array = np.array(self.data.points)
        x_mean = np.mean(pcd_array[:, 0])
        x_std = np.std(pcd_array[:, 0])

        y_mean = np.mean(pcd_array[:, 1])
        y_std = np.std(pcd_array[:, 1])

        z_mean = np.mean(pcd_array[:, 2])
        z_std = np.std(pcd_array[:, 2])

        return round(x_mean, 3), round(x_std, 3), round(y_mean, 3), round(y_std, 3), round(z_mean, 3), round(z_std, 3)

    def print_data_summary(self):
        """
        Prints data summary, specifying the number of points, and color availability
        """
        print("\n=======================================")
        print("{} contains:".format(self.DATA_TYPE_NAME))

        print("{} points".format(self.get_number_of_points()))
        if self.get_color():
            print('Color information')
        else:
            print('No color information')

        x_median, y_median, z_median = self.get_point_median()
        print(f"({x_median}, {y_median}, {z_median}) median coordinates")

        x_mean, x_std, y_mean, y_std, z_mean, z_std = self.get_point_mean_std()
        print(f"({x_mean} ± {x_std}, {y_mean} ± {y_std}, {z_mean} ± {z_std}) mean and std. coordinates")

        print("=======================================\n")
