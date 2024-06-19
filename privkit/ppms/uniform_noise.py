import numpy as np
import open3d as o3d

from privkit.ppms import PPM
from privkit.data import FacialData


class UniformNoise(PPM):
    """
    UniformNoise class to apply the mechanism

    References
    ----------
    Ricardo Andrade. 2023 
    Privacy-Preserving Face Detection: A Comprehensive Analysis of Face Anonymization Techniques.
    Master's Thesis. Universidade do Porto.
    """ 
    PPM_ID = "uniform_noise"
    PPM_NAME = "UniformNoise"
    PPM_INFO = "UniformNoise applies random values from a uniform distribution to the 3D coordinates (x, y, " \
               "z) of facial points. Each dimension of a point is independently altered by sampling noise from three " \
               "uniform distributions. To realign the face with its original position, the coordinates of each point " \
               "are adjusted by subtracting the mean of the corresponding uniform distribution. The intensity of the " \
               "noise is regulated by the parameters 'a' and 'b', representing the lower and upper bounds of the " \
               "uniform distribution for each dimension of a point."
    PPM_REF = "Ricardo Andrade. 2023 " \
              "Privacy-Preserving Face Detection: A Comprehensive Analysis of Face Anonymization Techniques. " \
              "Master's Thesis. Universidade do Porto."
    DATA_TYPE_ID = [FacialData.DATA_TYPE_ID]      
        
    def __init__(self, a_x: float, b_x: float, a_y: float, b_y: float, a_z: float, b_z: float):
        """
        Initializes the UniformNoise mechanism by defining the required parameters 

        :param float a_x: lower bound of the uniform distribution for coordinate x
        :param float b_x: upper bound of the uniform distribution for coordinate x
        :param float a_y: lower bound of the uniform distribution for coordinate y
        :param float b_y: upper bound of the uniform distribution for coordinate y
        :param float a_z: lower bound of the uniform distribution for coordinate z
        :param float b_z: upper bound of the uniform distribution for coordinate z
        """
        super().__init__()
        self.a_x = a_x
        self.b_x = b_x
        self.a_y = a_y
        self.b_y = b_y
        self.a_z = a_z
        self.b_z = b_z
            
    def execute(self, facial_data: FacialData): 
        """
        Executes the UniformNoise mechanism to the data given as a parameter

        :param privkit.FacialData facial_data: facial data where the UniformNoise should be executed
        :return: obfuscated facial data 
        """ 
        pcd = facial_data.data

        pcd_points = np.array(pcd.points)
        n_points = pcd_points.shape[0]
        
        random_x = np.random.uniform(self.a_x, self.b_x, n_points)
        mean_x = (self.a_x + self.b_x) / 2
        noise_x = random_x - mean_x
        
        random_y = np.random.uniform(self.a_y, self.b_y, n_points)
        mean_y = (self.a_y + self.b_y) / 2
        noise_y = random_y - mean_y
        
        random_z = np.random.uniform(self.a_z, self.b_z, n_points)
        mean_z = (self.a_z + self.b_z) / 2
        noise_z = random_z - mean_z
        
        noise = np.column_stack((noise_x, noise_y, noise_z))
        noise_pcd = pcd_points - noise        
        
        pcd_anon = o3d.geometry.PointCloud()
        pcd_anon.points = o3d.utility.Vector3dVector(noise_pcd)
        if bool(facial_data.data.colors):
            pcd_colors = np.array(pcd.colors)
            pcd_anon.colors = o3d.utility.Vector3dVector(pcd_colors)            
            
        return pcd_anon
