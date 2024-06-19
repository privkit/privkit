import numpy as np

from scipy.special import lambertw
from random import random
from math import pi, e, cos, sin, exp

from privkit.ppms import PPM
from privkit.data import LocationData
from privkit.metrics import QualityLoss
from privkit.utils import (
    geo_utils as gu,
    constants,
    GridMap
)


class PlanarLaplace(PPM):
    """
    Planar Laplace class to apply the mechanism

    References
    ----------
    Andrés, M. E., Bordenabe, N. E., Chatzikokolakis, K., & Palamidessi, C. (2013, November).
    Geo-indistinguishability: Differential privacy for location-based systems. In Proceedings of the 2013 ACM SIGSAC
    conference on Computer & communications security (pp. 901-914).
    """
    PPM_ID = "planar_laplace"
    PPM_NAME = "Planar Laplace"
    PPM_INFO = "The geo-indistinguishable Planar Laplace (PL) consists of adding 2-dimensional Laplacian noise " \
               "centred at the exact user location. The Laplacian distribution depends on a privacy parameter " \
               "epsilon defined as ε=l/r, which means that a privacy level l is guaranteed within a radius r. " \
               "This mechanism is suitable for sporadic scenarios (i.e. single queries)."
    PPM_REF = "Andrés, M. E., Bordenabe, N. E., Chatzikokolakis, K., & Palamidessi, C. (2013, November). " \
              "Geo-indistinguishability: Differential privacy for location-based systems. In Proceedings of the 2013 " \
              "ACM SIGSAC conference on Computer & communications security (pp. 901-914)."
    DATA_TYPE_ID = [LocationData.DATA_TYPE_ID]
    METRIC_ID = [QualityLoss.METRIC_ID]

    def __init__(self, epsilon: float):
        """
        Initializes the Planar Laplace mechanism by defining the privacy parameter epsilon

        :param epsilon: privacy parameter
        """
        super().__init__()
        self.epsilon = epsilon

    def execute(self, location_data: LocationData):
        """
        Executes the Planar Laplace mechanism to the data given as parameter

        :param privkit.LocationData location_data: location data where the planar laplace should be executed
        :return: location data with obfuscated latitude and longitude and the quality loss metric
        """
        if hasattr(location_data, 'grid'):
            grid = location_data.grid
        else:
            grid = None

        output = []
        for latitude, longitude in zip(location_data.data[constants.LATITUDE], location_data.data[constants.LONGITUDE]):
            obf_lat, obf_lon, quality_loss = self.get_obfuscated_point(latitude, longitude, grid)
            output.append([obf_lat, obf_lon, quality_loss])

        location_data.data[[constants.OBF_LATITUDE, constants.OBF_LONGITUDE, QualityLoss.METRIC_ID]] = output

        return location_data

    def get_obfuscated_point(self, latitude: float, longitude: float, grid: GridMap = None) -> [float, float]:
        """
        Returns a geo-indistinguishable single point

        :param float latitude: original latitude
        :param float longitude: original longitude
        :param GridMap grid: grid discretization, so it repeats this step if the obfuscated points falls outside the grid
        :return: obfuscated point (latitude, longitude) and distance between original and obfuscation point
        """
        theta = random() * pi * 2  # Random number in [0,2*PI)
        p = random()  # Random variable [0,1)
        r = self.inverse_cumulative_gamma(p)

        x, y, z = gu.geodetic2cartesian(latitude, longitude)
        x = x + r * cos(theta)
        y = y + r * sin(theta)

        obf_latitude, obf_longitude = gu.cartesian2geodetic(x, y, z)

        if not grid or (grid and grid.point_within_grid(obf_latitude, obf_longitude)):
            return obf_latitude, obf_longitude, r
        else:
            return self.get_obfuscated_point(latitude, longitude, grid)

    def inverse_cumulative_gamma(self, p: float) -> float:
        """
        Computes the inverse cumulative gamma of Laplacian Distribution function.

        :param float p: random variable in [0,1)
        :return: inverse cumulative gamma
        """
        return - ((lambertw((p - 1) / e, k=-1) + 1).real / self.epsilon)

    def expected_error(self) -> float:
        """
        Computes the expected error calculated by 2/epsilon

        :return: expected error
        """
        return 2 / self.epsilon

    def planar_laplace_distribuition(self, point: [float, float], G: GridMap):
        """
        Returns the Planar Laplace distribution

        :param [float, float] point: cartesian coordinates x and y
        :param GridMap G: grid map discretization
        :return: Planar Laplace distribution
        """
        rows, cols = G.get_size()
        f_zx = np.zeros(rows * cols)
        for i in range(rows):
            for j in range(cols):
                x, y, _ = G.get_cartesian_cell(i, j)
                cur_x = np.array([x, y])
                f_zx[G.cells[i][j].id - 1] = self.laplace_distribution(point, cur_x)

        return f_zx

    def laplace_distribution(self, point1: [float, float], point2: [float, float]) -> float:
        """
        Computes the laplacian distribution value

        :param [float, float] point1: cartesian coordinates x and y
        :param [float, float] point2: cartesian coordinates x and y
        :return: laplacian distribution value
        """
        dist = gu.euclidean_distance(point1[0], point1[1], point2[0], point2[1])
        return ((self.epsilon * 1000) ** 2 / (2 * pi)) * exp(-self.epsilon * 1000 * dist)
