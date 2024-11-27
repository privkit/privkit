import numpy as np

from privkit.attacks import Attack
from privkit.data import LocationData
from privkit.ppms import PlanarLaplace
from privkit.metrics import AdversaryError
from privkit.utils import (
    GridMap,
    constants,
    geo_utils as gu,
    training_utils as tu
)


class HW(Attack):
    """
    Class to execute the HW attack.

    References
    ----------
    R. Shokri, G. Theodorakopoulos, C. Troncoso, J.-P. Hubaux, and J.-Y. Le Boudec, “Protecting
    location privacy: optimal strategy against localization attacks,” in Proceedings of the 2012
    ACM conference on Computer and communications security, pp. 617–627, 2012.
    """

    ATTACK_ID = "hw"
    ATTACK_NAME = "HW"
    ATTACK_INFO = "HW is a mechanism that given the matrix f(z|...), multiplies it by the given mobility profile " \
                  "and then normalizes it, calculating the posterior. Then computes the geometric median with the " \
                  "posterior calculated returning the adversary estimation."
    ATTACK_REF = "R. Shokri, G. Theodorakopoulos, C. Troncoso, J.-P. Hubaux, and J.-Y. Le Boudec, “Protecting " \
                 "location privacy: optimal strategy against localization attacks,” in Proceedings of the 2012 " \
                 "ACM conference on Computer and communications security, pp. 617–627, 2012."
    DATA_TYPE_ID = [LocationData.DATA_TYPE_ID]
    METRIC_ID = [AdversaryError.METRIC_ID]

    def __init__(self, epsilon: float, mobility_profile: [float]):
        """
        Initializes the HW attack

        :param epsilon: LPPM privacy parameter
        :param [float] mobility_profile: priori knowledge build with a portion of the dataset
        """

        super().__init__()
        self.epsilon = epsilon
        self.mobility_profile = mobility_profile

    def execute(self, location_data: LocationData):
        """
        Executes the HW attack

        :param privkit.LocationData location_data: data where OptimalHW will be performed
        :return: location data updated with adversary estimation
        """
        if {constants.OBF_LATITUDE, constants.OBF_LONGITUDE}.issubset(location_data.data.columns):
            if not hasattr(location_data, "grid"):
                location_data.create_grid(*location_data.get_bounding_box_range(), spacing=2 / self.epsilon)
                location_data.filter_outside_points(*location_data.get_bounding_box_range())

            for test_user_index_uid, test_user_index_tid in location_data.test_user_indexes:
                user_data = location_data.data[(location_data.data[constants.UID] == test_user_index_uid) & (
                            location_data.data[constants.TID] == test_user_index_tid)]
                adv_latitude_output = []
                adv_longitude_output = []
                for obf_latitude, obf_longitude in zip(user_data[constants.OBF_LATITUDE],
                                                       user_data[constants.OBF_LONGITUDE]):
                    x, y, z = gu.geodetic2cartesian(obf_latitude, obf_longitude, gu.EARTH_RADIUS / 1000)
                    z_i = np.array([x, y])

                    f_zx = PlanarLaplace(self.epsilon).planar_laplace_distribuition(z_i, location_data.grid)

                    [adv_x, adv_y] = self.execute_attack(f_zx, location_data.grid)
                    adv_latitude, adv_longitude = gu.cartesian2geodetic(adv_x, adv_y, z)
                    adv_latitude_output.append(adv_latitude)
                    adv_longitude_output.append(adv_longitude)

                location_data.data.loc[user_data.index, constants.ADV_LATITUDE] = adv_latitude_output
                location_data.data.loc[user_data.index, constants.ADV_LONGITUDE] = adv_longitude_output
        else:
            raise KeyError(f"Obfuscated locations <{constants.OBF_LATITUDE}, {constants.OBF_LONGITUDE}> are missing, "
                           f"this attack should be applied after applying a privacy-preserving mechanism.")

        return location_data

    def execute_attack(self, f: [float], grid: GridMap):
        """
        Executes the HW attack at a point

        :param [float] f: LPPM function
        :param privkit.GridMap grid: map discretization
        :return: adversary estimation
        """

        posterior = f * self.mobility_profile
        posterior = posterior / np.sum(posterior)
        posterior = np.reshape(posterior, (-1, 1))

        cartesian_cells = grid.get_cartesian_cells()
        values = []

        size = grid.get_size()
        for i in range(size[0]):
            for j in range(size[1]):
                values.append(cartesian_cells[i][j][0:2])

        values = np.reshape(values, (-1, 2))

        return self.compute_geometric_median(posterior, values)

    def compute_geometric_median(self, unfiltered_probabilities: [float], unfiltered_values: [[float, float]]):
        """
        Computes the geometric median, which will be the adversary estimation at a certain query

        :param [float] unfiltered_probabilities: adversary priori knowledge about the user
        :param [[float, float]] unfiltered_values: center coordinates of grid map cells
        :return: geometric median
        """

        probabilities = []
        values = []
        N = np.size(unfiltered_probabilities, 0)

        for i in range(N):
            if unfiltered_probabilities[i][0] <= 0:
                continue
            probabilities.append(unfiltered_probabilities[i])
            values.append(unfiltered_values[i])

        probabilities = np.array(probabilities)
        values = np.array(values)

        geo_median_old = [np.inf, np.inf]
        geo_median = np.matmul(np.reshape(probabilities, (-1)), values)

        iteration = 1
        while np.sum((geo_median - geo_median_old) ** 2) > 1e-3 and iteration < 200:
            dmatrix = self.compute_dist(geo_median, values)

            if np.any(dmatrix == 0):
                break

            geo_median_old = geo_median

            t = np.reshape((probabilities / dmatrix), (-1))
            geo_median = np.matmul(t, values) / np.matmul(t, np.ones(np.shape(values)))

            iteration += 1

        return geo_median

    @staticmethod
    def compute_dist(geo_median_i: [float, float], grid_map_centers: [[float, float]]):
        """
        Computes distance

        :param [float, float] geo_median_i: geometric median at the i-th iteration
        :param [[float, float]] grid_map_centers: center coordinates of grid map cells
        :return: distance between the geometric median with every cell center
        """

        N = np.size(grid_map_centers, 0)
        geo_median_i = np.tile(geo_median_i, (N, 1))
        return np.sqrt(np.sum(np.square(geo_median_i - grid_map_centers), 1)).reshape(N, 1)


class OptimalHW(Attack):
    """
    Class to execute the OptimalHW attack.

    References
    ----------
    R. Shokri, G. Theodorakopoulos, C. Troncoso, J.-P. Hubaux, and J.-Y. Le Boudec, “Protecting
    location privacy: optimal strategy against localization attacks,” in Proceedings of the 2012
    ACM conference on Computer and communications security, pp. 617–627, 2012.
    """

    ATTACK_ID = "optHW"
    ATTACK_NAME = "OptHW"
    ATTACK_INFO = "OptHW is a mechanism that uses the HW attack using the mobility profile resulting from train data."
    ATTACK_REF = "R. Shokri, G. Theodorakopoulos, C. Troncoso, J.-P. Hubaux, and J.-Y. Le Boudec, “Protecting " \
                 "location privacy: optimal strategy against localization attacks,” in Proceedings of the 2012 " \
                 "ACM conference on Computer and communications security, pp. 617–627, 2012."
    DATA_TYPE_ID = [LocationData.DATA_TYPE_ID]
    METRIC_ID = [AdversaryError.METRIC_ID]

    def __init__(self, epsilon: float):
        """
        Initializes the OptimalHW attack

        :param float epsilon: LPPM privacy parameter
        """

        super().__init__()
        self.epsilon = epsilon

    def execute(self, location_data: LocationData):
        """
        Executes the OptimalHW attack

        :param privkit.LocationData location_data: data where OptimalHW will be performed
        :return: location data updated with adversary guess
        """

        mobility_profile = tu.BuildKnowledge().get_mobility_profile(location_data, constants.NORM_PROF)
        return HW(self.epsilon, mobility_profile).execute(location_data)


class OmniHW(Attack):
    """
    Class to execute the OmniHW attack.

    References
    ----------
    R. Shokri, G. Theodorakopoulos, C. Troncoso, J.-P. Hubaux, and J.-Y. Le Boudec, “Protecting
    location privacy: optimal strategy against localization attacks,” in Proceedings of the 2012
    ACM conference on Computer and communications security, pp. 617–627, 2012.
    """

    ATTACK_ID = "omniHW"
    ATTACK_NAME = "OmniHW"
    ATTACK_INFO = "OmniHW is a mechanism that uses the HW attack using the mobility profile resulting from test data."
    ATTACK_REF = "R. Shokri, G. Theodorakopoulos, C. Troncoso, J.-P. Hubaux, and J.-Y. Le Boudec, “Protecting " \
                 "location privacy: optimal strategy against localization attacks,” in Proceedings of the 2012 " \
                 "ACM conference on Computer and communications security, pp. 617–627, 2012."
    DATA_TYPE_ID = [LocationData.DATA_TYPE_ID]
    METRIC_ID = [AdversaryError.METRIC_ID]

    def __init__(self, epsilon: float):
        """
        Initializes the OmniHW attack

        :param float epsilon: LPPM privacy parameter
        """

        super().__init__()
        self.epsilon = epsilon

    def execute(self, location_data: LocationData):
        """
        Executes the OmniHW attack

        :param privkit.LocationData location_data: data where OmniHW will be performed
        :return: location data updated with adversary guess
        """

        mobility_profile = tu.BuildKnowledge().get_mobility_profile(location_data, constants.OMNI_PROF)
        return HW(self.epsilon, mobility_profile).execute(location_data)
