import numpy as np

from privkit.attacks import Attack, HW
from privkit.data import LocationData
from privkit.ppms import PlanarLaplace
from privkit.metrics import AdversaryError, F1ScoreMM
from privkit.utils import (
    GridMap,
    constants,
    geo_utils as gu,
    training_utils as tu
)


class PEBA(Attack):
    """
    Class to execute the PEBA attack.

    References
    ----------
    S. Oya, C. Troncoso, and F. Pérez-González, “Rethinking location privacy for unknown mobility behaviors,”
    in 2019 IEEE European Symposium on Security and Privacy (EuroS&P), pp. 416–431, IEEE, 2019.
    """

    ATTACK_ID = "peba"
    ATTACK_NAME = "Profile-Estimation-Based-Attack"
    ATTACK_INFO = "PEBA is a mechanism that first computes the Maximum Likelihood Estimator of the mobility profile" \
                  "using all the f(z|...) collected and then uses OptimalHW to return the adversary estimation."
    ATTACK_REF = "S. Oya, C. Troncoso, and F. Pérez-González, “Rethinking location privacy for unknown mobility " \
                 "behaviors, in 2019 IEEE European Symposium on Security and Privacy (EuroS&P), pp. 416–431, IEEE, " \
                 "2019. "
    DATA_TYPE_ID = [LocationData.DATA_TYPE_ID]
    METRIC_ID = [AdversaryError.METRIC_ID,
                 F1ScoreMM.METRIC_ID]

    def __init__(self, epsilon: float, Niter_ML_max: int = 50, tolerance_ML: float = 0.005):
        """
        Initializes the PEBA attack

        :param float epsilon: LPPM privacy parameter
        :param int Niter_ML_max: maximum number of iterations when updating profile
        :param float tolerance_ML: precision of the profile update
        """

        super().__init__()

        self.epsilon = epsilon
        self.hw = HW(epsilon=self.epsilon)
        self.Niter_ML_max = Niter_ML_max
        self.tolerance_ML = tolerance_ML

    def execute(self, location_data: LocationData):
        """
        Executes the PEBA attack

        :param privkit.LocationData location_data: data where PEBA will be performed
        :return: location data updated with adversary guess
        """
        if {constants.OBF_LATITUDE, constants.OBF_LONGITUDE}.issubset(location_data.data.columns):
            if not hasattr(location_data, "grid"):
                location_data.create_grid(*location_data.get_bounding_box_range(), spacing=2/self.epsilon)
                location_data.filter_outside_points(*location_data.get_bounding_box_range())

            mobility_profile = tu.BuildKnowledge().get_mobility_profile(location_data, constants.NORM_PROF)

            F_zx = None
            for test_user_index_uid, test_user_index_tid in location_data.test_user_indexes:
                user_data = location_data.data[(location_data.data[constants.UID] == test_user_index_uid) & (location_data.data[constants.TID] == test_user_index_tid)]
                output = []
                for obf_latitude, obf_longitude in zip(user_data[constants.OBF_LATITUDE], user_data[constants.OBF_LONGITUDE]):
                    x, y, z = gu.geodetic2cartesian(obf_latitude, obf_longitude, gu.EARTH_RADIUS/1000)
                    z_i = np.array([x, y])

                    f_zx = PlanarLaplace(self.epsilon).planar_laplace_distribuition(z_i, location_data.grid)
                    f_zx = np.reshape(f_zx, (-1, 1))

                    if F_zx is None:
                        F_zx = f_zx
                    else:
                        F_zx = np.hstack((F_zx, f_zx))

                    [adv_x, adv_y] = self.execute_attack(F_zx, mobility_profile, location_data.grid)
                    adv_latitude, adv_longitude = gu.cartesian2geodetic(adv_x, adv_y, z)
                    output.append([adv_latitude, adv_longitude])

                location_data.data.loc[user_data.index, [constants.ADV_LATITUDE, constants.ADV_LONGITUDE]] = output

                if iter == self.Niter_ML_max:
                    break
        else:
            raise KeyError(f"Obfuscated locations <{constants.OBF_LATITUDE}, {constants.OBF_LONGITUDE}> are missing, "
                           f"this attack should be applied after applying a privacy-preserving mechanism.")

        return location_data

    def execute_attack(self, f: [[float]], mobility_profile: [float], grid: GridMap):
        """
        Executes the PEBA attack at a point

        :param [[float]] f: LPPM function
        :param [float] mobility_profile: priori knowledge build with a portion of the dataset
        :param privkit.GridMap grid: map discretization
        :return: adversary estimation
        """

        mobility_profile = np.reshape(mobility_profile, (-1, 1))
        pi_MLE = self.get_profile_MLE(f, mobility_profile)

        cur_query_index = np.size(f, 1)

        norm_pi_MLE = 1/np.sqrt(cur_query_index)*mobility_profile + (1 - 1 / np.sqrt(cur_query_index))*pi_MLE

        return self.hw.execute_attack(np.reshape(f[:, cur_query_index-1], (-1, 1)), norm_pi_MLE, grid)

    def get_profile_MLE(self, f: [[float]], mobility_profile: [float]):
        """
        Computes the Maximum Likelihood Estimator (MLE) of the mobility profile

        :param [[float]] f: LPPM function
        :param [float] mobility_profile: priori knowledge build with a portion of the dataset
        :return: updated mobility profile
        """

        pi_MLE = 0.8 * mobility_profile + 0.2 / np.size(mobility_profile)

        if self.tolerance_ML is None:
            for _ in range(self.Niter_ML_max):
                pi_MLE = self.update(f, pi_MLE)

        else:
            for _ in range(self.Niter_ML_max):
                pi_IB_new = self.update(f, pi_MLE)
                if np.max(np.abs(pi_IB_new - pi_MLE)) < self.tolerance_ML:
                    pi_MLE = pi_IB_new
                    break
                else:
                    pi_MLE = pi_IB_new

        pi_MLE[pi_MLE < 0] = 0
        pi_MLE = pi_MLE / np.sum(pi_MLE)

        return pi_MLE

    @staticmethod
    def update(f: [[float]], pi_MLE: [float]):
        """
        Updates the Maximum Likelihood Estimator of the mobility profile

        :param [[float]] f: LPPM function
        :param [float] pi_MLE: approximation of the Maximum Likelihood Estimator of the mobility profile
        :return: updated Maximum Likelihood Estimator of the mobility profile
        """

        pi_MLE_ = np.reshape(pi_MLE, (-1))
        a = np.matmul(np.diag(pi_MLE_), f)
        b = np.diag(np.matmul(pi_MLE_, f))

        pi_MLE_new = np.matmul(np.matmul(a, b.T), np.linalg.inv(np.matmul(b, b.T)))

        return np.reshape(np.mean(pi_MLE_new, 1), (-1, 1))
