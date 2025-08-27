import numpy as np
import pandas as pd
import scipy.stats as st
from scipy.special import ndtr

from privkit.utils import constants
from privkit.data import LocationData
from privkit.metrics import QualityLoss
from privkit.utils import dev_utils as du
from privkit.utils import geo_utils as gu
from privkit.ppms import PPM, PlanarLaplace
from privkit.utils.training_utils import Velocities


class VAGI(PPM):
    """
    VA-GI class to apply the mechanism

    References
    ----------
    Mendes, R., Cunha, M., & Vilela, J. P. (2023, April). Velocity-Aware Geo-Indistinguishability. In Proceedings of
    the Thirteenth ACM Conference on Data and Application Security and Privacy (pp. 141-152).
    """

    PPM_ID = "va_gi"
    PPM_NAME = "VA-GI"
    PPM_INFO = "The velocity-aware geo-indistinguishability uses the Planar Laplace mechanism as baseline, " \
               "but dynamically adapts the privacy parameter epsilon according to the user velocities as well as the " \
               "reporting speed, obtained by performing the Kernel Density Function on the user and report velocities. "
    PPM_REF = "Mendes, R., Cunha, M., & Vilela, J. P. (2023, April). Velocity-Aware Geo-Indistinguishability. In " \
              "Proceedings of the Thirteenth ACM Conference on Data and Application Security and Privacy (pp. " \
              "141-152). "
    DATA_TYPE_ID = [LocationData.DATA_TYPE_ID]
    METRIC_ID = [QualityLoss.METRIC_ID]

    def __init__(self, epsilon: float, m: float):
        """
        Initializes the VA-GI mechanism by defining the privacy parameter epsilon as well as a tweaker of privacy and
        utility.

        :param float epsilon: privacy parameter
        :param float m: constant used to adjust the privacy and utility bounds
        """

        super().__init__()
        self.epsilon = epsilon
        self.m = m

    def execute(self, location_data: LocationData, train_data: pd.DataFrame = None, user_velocity_max: float = None, report_velocity_max: float = None):
        """
        Executes the VA-GI mechanism to the data given as parameter.

        :param privkit.LocationData location_data: location data where the planar laplace should be executed
        :param DataFrame train_data: data which will be used to compute the velocities' distribution - if not provided VA-GI performs data division
        :param float user_velocity_max: maximum user velocity to consider
        :param float report_velocity_max: maximum report velocity to consider
        :return: location data with obfuscated latitude and longitude and the quality loss metric and epsilon array
        """

        if constants.TIMESTAMP not in location_data.data:
            du.warn(
                "There is no available timestamps in data, computing timestamps with a default time interval of 1s")
            location_data.set_timestamp(1)

        test_data = location_data.data
        if train_data is None:
            du.warn('Train_data was not provided, using location_data division.')
            test_data = location_data.get_test_data()
            train_data = location_data.get_train_data()

        train_user_vel_pd, train_report_vel_pd = Velocities(user_velocity_max, report_velocity_max).get_velocities_pd(train_data)

        epsilon_array = np.empty(test_data.index.max() + 1)
        test_user_indexes = list(
            set(map(lambda e: (test_data[constants.UID][e], test_data[constants.TID][e]), test_data.index)))

        for test_user_index_uid, test_user_index_tid in test_user_indexes:
            user_data = test_data[(test_data[constants.UID] == test_user_index_uid) & (test_data[constants.TID] == test_user_index_tid)]
            prev_x_i = None
            obf_latitude_output = []
            obf_longitude_output = []
            quality_loss_output = []
            for index, latitude, longitude, timestamp in zip(user_data.index, user_data[constants.LATITUDE],
                                                             user_data[constants.LONGITUDE],
                                                             user_data[constants.TIMESTAMP]):

                x_i = np.array([latitude, longitude, timestamp])

                if prev_x_i is not None:
                    epsilon = self.get_epsilon(x_i, prev_x_i, train_user_vel_pd, train_report_vel_pd)
                else:
                    epsilon = self.epsilon

                epsilon_array[index] = epsilon
                planar_laplace = PlanarLaplace(epsilon=epsilon)
                obf_latitude, obf_longitude, r = planar_laplace.get_obfuscated_point(latitude, longitude)
                obf_latitude_output.append(obf_latitude)
                obf_longitude_output.append(obf_longitude)
                quality_loss_output.append(r)

                prev_x_i = x_i

            location_data.data.loc[user_data.index, constants.OBF_LATITUDE] = obf_latitude_output
            location_data.data.loc[user_data.index, constants.OBF_LONGITUDE] = obf_longitude_output
            location_data.data.loc[user_data.index, QualityLoss.METRIC_ID] = quality_loss_output

        return location_data, epsilon_array

    def get_epsilon(self, x_i: [float, float, int], prev_x_i: [float, float, int], train_user_vel_pd: st.gaussian_kde,
                    train_report_vel_pd: st.gaussian_kde):
        """
        Computes the new epsilon at a certain timestamp.

        :param [float, float, int] x_i: current location
        :param [float, float, int] prev_x_i: previous location
        :param spicy.stats.gaussian_kde train_user_vel_pd: user velocity pd
        :param spicy.stats.gaussian_kde train_report_vel_pd: report velocity pd
        """

        lat1 = x_i[0]
        lon1 = x_i[1]
        lat2 = prev_x_i[0]
        lon2 = prev_x_i[1]

        distance = gu.great_circle_distance(lat1, lon1, lat2, lon2, gu.EARTH_RADIUS / 1000)
        time = (x_i[2] - prev_x_i[2]) / 3600
        vu = distance / time
        vr = 1 / time
        fu_value = self.fu(vu, train_user_vel_pd)
        fr_value = self.fr(vr, train_report_vel_pd)
        f_vu_vr = 0.5 * (fu_value + fr_value)

        a = self.epsilon / self.m
        b = self.m ** (2 * f_vu_vr)
        new_epsilon = a * b

        return new_epsilon

    def fu(self, vu: float, user_vel_pd: st.gaussian_kde):
        """
        Computes Cumulative Distribution Function of the user velocity distribution evaluated with the value of the user
        velocity.

        :param float vu: user velocity
        :param st.gaussian_kde user_vel_pd: user velocity pd
        :return:
        """
        return self.cdf(user_vel_pd, vu)

    def fr(self, vr: float, report_vel_pd: st.gaussian_kde):
        """
        Computes the inverse of the Cumulative Distribution Function of the report velocity distribution evaluated with the
        value of the report velocity.

        :param float vr: report velocity
        :param st.gaussian_kde report_vel_pd: report velocity pd
        :return:
        """
        return 1 - self.cdf(report_vel_pd, vr)

    @staticmethod
    def cdf(kde: st.gaussian_kde, velocity: float):
        """
        Given a distribution and a value, computes the Cumulative Distribution Function (cdf) of that value in the
        distribution.

        :param st.gaussian_kde kde: user or report velocity pd
        :param float velocity: user or report velocity
        :returns: cdf evaluated at the value of the velocity at the distribution kde
        """

        return ndtr(np.ravel(velocity - kde.dataset) / kde.factor).mean()
