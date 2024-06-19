import numpy as np
import pandas as pd

from typing import List
from sklearn.linear_model import LinearRegression

from privkit.data import LocationData
from privkit.metrics import QualityLoss
from privkit.ppms import PPM, PlanarLaplace
from privkit.utils import geo_utils as gu, constants


class AdaptiveGeoInd(PPM):
    """
    Adaptive Geo-indistinguishability class to apply the mechanism

    References
    ----------
    Al-Dhubhani, R., & Cazalas, J. M. (2018). An adaptive geo-indistinguishability mechanism
    for continuous LBS queries. Wireless Networks, 24(8), 3221-3239.
    """
    PPM_ID = "adaptive_geo_ind"
    PPM_NAME = "Adaptive Geo-indistinguishability"
    PPM_INFO = "The adaptive geo-indistinguishability uses the Planar Laplace mechanism as baseline, but dynamically " \
               "adapts the privacy parameter epsilon according to the correlation between the current and the past " \
               "locations. This correlation is measured by using a simple linear regression. The parameters of this " \
               "mechanism are the privacy parameter epsilon, two estimation error thresholds Δ1 and Δ2, and two " \
               "privacy parameter multiplication factors α and β. The thresholds define if the mechanism increases " \
               "either privacy or utility by adjusting the privacy parameter with the multiplication factors. This " \
               "mechanism is suitable for continuous scenarios."
    PPM_REF = "Al-Dhubhani, R., & Cazalas, J. M. (2018). An adaptive geo-indistinguishability mechanism for " \
              "continuous LBS queries. Wireless Networks, 24(8), 3221-3239. "
    DATA_TYPE_ID = [LocationData.DATA_TYPE_ID]
    METRIC_ID = [QualityLoss.METRIC_ID]

    def __init__(self, epsilon: float, ws: int, delta1: float = None, delta2: float = None, alpha: float = 0.1, beta: float = 5):
        """
        Initializes the Adaptive Geo-Ind mechanism by defining the required parameters

        :param float epsilon: privacy parameter
        :param int ws: window size
        :param float delta1: threshold Δ1
        :param float delta2: threshold Δ2
        :param float alpha: privacy parameter multiplication factor α
        :param float beta: privacy parameter multiplication factor β
        """
        super().__init__()
        self.epsilon = epsilon
        self.ws = ws
        self.delta1 = delta1 or 0.96 / self.epsilon
        self.delta2 = delta2 or 2.7 / self.epsilon
        self.alpha = alpha
        self.beta = beta

    def execute(self, location_data: LocationData):
        """
        Executes the Adaptive Geo-Ind mechanism to the data given as parameter

        :param privkit.LocationData location_data: location data where the planar laplace should be executed
        :return: location data with obfuscated latitude and longitude and the quality loss metric
        """

        trajectories = location_data.get_trajectories()

        for _, trajectory in trajectories:
            i = 0
            while i < len(trajectory):
                obf_lat, obf_lon, quality_loss = self.get_obfuscated_point(i, trajectory)

                index = trajectory.iloc[i].name
                location_data.data.loc[index, constants.OBF_LATITUDE] = obf_lat
                location_data.data.loc[index, constants.OBF_LONGITUDE] = obf_lon
                location_data.data.loc[index, QualityLoss.METRIC_ID] = quality_loss
                i += 1

        return location_data

    def get_obfuscated_point(self, i: int, trajectory: pd.DataFrame) -> [float, float]:
        """
        Returns a geo-indistinguishable single point

        :param int i: timestamp
        :param pd.DataFrame trajectory: trajectory to be obfuscated
        :return: obfuscated latitude, longitude, and quality loss
        """
        epsilon_i = self.epsilon

        curr_latitude = trajectory[constants.LATITUDE].iloc[i]
        curr_longitude = trajectory[constants.LONGITUDE].iloc[i]
        curr_date = trajectory[constants.DATETIME].iloc[i]

        if i >= self.ws:
            prev_latitudes, prev_longitudes, prev_dates = self._get_windowed_trajectory(trajectory, i)
            est_error = self.estimation_error(prev_dates, prev_latitudes, prev_longitudes, curr_date, curr_latitude, curr_longitude)
            if est_error < self.delta1:
                epsilon_i = self.alpha * self.epsilon
            elif est_error > self.delta2:
                epsilon_i = self.beta * self.epsilon

        planar_laplace = PlanarLaplace(epsilon_i)
        obf_latitude, obf_longitude, quality_loss = planar_laplace.get_obfuscated_point(curr_latitude, curr_longitude)

        return obf_latitude, obf_longitude, quality_loss

    def _get_windowed_trajectory(self, trajectory: pd.DataFrame, i: int):
        """
        Gets the trajectory points within the window size from the timestamp i

        :param pd.DataFrame trajectory: current trajectory
        :param int i: timestamp i
        :return: latitudes, longitudes, and dates of the points within the window size from the timestamp i
        """
        latitudes = trajectory[constants.LATITUDE].iloc[i-self.ws:i].values
        longitudes = trajectory[constants.LONGITUDE].iloc[i-self.ws:i].values
        dates = trajectory[constants.DATETIME].iloc[i-self.ws:i].values
        return latitudes, longitudes, dates

    def estimation_error(self, prev_dates: List, prev_latitudes: List, prev_longitudes: List, curr_date: np.datetime64, curr_latitude: float, curr_longitude: float):
        """
        Computes the estimation error between the current point and the previous location points

        :param List prev_dates: previous dates
        :param List prev_latitudes: previous latitudes
        :param List prev_longitudes: previous longitudes
        :param np.datetime64 curr_date: current date
        :param float curr_latitude: current latitude
        :param float curr_longitude: current longitude
        :return: estimation error
        """
        estimated_lat = self.linear_regression(prev_dates, prev_latitudes, curr_date)
        estimated_lon = self.linear_regression(prev_dates, prev_longitudes, curr_date)

        return gu.great_circle_distance(estimated_lat, estimated_lon, curr_latitude, curr_longitude)

    def linear_regression(self, x_train: List, y_train: List, x_test: np.datetime64 = None):
        """
        Computes the linear regression

        :param List x_train: data to train
        :param List y_train: data to train
        :param np.datetime64 x_test: data to test
        :return: model prediction value
        """
        model = LinearRegression()  # Create linear regression object

        x_train, x_test = self.__format_datetime_to_secs(x_train, x_test)

        x_train = self.transpose(x_train)
        y_train = self.transpose(y_train)
        x_test = self.transpose([x_test])

        # Train the model
        model.fit(x_train, y_train)

        y_pred = model.predict(x_test)

        return y_pred

    @staticmethod
    def transpose(to_transpose):
        return np.transpose([to_transpose])

    @staticmethod
    def __format_datetime_to_secs(x_train: List, x_test: np.datetime64):
        """
        Formats the datetime values to seconds

        :param List x_train: list of datetime
        :param np.datetime64 x_test: current datetime
        :return: list of datetime and datetime in seconds
        """
        if type(x_test) != np.datetime64:
            x_test = x_test.to_datetime64()

        x_test = (x_test - min(x_train)) / np.timedelta64(1, 's')
        x_train = (x_train - min(x_train)) / np.timedelta64(1, 's')

        return x_train, x_test
