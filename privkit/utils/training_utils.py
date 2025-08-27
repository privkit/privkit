import numpy as np
import pandas as pd
import scipy.stats as st

from pandas import Timestamp, Timedelta

from privkit.data import LocationData
from privkit.utils import constants, GridMap
from privkit.utils import geo_utils as gu


class BuildKnowledge:
    def __init__(self):
        """
        Initializes Build Knowledge
        """

    def get_mobility_profile(self, location_data: LocationData, chosen_profile: str):
        """
        Computes the mobility profile using the chosen data

        :param privkit.LocationData location_data: location data
        :param str chosen_profile: type of mobility profile to be computed, with the training or the testing data
        :return: mobility profile of type chosen_profile
        """

        if not hasattr(location_data, "grid"):
            location_data.create_grid(*location_data.get_bounding_box_range(), spacing=250)  # Using default values
            location_data.filter_outside_points(*location_data.get_bounding_box_range())

        self.grid = location_data.grid
        data = None

        if constants.LOCATIONSTAMP not in location_data.data.columns:
            location_data.set_locationstamp()

        if chosen_profile == constants.NORM_PROF:
            data = location_data.get_train_data()[constants.LOCATIONSTAMP]

        elif chosen_profile == constants.OMNI_PROF:
            data = location_data.get_test_data()[constants.LOCATIONSTAMP]

        return self.norm_location_histogram(data.to_numpy())

    def norm_location_histogram(self, data: [int]):
        """
        Computes the histogram for mobility profile

        :param [int] data: location stamps from train or test data to generate mobility profile
        :return: mobility profile
        """

        if np.size(data, 0) < 1:
            raise Exception('Not enough data to build the histogram')

        sizes = self.grid.get_size()
        edges = np.arange(1, sizes[0]*sizes[1] + 2)

        normed_histogram = np.histogram(data, bins=edges, density=True)[0]
        return normed_histogram


class Velocities:

    def __init__(self, user_velocity_max: float = None, report_velocity_max: float = None):
        """
        Initializes Velocities

        :param float user_velocity_max: maximum user velocity to consider
        :param float report_velocity_max: maximum report velocity to consider
        """
        self.user_velocity_max = user_velocity_max
        self.report_velocity_max = report_velocity_max

    def get_velocities_pd(self, data: pd.DataFrame):
        """
        Builds the velocities distributions

        :param pandas.DataFrame data: data used to calculate the velocities
        :return: user velocity pd and report velocity pd
        """

        user_velocities = None
        report_velocities = None

        for user_id in data[constants.UID].unique():
            user_data = data[data[constants.UID] == user_id]
            dt, distance = self.process_user_data(user_data)

            if dt is None and distance is None:
                continue

            dt[dt == 0] = dt.mean()

            user_velocity = distance/dt
            report_velocity = 1/dt

            if user_velocities is None:
                user_velocities = user_velocity
                report_velocities = report_velocity

            else:
                user_velocities = np.vstack((user_velocities, user_velocity))
                report_velocities = np.vstack((report_velocities, report_velocity))

        user_velocity_pd = self.kernel_distribution(user_velocities, constants.USER)
        report_velocity_pd = self.kernel_distribution(report_velocities, constants.REPORT)

        return user_velocity_pd, report_velocity_pd

    @staticmethod
    def process_user_data(user_data: pd.DataFrame):
        """
        Computes the time interval and distance between the user data

        :param pandas.DataFrame user_data: data from a certain user
        :return: the time gap and the distance between every two consecutive report
        """

        if len(user_data) < 2:
            return None, None

        # user_data = user_data.sort_values(by=constants.DATETIME)
        user_data_size = len(user_data)

        latitudes = user_data[constants.LATITUDE].to_numpy()
        longitudes = user_data[constants.LONGITUDE].to_numpy()
        dates = ((user_data[constants.DATETIME] - Timestamp("1970-01-01")) // Timedelta('1s')).to_numpy()  # pd.timestamp -> epoch

        last_even_idx = user_data_size - user_data_size % 2 - 1
        odd_i = np.arange(1, last_even_idx + 1, 2) - 1
        even_i = np.arange(2, last_even_idx + 2, 2) - 1
        dt = dates[even_i] - dates[odd_i]

        distance = np.array([])
        for lat1, lon1, lat2, lon2 in zip(latitudes[odd_i], longitudes[odd_i], latitudes[even_i], longitudes[even_i]):
            distance = np.append(distance, gu.great_circle_distance(lat1, lon1, lat2, lon2, gu.EARTH_RADIUS/1000))

        if last_even_idx < len(latitudes):
            dt = np.append(dt, dates[-1] - dates[last_even_idx-1])
            distance = np.append(distance, gu.great_circle_distance(latitudes[last_even_idx-1], longitudes[last_even_idx-1],
                                                                    latitudes[-1], longitudes[-1], gu.EARTH_RADIUS/1000))
        dt = dt/3600
        if np.any(dates < 0) or np.any(distance < 0):
            raise Exception("Bug in get_velocities or malformed dataset")

        return np.reshape(dt, (-1, 1)), np.reshape(distance, (-1, 1))

    def kernel_distribution(self, velocities: [float], type_of_data: str):
        """
        Computes the Kernel Density Distribution of the user or report velocities

        :param [float] velocities: user or report velocities
        :param str type_of_data: indicates if it is being considered the user or report velocities
        :returns: velocity pd
        """
        if type_of_data == constants.USER:
            if self.user_velocity_max is None:
                self.user_velocity_max = np.percentile(velocities, 90)
            velocities = velocities[velocities <= self.user_velocity_max]

        elif type_of_data == constants.REPORT:
            if self.report_velocity_max is None:
                self.report_velocity_max = np.percentile(velocities, 90)
            velocities = velocities[velocities <= self.report_velocity_max]

        velocity_pd = st.gaussian_kde(velocities)

        return velocity_pd
