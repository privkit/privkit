import numpy as np
import pandas as pd
import privkit as pk
import networkx as nx

from typing import List
from pathlib import Path
from pandas.core.frame import DataFrame
from sklearn.model_selection import train_test_split

from privkit.data import DataType
from privkit.utils import (
    constants,
    io_utils,
    plot_utils,
    geo_utils,
    GridMap,
    dev_utils as du
)


class LocationData(DataType):
    """
    LocationData is a privkit.DataType to handle location data. Location data is defined by a <latitude, longitude>
    coordinates (and optionally datetime) and is stored as a Pandas DataFrame.
    """
    DATA_TYPE_ID = "location_data"
    DATA_TYPE_NAME = "Location Data"
    DATA_TYPE_INFO = "Location data can be imported through a Pandas dataframe or a read by a delimited file, " \
                     "a file-like object or an object. To be supported, data should contain at least one point (" \
                     "latitude, longitude). "

    def __init__(self, id_name: str = None):
        super().__init__()
        self.id_name = id_name or self.DATA_TYPE_ID
        """Identifier name of this location data instance"""
        self.original_data = None
        """Original location data is stored as a pd.DataFrame"""
        self.data = None
        """Location data is stored as a pd.DataFrame"""

    def load_data(self, data_to_load: DataFrame or str or Path or object,
                  latitude: str or int = constants.LATITUDE,
                  longitude: str or int = constants.LONGITUDE,
                  datetime: str or int = constants.DATETIME,
                  user_id: str or int = constants.UID,
                  trajectory_id: str or int = constants.TID,
                  save: bool = False,
                  **kwargs):
        """
        Loads location data from a pd.DataFrame or a file that can be read by a Pandas read() method.
        The definition of the parameters came from the parameters of Pandas.

        :param DataFrame or str or Path or object data_to_load: either a Pandas Dataframe, a path to a file (str or Path) or any object that can be read from pandas read() methods.
        :param str or int latitude: the position or the name of the column containing the latitude. The default is `constants.LATITUDE`.
        :param str or int longitude: the position or the name of the column containing the longitude. The default is `constants.LONGITUDE`.
        :param str or int datetime: the position or the name of the column containing the datetime. The default is `constants.DATETIME`.
        :param str or int user_id: the position or the name of the column containing the user id. The default is `constants.UID`.
        :param str or int trajectory_id: the position or the name of the column containing the trajectory id. The default is `constants.TID`.
        :param bool save: if `True`, data is saved to a file. The default is `False`.
        :param kwargs: parameters for the Pandas read methods.
        """
        original2default = {
            latitude: constants.LATITUDE,
            longitude: constants.LONGITUDE,
            datetime: constants.DATETIME,
            user_id: constants.UID,
            trajectory_id: constants.TID,
        }

        if isinstance(data_to_load, pd.DataFrame):
            tdf = data_to_load.rename(columns=original2default)
        else:
            tdf = io_utils.read_dataframe(data_to_load, unique=False, **(kwargs or {})).rename(columns=original2default)

        if self._has_location_columns(tdf):
            tdf = self._dataframe_processing(tdf)

            tdf[constants.ORIGINAL_LATITUDE] = tdf[constants.LATITUDE]
            tdf[constants.ORIGINAL_LONGITUDE] = tdf[constants.LONGITUDE]

            self.data = tdf
            self.original_data = tdf

            filename = 'original_{}'.format(self.id_name)
            if save:
                self.save_data(filename=filename)  # To save the original data

            if save:
                self.save_data()  # To save the processed data
        else:
            raise KeyError("Latitude or longitude columns are missing.")

    def save_data(self, filepath: str = constants.data_folder, filename: str = None, extension: str = 'pkl'):
        """
        Saves data to a file.

        :param str filepath: path where data should be saved.
        :param str filename: name of the file to be saved.
        :param str extension: extension of the format of how the file should be saved. The default value is `'pkl'`.
        """
        if filename is None:
            filename = self.id_name
        io_utils.write_dataframe(self.data, filepath, filename, extension)

    @staticmethod
    def _has_location_columns(df: pd.DataFrame) -> bool:
        """
        Checks if location columns are in the given dataframe.

        :param pd.Dataframe df: input dataframe
        :return: `False` if location columns are missing and `True` otherwise.
        :rtype: bool
        """
        if (constants.LATITUDE in df) and (constants.LONGITUDE in df):
            return True
        return False

    @staticmethod
    def _dataframe_processing(df: pd.DataFrame) -> pd.DataFrame:
        """
        Processes the Pandas dataframe such as a location data dataframe.

        :param pd.Dataframe df: input dataframe
        :return: processed dataframe
        :rtype: pd.DataFrame
        """
        if not pd.core.dtypes.common.is_float_dtype(df[constants.LATITUDE].dtype):
            df[constants.LATITUDE] = df[constants.LATITUDE].astype("float")
        if not pd.core.dtypes.common.is_float_dtype(df[constants.LONGITUDE].dtype):
            df[constants.LONGITUDE] = df[constants.LONGITUDE].astype("float")
        if (constants.DATETIME in df) and (not pd.core.dtypes.common.is_datetime64_any_dtype(df[constants.DATETIME].dtype)):
            df[constants.DATETIME] = pd.to_datetime(df[constants.DATETIME])

        if constants.UID not in df:
            df[constants.UID] = 1
        if constants.TID not in df:
            df[constants.TID] = 1

        return df

    def get_trajectories(self, user_id: int = None, trajectory_id: int = None):
        """
        Gets trajectories from the given location data grouped by trajectory id and/or user id.

        :param int user_id: user identifier whose trajectories should be returned.
        :param int trajectory_id: trajectory identifier to return
        :return: trajectories
        :rtype: pd.DataFrameGroupBy
        """
        temp_data = self.data
        if user_id and constants.UID in self.data.columns:
            temp_data = self.data[self.data[constants.UID] == user_id]

        if trajectory_id and constants.TID in self.data.columns:
            temp_data = self.data[self.data[constants.TID] == trajectory_id]

        if {constants.UID, constants.TID}.issubset(self.data.columns):
            trajectories = temp_data.groupby([constants.UID, constants.TID])
        elif constants.UID in self.data.columns:
            trajectories = temp_data.groupby([constants.UID])
        elif constants.TID in self.data.columns:
            trajectories = temp_data.groupby([constants.TID])
        else:
            self.data[constants.UID] = 1
            trajectories = self.data.groupby([constants.UID])

        return trajectories

    def process_data(self):
        """
        Performs location data processing.
        The `first step` consists of sorting location data by datetime (if user id and trajectory id are
        columns of the dataframe, it first sorts location data by uid and/or tid).
        """
        du.log("Starting data processing.")
        # 1st step: sorts location data by datetime
        self._order_by_datetime()
        du.log("Data processing done.")

    def mm_data_processing(self, G: nx.MultiDiGraph, sigma: float = 6.86, error_range: float = 50,
                           lambda_y: float = 0.69, lambda_z: float = 13.35):
        """
        Applies map-matching as a pre-processing method to generate the ground thruth. The default values came from
        the following papers:

        [1] `Goh, C. Y., Dauwels, J., Mitrovic, N., Asif, M. T., Oran, A., & Jaillet, P. (2012, September). Online
        map-matching based on hidden markov model for real-time traffic sensing applications. In 2012 15th
        International IEEE Conference on Intelligent Transportation Systems (pp. 776-781). IEEE.`

        [2] `Jagadeesh, G. R., & Srikanthan, T. (2017). Online map-matching of noisy and sparse location data with hidden
        Markov and route choice models. IEEE Transactions on Intelligent Transportation Systems, 18(9), 2423-2434.`

        :param networkx.MultiDiGraph G: road network represented as a directed graph
        :param float sigma: standard deviation of the location measurement noise in meters. The default value is `6.86` [1].
        :param float error_range: defines the range to search for states for each observation point. The default value is `50` [1].
        :param float lambda_y: multiplier of circuitousness used to compute the probability of transition. The default value is `0.69` [2].
        :param float lambda_z: multiplier of temporal implausibility used to compute the probability of transition. The default value is `13.35` [2].
        """
        du.log("Starting MM processing.")
        mm = pk.MapMatching(G, sigma=sigma, error_range=error_range, lambda_y=lambda_y, lambda_z=lambda_z)
        mm.execute(self, data_processing=True)
        du.log("MM processing done.")

    def generate_ground_truth(self, mechanism: str,
                              G: nx.MultiDiGraph = None, sigma: float = 6.86, error_range: float = 50,
                              lambda_y: float = 0.69, lambda_z: float = 13.35):
        """
        Generates ground truth by applying a mechanism.

        For the Map-Matching mechanism, the following references were used:

        [1] `Goh, C. Y., Dauwels, J., Mitrovic, N., Asif, M. T., Oran, A., & Jaillet, P. (2012, September). Online
        map-matching based on hidden markov model for real-time traffic sensing applications. In 2012 15th
        International IEEE Conference on Intelligent Transportation Systems (pp. 776-781). IEEE.`

        [2] `Jagadeesh, G. R., & Srikanthan, T. (2017). Online map-matching of noisy and sparse location data with hidden
        Markov and route choice models. IEEE Transactions on Intelligent Transportation Systems, 18(9), 2423-2434.`

        :param str mechanism: mechanism identifier that should be used to generate ground truth data
        :param networkx.MultiDiGraph G: road network represented as a directed graph
        :param float sigma: standard deviation of the location measurement noise in meters. The default value is `6.86` [1].
        :param float error_range: defines the range to search for states for each observation point. The default value is `50` [1].
        :param float lambda_y: multiplier of circuitousness used to compute the probability of transition. The default value is `0.69` [2].
        :param float lambda_z: multiplier of temporal implausibility used to compute the probability of transition. The default value is `13.35` [2].
        """
        du.log("Starting ground-truth generation.")
        if mechanism == pk.MapMatching.ATTACK_ID:
            if G is None:
                du.warn("A road network is required for this mechanism")
            else:
                self.mm_data_processing(G=G, sigma=sigma, error_range=error_range, lambda_y=lambda_y, lambda_z=lambda_z)

        if {constants.GT_LATITUDE, constants.GT_LONGITUDE}.issubset(self.data.columns):
            self.data[constants.LATITUDE] = self.data[constants.GT_LATITUDE]
            self.data[constants.LONGITUDE] = self.data[constants.GT_LONGITUDE]
            du.log(f"{constants.LATITUDE} and {constants.LONGITUDE} columns are now equal to {constants.GT_LATITUDE} and {constants.GT_LONGITUDE}.")
        du.log("Generation of ground-truth data done.")

    def _order_by_datetime(self):
        """
        Sorts location data by datetime (if user id and trajectory id are columns of the dataframe, it first
        sorts location data by uid and/or tid).
        """
        if constants.DATETIME not in self.data.columns:
            du.log("There is no datetime column in data.")
            return

        du.log("Ordering data by datetime.")
        if {constants.UID, constants.TID}.issubset(self.data.columns):
            self.data.sort_values(by=[constants.UID, constants.TID, constants.DATETIME], ascending=[True, True, True],
                                  inplace=True)
        elif constants.UID in self.data.columns:
            self.data.sort_values(by=[constants.UID, constants.DATETIME], ascending=[True, True], inplace=True)
        elif constants.TID in self.data.columns:
            self.data.sort_values(by=[constants.TID, constants.DATETIME], ascending=[True, True], inplace=True)
        else:
            self.data.sort_values(by=[constants.DATETIME], ascending=True, inplace=True)

        self.data.reset_index(drop=True, inplace=True)

    def filter_by_duration(self, min_duration: float = 60, max_duration: float = 2 * 3600, time_unit: str = 's'):
        """
        Filters trajectories by duration to avoid either extremely long or short trajectories

        :param float min_duration: minimum duration that the trajectory must have. The default is `60 seconds`.
        :param float max_duration: maximum duration that the trajectory must have. The default is `2 hours = 2x3600 seconds`.
        :param str time_unit: time unit (e.g. seconds, minutes or hours). The default is `s` (seconds).
        """
        if constants.DATETIME not in self.data.columns:
            du.log("There is no datetime column in data.")
            return

        du.log("Filtering by duration.")
        trajectories = self.get_trajectories()
        trajectory_indices = []
        for trajectory_id, trajectory in trajectories:
            duration = (trajectory.datetime.max() - trajectory.datetime.min()) / np.timedelta64(1, time_unit)
            if duration < min_duration or duration > max_duration:
                trajectory_indices.extend(trajectories.get_group(trajectory_id).index)
        self.data.drop(trajectory_indices, inplace=True)
        du.log("Data filtered by duration done.")

    def filter_by_distance(self, min_distance: float = 0, max_distance: float = 2000):
        """
        Filters trajectories by distance to avoid either extremely long or short trajectories

        :param float min_distance: minimum distance that the trajectory must have. The default is `0 meters`.
        :param float max_distance: maximum distance that the trajectory must have. The default is `2000 meters = 2 km`.
        """
        du.log("Filtering by distance.")
        trajectories = self.get_trajectories()
        trajectory_indices = []
        for trajectory_id, trajectory in trajectories:
            distance = geo_utils.compute_trace_distance(trajectory[constants.LATITUDE].values,
                                                        trajectory[constants.LONGITUDE].values)
            if distance < min_distance or distance > max_distance:
                trajectory_indices.extend(trajectories.get_group(trajectory_id).index)
        self.data.drop(trajectory_indices, inplace=True)
        du.log("Data filtered by distance done.")

    def filter_by_timedelta(self, timedelta: float, time_unit: str = 's'):
        """
        Filters trajectories by timedelta to avoid discontinuity between points

        :param float timedelta: defines the maximum timedelta between subsequent points
        :param str time_unit: time unit (e.g. seconds, minutes or hours). The default is `s` (seconds).
        """
        if constants.DATETIME not in self.data.columns:
            du.log("There is no datetime column in data.")
            return

        trajectories = self.get_trajectories()
        trajectory_indices = []
        for trajectory_id, trajectory in trajectories:
            timedelta_values = trajectory.datetime.diff().fillna(pd.Timedelta(seconds=0)) / np.timedelta64(1, time_unit)
            if timedelta_values[timedelta_values > timedelta].any():
                trajectory_indices.extend(trajectories.get_group(trajectory_id).index)
        self.data.drop(trajectory_indices, inplace=True)

    def filter_outside_points(self, min_latitude: float, max_latitude: float, min_longitude: float,
                              max_longitude: float):
        """
        Filters all location points that fall outside the given latitude/longitude grid/bounding-box. Note: this can produce time gaps

        :param float min_latitude: minimum latitude coordinate
        :param float max_latitude: maximum latitude coordinate
        :param float min_longitude: minimum longitude coordinate
        :param float max_longitude: maximum longitude coordinate
        """
        latitude = self.data[constants.LATITUDE]
        longitude = self.data[constants.LONGITUDE]
        indices = np.where(
            (latitude >= min_latitude) &
            (longitude >= min_longitude) &
            (latitude <= max_latitude) &
            (longitude <= max_longitude)
        )
        self.data = self.data.iloc[indices]

    def subsample_data(self, min_timedelta: float, save_subsampled_data: bool = False):
        """
        Subsamples data according to a minimum timedelta between subsequent points

        :param float min_timedelta: minimum timedelta between subsequent points
        :param bool save_subsampled_data: if `True`, subsampled data is saved to a file
        """
        if constants.DATETIME not in self.data.columns:
            du.log("There is no datetime column in data.")
            return

        subsampled_data_indices = []
        trajectories = self.get_trajectories()
        for trajectory_id, trajectory in trajectories:
            start_point_idx = 0
            end_point_idx = 1
            point_indices_to_keep = [start_point_idx]  # The first point is always kept
            while end_point_idx < len(trajectory):
                timedelta = (trajectory[constants.DATETIME].iloc[end_point_idx] - trajectory[constants.DATETIME].iloc[
                    start_point_idx]) / np.timedelta64(1, 's')
                if timedelta >= min_timedelta:
                    point_indices_to_keep.append(end_point_idx)
                    start_point_idx = end_point_idx
                end_point_idx += 1

            if len(point_indices_to_keep) > 1:  # Assuming that >=2 points make a trajectory
                subsampled_data_indices.extend(trajectory.iloc[point_indices_to_keep].index)

        self.data = self.data.loc[subsampled_data_indices]

        if save_subsampled_data:
            filename = '{}_dt{}'.format(self.id_name, int(min_timedelta))
            self.save_data(filename=filename)

    def set_timestamp(self, timestamp_interval: int):
        """
        Computes timestamps for location data
        :param int timestamp_interval: time interval
        """
        du.log(f"Computing timestamps for an interval of {timestamp_interval}s")

        if constants.DATETIME not in self.data.columns:
            du.log("There is no datetime column in data.")
            return

        min_dataset_datetime = self.data.datetime.min().to_datetime64()
        timestamp_interval = np.timedelta64(timestamp_interval, 's')
        trajectories = self.get_trajectories()

        for index_print, (trajectory_id, trajectory) in enumerate(trajectories):
            # the timestamps contain a full time interval of timestamp_interval seconds. Any point with a date in
            # that interval belongs to the respective timestamp. Timestamps start at 1.
            time_deltas = trajectory[constants.DATETIME] - min_dataset_datetime
            timestamps = (np.ceil(time_deltas / timestamp_interval) + 1).astype(int)
            self.data.loc[trajectories.groups[trajectory_id], constants.TIMESTAMP] = timestamps

        self.data[constants.TIMESTAMP] = self.data[constants.TIMESTAMP].astype(int)

    def set_locationstamp(self):
        """
        Computes locationstamp for location data
        """
        du.log(f"Computing locationstamp.")

        trajectories = self.get_trajectories()

        for index_print, (trajectory_id, trajectory) in enumerate(trajectories):
            locationstamps = []
            for latitude, longitude in zip(trajectory[constants.LATITUDE], trajectory[constants.LONGITUDE]):
                locationstamps.append(self.grid.get_locationstamp(latitude, longitude))
            self.data.loc[trajectories.groups[trajectory_id], constants.LOCATIONSTAMP] = locationstamps

        self.data[constants.LOCATIONSTAMP] = self.data[constants.LOCATIONSTAMP].astype(int)

    def create_grid(self, min_lat: float, max_lat: float, min_lon: float, max_lon: float, spacing: float, timestamp: int = None):
        """
        Discretizes the space defined by the min and max latitude and longitude of the location data

        :param float min_lat: minimum latitude coordinate
        :param float max_lat: maximum latitude coordinate
        :param float min_lon: minimum longitude coordinate
        :param float max_lon: maximum longitude coordinate
        :param float spacing: grid cell spacing in meters
        :param int timestamp: time interval
        """
        du.log(f"Creating a grid from locations with {spacing}m squared cells.")
        self.grid = GridMap(min_lat, max_lat, min_lon, max_lon, spacing=spacing)
        self.set_locationstamp()
        if timestamp:
            self.set_timestamp(timestamp)

    def compute_timedelta(self, time_unit: str = 's', boxplot: bool = False) -> List:
        """
        Computes the timedelta between the datetime of the trajectory points

        :param str time_unit: time unit (e.g. seconds, minutes or hours). The default is `s` (seconds).
        :param bool boxplot: if `True`, a boxplot is generated
        :return: list of timedelta values
        """
        timedelta_values = []
        if constants.DATETIME not in self.data.columns:
            du.log("There is no datetime column in data.")
            return timedelta_values

        trajectories = self.get_trajectories()
        for trajectory_id, trajectory in trajectories:
            timedelta_values.extend(
                trajectory.datetime.diff().fillna(pd.Timedelta(seconds=0)) / np.timedelta64(1, time_unit))

        if boxplot:
            plot_utils.boxplot(values=timedelta_values)

        return timedelta_values

    def resample_by_time(self, R: int):
        """
        Resample the location reports on the time axis to solve the non-uniformity distribution over time.
        Within each resampling interval R, calculates the center coordinates as the mean of all reports
        within that interval.

        :param int R: resampling interval in minutes
        """
        if constants.DATETIME not in self.data.columns:
            du.log("There is no datetime column in data.")
            return

        resampled_data = []
        data = self.data

        for user_id in data[constants.UID].unique():
            user_data = data[data[constants.UID] == user_id]

            start_datetime = user_data.loc[user_data.index[0]][constants.DATETIME]
            start_index = 0
            end_index = 0
            for report_index, report in user_data.iterrows():
                report_datetime = report[constants.DATETIME]
                report_index -= user_data.index[0]

                if (report_datetime - start_datetime) / np.timedelta64(1, 'm') > R:
                    sample_mean = self.resample(user_id, start_index, end_index)
                    resampled_data.append(sample_mean)

                    start_datetime = report_datetime
                    start_index = report_index

                end_index = report_index

            if start_index != end_index:
                sample_mean = self.resample(user_id, start_index, end_index)
                resampled_data.append(sample_mean)

        df = pd.DataFrame(resampled_data,
                          columns=[constants.LATITUDE, constants.LONGITUDE, constants.DATETIME, constants.UID])

        self.data = df

    def resample(self, user_id: int, start_index: int, end_index: int):
        """
        Given two indexes, calculates the center coordinates as the mean of all report within the interval given by the
        indexes.

        :param int user_id: user which reports are being resampled
        :param start_index: starting index of the time interval
        :param end_index: ending index of the time interval
        :return: mean of the reports in the range of the two given indexes
        """
        if constants.DATETIME not in self.data.columns:
            du.log("There is no datetime column in data.")
            return -1

        sample = self.data[self.data[constants.UID] == user_id].take(np.arange(start_index, end_index + 1))
        sample_mean = [sample[constants.LATITUDE].mean(), sample[constants.LONGITUDE].mean(),
                       sample[constants.DATETIME].mean(), user_id]
        return sample_mean

    # ========================= ML-related methods =========================

    def divide_data(self, test_size: float = 0.2):
        """
        Divides data into train and test
        :param test_size: size of test data
        """
        du.log(f"Dividing data into train and test data with a {test_size} ratio for testing.")
        _, _, train_indexes, test_indexes = train_test_split(self.data, self.data.index, test_size=test_size)
        self.train_indexes = np.sort(train_indexes)
        self.test_indexes = np.sort(test_indexes)
        self.test_user_indexes = list(set(map(lambda e: (self.data[constants.UID][e], self.data[constants.TID][e]), self.test_indexes)))

    def get_train_data(self):
        """
        Returns train data
        :return: train data
        """
        if not hasattr(self, "train_indexes"):
            du.warn("Data should be divided first through divide_data method.")
            self.divide_data()
        return self.data.loc[self.train_indexes]

    def get_test_data(self):
        """
        Returns test data
        :return: test data
        """
        if not hasattr(self, "test_indexes"):
            du.warn("Data should be divided first through divide_data method.")
            self.divide_data()
        return self.data.loc[self.test_indexes]

    # ========================= Statistics methods =========================

    def get_number_of_users(self):
        """
        Returns the number of users
        :return: number of users
        """
        return len(self.data[constants.UID].unique())

    def get_number_of_trajectories(self):
        """
        Returns the number of trajectories
        :return: number of trajectories
        """
        trajectories = self.get_trajectories()
        return trajectories.ngroups

    def average_update_rate(self):
        """
        Returns the average of update rate (i.e. timedelta between subsequent points)
        :return: average update rate
        """
        return np.mean(self.compute_timedelta())

    def get_datetime_range(self):
        """
        Returns the data range of datatime
        :return: minimum datetime, maximum datetime
        """
        if constants.DATETIME in self.data.columns:
            return self.data[constants.DATETIME].min(), self.data[constants.DATETIME].max()
        else:
            du.log("There is no datetime column in the data.")
            return 0, 0

    def get_bounding_box_range(self):
        """
        Returns the minimum and maximum latitude/longitude cover by data

        :return: min_latitude, max_latitude, min_longitude, max_longitude
        """
        return self.data[constants.LATITUDE].min(), self.data[constants.LATITUDE].max(), \
               self.data[constants.LONGITUDE].min(), self.data[constants.LONGITUDE].max()

    def get_trajectory_statistics(self):
        """
        Returns trajectory statistics, specifically the total number of points, trajectory with the minimum and maximum
        number of points, and the average of points per trajectory.

        :return: total_number_of_points, min_number_of_points, max_number_of_points, average_number_of_points
        """
        trajectories = self.get_trajectories()
        number_of_trajectories = self.get_number_of_trajectories()
        total_number_of_points = len(self.data)
        min_number_of_points = min([len(n) for n in trajectories.groups.values()])
        max_number_of_points = max([len(n) for n in trajectories.groups.values()])

        average_number_of_points = total_number_of_points / number_of_trajectories

        return total_number_of_points, min_number_of_points, max_number_of_points, average_number_of_points

    def print_statistics_by_user(self, dataset_name: str = None):
        """
        Prints statistics of data by user, specifying the number of trajectories, points, and other statistics per user

        :param str dataset_name: dataset name (optional). The default value is `None`.
        """
        if dataset_name is None:
            print("{} contains {} users:".format(self.DATA_TYPE_NAME, self.get_number_of_users()))
        else:
            print("Dataset {} contains {} users:".format(dataset_name, self.get_number_of_users()))

        if constants.DATETIME in self.data.columns:
            datetime = True
        else:
            datetime = False

        for user_id in self.data[constants.UID].unique():
            user_string = "User {}".format(user_id)

            trajectories = self.get_trajectories(user_id)
            number_of_trajectories = trajectories.ngroups

            user_data = self.data[self.data[constants.UID] == user_id]
            nr_points = len(user_data)

            if datetime:
                min_date = user_data[constants.DATETIME].min()
                max_date = user_data[constants.DATETIME].max()

            min_lat = user_data[constants.LATITUDE].min()
            max_lat = user_data[constants.LATITUDE].max()
            min_lon = user_data[constants.LONGITUDE].min()
            max_lon = user_data[constants.LONGITUDE].max()

            user_string += " has {} points in {} trajectories (average={})".format(nr_points, number_of_trajectories,
                                                                                   np.round(
                                                                                       nr_points / number_of_trajectories,
                                                                                       2))
            if datetime:
                user_string += " spawning from {} to {}".format(min_date, max_date)
            user_string += " covering the bounding box [{}, {}, {}, {}]".format(min_lat, max_lat, min_lon, max_lon)
            print(user_string)

    def print_data_summary(self, dataset_name: str = None):
        """
        Prints data summary, specifying the number of users, trajectories, and other statistics

        :param str dataset_name: dataset name (optional). The default value is `None`.
        """
        print("\n=======================================")
        if dataset_name is None:
            print("{} contains:".format(self.DATA_TYPE_NAME))
        else:
            print("Dataset {} contains:".format(dataset_name))

        print("#users: {}".format(self.get_number_of_users()))
        print("#trajectories: {}".format(self.get_number_of_trajectories()))

        total_nr_points, min_nr_points, max_nr_points, average_nr_points = self.get_trajectory_statistics()

        print('The total number of points is {}, with the smallest trajectory having {} points and the biggest {}'.format(
            total_nr_points, min_nr_points, max_nr_points))
        print('The average number of points per trajectory is {}'.format(np.round(average_nr_points, 2)))

        if constants.DATETIME in self.data.columns:
            print("The average of update rate is {} {}".format(np.round(self.average_update_rate(), 2), 's'))

            min_datetime, max_datetime = self.get_datetime_range()
            print("Spawning from {} till {}".format(min_datetime, max_datetime))

        min_latitude, max_latitude, min_longitude, max_longitude = self.get_bounding_box_range()
        print("Covering the bounding box [{}, {}, {}, {}]".format(min_latitude, max_latitude, min_longitude,
                                                                  max_longitude))

        print("=======================================\n")
