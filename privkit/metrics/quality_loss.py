from privkit.data import LocationData
from privkit.metrics import Metric
from privkit.utils import (
    geo_utils as gu,
    constants,
    plot_utils
)


class QualityLoss(Metric):
    METRIC_ID = "quality_loss"
    METRIC_NAME = "Quality Loss"
    METRIC_INFO = "Quality loss measures the loss of data quality, resulting from the application of a " \
                  "privacy-preserving mechanism."
    DATA_TYPE_ID = [LocationData.DATA_TYPE_ID]

    def __init__(self):
        super().__init__()
        self.values = []

    def execute(self, location_data: LocationData):
        """
        Executes the quality loss metric.

        :param privkit.LocationData location_data: data where quality loss will be computed
        :return: data with the computed metric
        """
        if self.METRIC_ID not in location_data.data.columns or location_data.data.get(self.METRIC_ID).isna().any():
            try:
                for latitude, longitude, obf_latitude, obf_longitude in zip(location_data.data[constants.LATITUDE],
                                                                            location_data.data[constants.LONGITUDE],
                                                                            location_data.data[constants.OBF_LATITUDE],
                                                                            location_data.data[constants.OBF_LONGITUDE]):
                    quality_loss = gu.great_circle_distance(latitude, longitude, obf_latitude, obf_longitude)
                    self.values.append(quality_loss)
                location_data.data[self.METRIC_ID] = self.values
            except Exception:
                raise KeyError("There is no available data to compute {}.".format(self.METRIC_NAME))
        else:
            self.values = location_data.data.get(self.METRIC_ID)
        self._print_statistics()

        return location_data

    @staticmethod
    def get_quality_loss_point(latitude: float, longitude: float, obf_latitude: float, obf_longitude: float):
        """
        Executes the quality loss metric at a given the ground-truth and the obfuscated report.
        :param latitude: original latitude
        :param longitude: original longitude
        :param obf_latitude: obfuscated latitude
        :param obf_longitude: obfuscated longitudes
        :returns: quality loss between original and obfuscated data
        """
        return gu.great_circle_distance(latitude, longitude, obf_latitude, obf_longitude)

    def plot(self):
        """Plot quality loss metric"""
        plot_utils.boxplot(self.values, title=self.METRIC_NAME)
