from privkit.data import LocationData
from privkit.metrics import Metric
from privkit.utils import geo_utils as gu, constants, plot_utils


class AdversaryError(Metric):
    METRIC_ID = "adv_error"
    METRIC_NAME = "Adversary Error"
    METRIC_INFO = "The adversary error measures the error between the data obtained by an adversary after applying " \
                  "an attack and the original data. "
    DATA_TYPE_ID = [LocationData.DATA_TYPE_ID]

    def __init__(self):
        super().__init__()
        self.values = []

    def execute(self, location_data: LocationData):
        """
        Executes the adversary error metric.

        :param privkit.LocationData location_data: data where adversary error will be computed
        :return: data with the computed metric
        """
        if self.METRIC_ID not in location_data.data.columns or location_data.data.get(self.METRIC_ID).isna().any():
            try:
                for latitude, longitude, adv_latitude, adv_longitude in zip(location_data.data[constants.LATITUDE],
                                                                            location_data.data[constants.LONGITUDE],
                                                                            location_data.data[constants.ADV_LATITUDE],
                                                                            location_data.data[constants.ADV_LONGITUDE]):
                    adv_error = gu.great_circle_distance(latitude, longitude, adv_latitude, adv_longitude)
                    self.values.append(adv_error)
                location_data.data[self.METRIC_ID] = self.values
            except Exception:
                raise KeyError("There is no available data to compute {}.".format(self.METRIC_NAME))
        else:
            self.values = location_data.data.get(self.METRIC_ID)
        self._print_statistics()

        return location_data

    def plot(self):
        """Plot adversary error metric"""
        plot_utils.boxplot(self.values, title=self.METRIC_NAME)
