from privkit.data import LocationData
from privkit.metrics import QualityLoss
from privkit.ppms import PPM, PlanarLaplace
from privkit.utils import geo_utils as gu, constants


class ClusteringGeoInd(PPM):
    """
    Clustering Geo-indistinguishability class to apply the mechanism

    References
    ----------
    Cunha, M., Mendes, R., & Vilela, J. P. (2019, October). Clustering Geo-Indistinguishability
    for Privacy of Continuous Location Traces. In 2019 4th International Conference on Computing, Communications and
    Security (ICCCS) (pp. 1-8). IEEE.
    """
    PPM_ID = "clustering_geo_ind"
    PPM_NAME = "Clustering Geo-indistinguishability"
    PPM_INFO = "Clustering geo-indistinguishability consists of creating obfuscation clusters to aggregate nearby " \
               "locations into a single obfuscated location. This obfuscated location is produced by Planar Laplace. " \
               "The parameters of this mechanism are the privacy parameter epsilon defined as ε=l/r and the " \
               "obfuscation radius r used to define epsilon. This mechanism is suitable for sporadic and continuous " \
               "scenarios."
    PPM_REF = "Cunha, M., Mendes, R., & Vilela, J. P. (2019, October). Clustering Geo-Indistinguishability for " \
              "Privacy of Continuous Location Traces. In 2019 4th International Conference on Computing, " \
              "Communications and Security (ICCCS) (pp. 1-8). IEEE. "
    DATA_TYPE_ID = [LocationData.DATA_TYPE_ID]
    METRIC_ID = [QualityLoss.METRIC_ID]

    def __init__(self, r: float, epsilon: float):
        """
        Initializes the Clustering Geo-Ind mechanism by defining the radius of obfuscation and the privacy parameter epsilon

        :param float r: radius of obfuscation
        :param float epsilon: privacy parameter
        """
        super().__init__()

        self.r = r
        self.epsilon = epsilon

        self.obfuscated_points = []

    def execute(self, location_data: LocationData):
        """
        Executes the Clustering Geo-Ind mechanism to the data given as parameter

        :param privkit.LocationData location_data: location data where the planar laplace should be executed
        :return: location data with obfuscated latitude and longitude and the quality loss metric
        """

        trajectories = location_data.get_trajectories()
        for _, trajectory in trajectories:
            i = 0
            obf_lat_output = []
            obf_lon_output = []
            quality_loss_output = []
            self.obfuscated_points = []
            for latitude, longitude in zip(trajectory[constants.LATITUDE], trajectory[constants.LONGITUDE]):
                obf_lat, obf_lon, quality_loss = self.get_obfuscated_point(latitude, longitude, i)
                obf_lat_output.append(obf_lat)
                obf_lon_output.append(obf_lon)
                quality_loss_output.append(quality_loss)
                i += 1

            location_data.data.loc[trajectory.index, constants.OBF_LATITUDE] = obf_lat_output
            location_data.data.loc[trajectory.index, constants.OBF_LONGITUDE] = obf_lon_output
            location_data.data.loc[trajectory.index, QualityLoss.METRIC_ID] = quality_loss_output

        return location_data

    def get_obfuscated_point(self, latitude: float, longitude: float, i: int = 0) -> [float, float]:
        """
        Returns a geo-indistinguishable single point

        :param float latitude: original latitude
        :param float longitude: original longitude
        :param int i: timestamp
        :return: obfuscated point (latitude, longitude) and distance between original and obfuscation point
        """
        apply_pl = False
        if i != 0:
            obf_latitude = self.obfuscated_points[i-1][0]
            obf_longitude = self.obfuscated_points[i-1][1]
            distance = gu.great_circle_distance(latitude, longitude, obf_latitude, obf_longitude)

            if distance > self.r:
                apply_pl = True
        else:
            apply_pl = True

        if apply_pl:
            planar_laplace = PlanarLaplace(self.epsilon)
            obf_latitude, obf_longitude, distance = planar_laplace.get_obfuscated_point(latitude, longitude)
            distance = gu.great_circle_distance(latitude, longitude, obf_latitude, obf_longitude)

        self.obfuscated_points.append([obf_latitude, obf_longitude])
        return obf_latitude, obf_longitude, distance
