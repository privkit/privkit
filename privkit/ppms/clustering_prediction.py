from privkit.data import LocationData
from privkit.metrics import QualityLoss
from privkit.ppms import PPM, PlanarLaplace
from privkit.utils import geo_utils as gu, constants


class ClusteringPrediction(PPM):
    """
    Prediction-aware Clustering Geo-indistinguishability class to apply the mechanism

    Builds on Clustering Geo-Ind: consecutive points flagged as `close_points` are
    grouped into a segment, the segment centroid is obfuscated once with Planar
    Laplace, and that single obfuscated location is assigned to every point in the
    segment. Points that are not part of a segment are obfuscated individually.

    References
    ----------
    Cunha, M., Mendes, R., & Vilela, J. P. (2019, October). Clustering Geo-Indistinguishability
    for Privacy of Continuous Location Traces. In 2019 4th International Conference on Computing, Communications and
    Security (ICCCS) (pp. 1-8). IEEE.
    """
    PPM_ID = "clustering_pred"
    PPM_NAME = "Clustering Geo-indistinguishability"
    PPM_INFO = "Prediction-aware clustering geo-indistinguishability groups consecutive predicted close points into " \
               "a single obfuscation cluster. The cluster centroid is obfuscated once with Planar Laplace and the " \
               "resulting location is shared by every point in the cluster, while remaining points are obfuscated " \
               "individually. This mechanism is suitable for continuous scenarios."
    PPM_REF = "Cunha, M., Mendes, R., & Vilela, J. P. (2019, October). Clustering Geo-Indistinguishability for " \
              "Privacy of Continuous Location Traces. In 2019 4th International Conference on Computing, " \
              "Communications and Security (ICCCS) (pp. 1-8). IEEE. "
    DATA_TYPE_ID = [LocationData.DATA_TYPE_ID]
    METRIC_ID = [QualityLoss.METRIC_ID]

    def __init__(self, epsilon: float):
        """
        Initializes the prediction-aware Clustering Geo-Ind mechanism by defining the privacy parameter epsilon

        :param float epsilon: privacy parameter
        """
        super().__init__()
        self.epsilon = float(epsilon)
        self.obfuscated_points = []

    def execute(self, location_data: LocationData):
        """
        Executes the prediction-aware Clustering Geo-Ind mechanism to the data given as parameter

        :param privkit.LocationData location_data: location data where the planar laplace should be executed
        :return: location data with obfuscated latitude and longitude and the quality loss metric
        """
        print("Executing Clustering Geo-Indistinguishability (considering predictions) on location data...")
        trajectories = location_data.get_trajectories()

        count_close = 0
        total = 0

        for _, trajectory in trajectories:
            i = 0
            n = len(trajectory)
            while i < n:
                total += 1
                row = trajectory.iloc[i]
                idx = row.name
                row_type = row.get("type", "point")  # default if column missing

                if row_type == "close_points":
                    # Collect the whole consecutive segment of close points
                    group_indices = []
                    while i < n and trajectory.iloc[i].get("type", "point") == "close_points":
                        group_indices.append(trajectory.index[i])
                        count_close += 1
                        i += 1

                    # Obfuscate the centroid of the true lat/lon in the group once
                    group_rows = trajectory.loc[group_indices]
                    centroid_lat = float(group_rows[constants.LATITUDE].mean())
                    centroid_lon = float(group_rows[constants.LONGITUDE].mean())
                    obf_lat, obf_lon, qloss = self.get_obfuscated_point(centroid_lat, centroid_lon)

                    # Write the same obfuscation to all group members; quality loss is
                    # measured against each member's own true point
                    for gidx in group_indices:
                        true_lat = float(location_data.data.at[gidx, constants.LATITUDE])
                        true_lon = float(location_data.data.at[gidx, constants.LONGITUDE])
                        qloss_g = gu.great_circle_distance(true_lat, true_lon, obf_lat, obf_lon)
                        location_data.data.at[gidx, constants.OBF_LATITUDE] = obf_lat
                        location_data.data.at[gidx, constants.OBF_LONGITUDE] = obf_lon
                        location_data.data.at[gidx, QualityLoss.METRIC_ID] = qloss_g

                else:
                    # Single-point obfuscation
                    true_lat = float(row[constants.LATITUDE])
                    true_lon = float(row[constants.LONGITUDE])
                    obf_lat, obf_lon, qloss = self.get_obfuscated_point(true_lat, true_lon)

                    location_data.data.at[idx, constants.OBF_LATITUDE] = obf_lat
                    location_data.data.at[idx, constants.OBF_LONGITUDE] = obf_lon
                    location_data.data.at[idx, QualityLoss.METRIC_ID] = qloss

                    i += 1  # advance one step for non-close rows
        return location_data

    def get_obfuscated_point(self, latitude: float, longitude: float, i: int = 0):
        """
        Returns a geo-indistinguishable single point

        :param float latitude: original latitude
        :param float longitude: original longitude
        :param int i: timestamp
        :return: obfuscated point (latitude, longitude) and distance between original and obfuscation point
        """
        planar_laplace = PlanarLaplace(self.epsilon)
        obf_latitude, obf_longitude, distance = planar_laplace.get_obfuscated_point(latitude, longitude)

        self.obfuscated_points.append([obf_latitude, obf_longitude])
        return obf_latitude, obf_longitude, distance
