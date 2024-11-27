import pandas as pd

from privkit.attacks import Attack
from privkit.data import LocationData
from privkit.utils import constants, dev_utils as du


class TopN(Attack):
    """
    The Top-N Attack is a re-identification attack, i.e, verifies if a user is re-identifiable considering the top
    locations. It first generates a map that associates users to their top locations, i.e. the N most visited locations.
    Then it verifies if the top locations for a given user is unique, i.e. a user is considered re-identifiable if no
    users share the same top.

    References
    ----------
    H. Zang and J. Bolot, “Anonymization of location data does not work: A large-scale measurement study,” in
    Proceedings of the 17th annual international conference on Mobile computing and networking, pp. 145–156, 2011.
    """

    ATTACK_ID = "topN"
    ATTACK_NAME = "TopN"
    ATTACK_INFO = "TopN is a mechanism that associates individuals to their Nth most visited locations and tries to " \
                  "re-identify them considering those locations."
    ATTACK_REF = "H. Zang and J. Bolot, “Anonymization of location data does not work: A large-scale measurement " \
                 "study, in Proceedings of the 17th annual international conference on Mobile computing and " \
                 "networking, pp. 145–156, 2011."
    DATA_TYPE_ID = [LocationData.DATA_TYPE_ID]
    METRIC_ID = [constants.RE_IDENTIFIED]

    def __init__(self, N: int, over_grid: bool = False, over_ppm: bool = False, ordered: bool = True):
        """
        Initializes the TopN attack

        :param int N: corresponds to the number of locations that the attack will associate to a user
        :param bool over_grid: flag that indicates if the attack will run over grid map's locations or over true data
        :param bool over_ppm: flag that indicates if the attack will run over LPPM's locations or over true data
        :param bool ordered: flag that indicates if the set of top N locations will be ordered
        """

        super().__init__()
        self.N = N
        self.over_grid = over_grid
        self.over_ppm = over_ppm
        self.ordered = ordered

    def execute(self, location_data: LocationData):
        """
        Executes the TopN attack

        :param privkit.LocationData location_data: data where TopN will be performed
        :return: dictionary that associates userIDs to boolean - whether the user was successfully re-identified or not
        """

        user_to_top_real = self.build_top_map(location_data, constants.LATITUDE, constants.LONGITUDE)
        user_to_top_obf = dict()  # only used if over_ppm flag is set
        if self.over_ppm:
            if {constants.OBF_LATITUDE, constants.OBF_LONGITUDE}.issubset(location_data.data.columns):
                user_to_top_obf = self.build_top_map(location_data, constants.OBF_LATITUDE, constants.OBF_LONGITUDE)
            else:
                du.warn(
                    f"Obfuscated locations <{constants.OBF_LATITUDE},{constants.OBF_LONGITUDE}> are missing, "
                    f"this attack will be applied considering the original <{constants.LATITUDE},{constants.LONGITUDE}>.")
                self.over_ppm = False

        results = dict()

        data = location_data.data
        for userID in data[constants.UID].unique():
            if not self.over_ppm:
                if userID not in user_to_top_real:
                    results[userID] = False
                    continue
                user_top = user_to_top_real[userID]
                set_size = list(user_to_top_real.values()).count(user_top)
                re_identified = set_size == 1
                results[userID] = re_identified
            else:
                if userID not in user_to_top_real or userID not in user_to_top_obf:
                    results[userID] = False
                    continue
                user_top_real = user_to_top_real[userID]
                user_top_obf = user_to_top_obf[userID]
                set_size = list(user_to_top_obf.values()).count(user_top_obf)
                re_identified = (user_top_real == user_top_obf) and (set_size == 1)
                results[userID] = re_identified

        location_data.data[constants.RE_IDENTIFIED] = location_data.data[constants.UID].map(results)
        return location_data

    def build_top_map(self, location_data: LocationData, latitude_index: str, longitude_index: str):
        """
        Computes the map that associates users to their top locations

        :param privkit.LocationData location_data: location data where the attack is being applied
        :param str latitude_index: ground-truth latitude index or obfuscated latitude index
        :param str longitude_index: ground-truth longitude index or obfuscated longitude index
        :return: dictionary that associates userIDs to their top locations
        """

        if self.over_grid:
            if not hasattr(location_data, "grid"):
                du.warn("Grid-Map is not defined. It will be defined from locations.")
                location_data.create_grid(*location_data.get_bounding_box_range(), spacing=250)  # Using default values
                location_data.filter_outside_points(*location_data.get_bounding_box_range())

        user_to_top = {}
        data = location_data.data

        for userID in data[constants.UID].unique():
            clusters = {}
            user_data = data[data[constants.UID] == userID]

            for latitude, longitude in zip(user_data[latitude_index], user_data[longitude_index]):
                if pd.isna(latitude) or pd.isna(longitude):
                    continue

                i, j = latitude, longitude
                if self.over_grid:
                    i, j = location_data.grid.get_cell_with_point_within(i, j)

                if (not self.over_grid) or (self.over_grid and i >= 0 and j >= 0):
                    if (i, j) not in clusters:
                        clusters[(i, j)] = 1
                    else:
                        clusters[(i, j)] += 1

            clusters = list(dict(reversed(sorted(clusters.items(), key=lambda e: e[1]))))
            if len(clusters) < self.N:
                continue

            user_top = [x for index, x in enumerate(clusters) if index < self.N]
            if self.ordered:
                user_top = sorted(user_top)

            user_to_top[userID] = user_top

        return user_to_top
