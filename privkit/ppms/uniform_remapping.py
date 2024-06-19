from privkit.ppms import PPM
from privkit.data import LocationData
from privkit.metrics import QualityLoss, AdversaryError
from privkit.utils import (
    constants,
    GridMap
)


class UniformRemapping(PPM):
    """
    UniformRemapping class to perform remapping as a privacy-preserving mechanism based on reference [1].

    References
    ----------
    [1] Andrés, M. E., Bordenabe, N. E., Chatzikokolakis, K., & Palamidessi, C. (2013, November).
    Geo-indistinguishability: Differential privacy for location-based systems. In Proceedings of the 2013 ACM SIGSAC
    conference on Computer & communications security (pp. 901-914).
    """
    PPM_ID = "uniform_remapping"
    PPM_NAME = "Uniform Remapping"
    PPM_INFO = "Uniform remapping consists of discretizing location data by remapping each location to the closest " \
               "point in the discrete domain. This mechanism can be applied independently or combined with other " \
               "privacy-preserving mechanism."
    PPM_REF = "Andrés, M. E., Bordenabe, N. E., Chatzikokolakis, K., & Palamidessi, C. (2013, November). " \
              "Geo-indistinguishability: Differential privacy for location-based systems. In Proceedings of the 2013 " \
              "ACM SIGSAC conference on Computer & communications security (pp. 901-914)."
    DATA_TYPE_ID = [LocationData.DATA_TYPE_ID]
    METRIC_ID = [QualityLoss.METRIC_ID,
                 AdversaryError.METRIC_ID]

    def __init__(self, grid: GridMap = None):
        """
        Initializes the Uniform Remapping mechanism

        :param GridMap grid: map discretized into cells. If no grid map is configured, it will be considered the grid
        from LocationData class.
        """
        super().__init__()
        self.grid = grid

    def execute(self, location_data: LocationData, data_processing: bool = False, combined_ppm: bool = False):
        """
        Executes the Uniform Remapping mechanism to the data given as parameter

        :param privkit.LocationData location_data: location data where the uniform remapping will be executed
        :param bool data_processing: boolean to specify if it should be applied as a data processing mechanism. By default, it is False
        :param bool combined_ppm: boolean to specify if uniform rempaping is applied as a combined PPM. By default, it is False
        :return: location data with obfuscated latitude and longitude
        """
        if (self.grid is None) and (not hasattr(location_data, "grid")):
            location_data.create_grid(*location_data.get_bounding_box_range(), spacing=250)
            location_data.filter_outside_points(*location_data.get_bounding_box_range())

        input_latitude = constants.LATITUDE
        input_longitude = constants.LONGITUDE
        output_latitude = constants.OBF_LATITUDE
        output_longitude = constants.OBF_LONGITUDE

        if data_processing:
            output_latitude = input_latitude
            output_longitude = input_longitude

        if combined_ppm:
            input_latitude = constants.OBF_LATITUDE
            input_longitude = constants.OBF_LONGITUDE

        discrete_data = []
        for latitude, longitude in zip(location_data.data[input_latitude], location_data.data[input_longitude]):
            # Grab cell offsets
            c_i, c_j = location_data.grid.get_cell_with_point_within(latitude, longitude)
            # Get center coordinates
            cell_center = location_data.grid.cells[c_i][c_j].center
            # Discretize
            discrete_lat = cell_center[0]
            discrete_lon = cell_center[1]
            discrete_data.append([discrete_lat, discrete_lon])

        location_data.data[[output_latitude, output_longitude]] = discrete_data

        return location_data
