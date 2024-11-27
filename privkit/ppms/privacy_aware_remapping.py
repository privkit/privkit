import numpy as np
import pandas as pd

from privkit.data import LocationData
from privkit.metrics import QualityLoss
from privkit.ppms import PPM, PlanarLaplace
from privkit.utils import (
    constants,
    GridMap,
    geo_utils as gu,
    dev_utils as du
)


class PrivacyAwareRemapping(PPM):
    """
    Privacy-Aware Remapping class to apply the mechanism

    References
    ----------
    Guilherme Duarte, Mariana Cunha, and João P. Vilela. 2024. A Privacy-Aware Remapping
    Mechanism for Location Data. In The 39th ACM/SIGAPP Symposium on Applied Computing (Avila, Spain) (SAC ’24).
    Association for Computing Machinery, New York, NY, USA, 8 pages.
    """
    PPM_ID = "privacy_aware_remapping"
    PPM_NAME = "Privacy-Aware Remapping"
    PPM_INFO = "The Privacy-Aware Remapping Mechanism takes in location data which had the Planar Laplace mechanism  " \
               "applied a priori. For each cell on the grid discretization of the data, it finds the cell which " \
               "minimizes the weighted quality loss of the distance between the ground-truth locations and the " \
               "obfuscated report. The cell must be at a distance bounded by a function between the privacy parameter " \
               "of PL mechanism and the cell's spacing, in order to take into account the noise added by Planar " \
               "Laplace. The new obfuscated locations are then given by the center of the cell which the original " \
               "cell got remapped into."
    PPM_REF = "Guilherme Duarte, Mariana Cunha, and João P. Vilela. 2024. A Privacy-Aware Remapping Mechanism for " \
              "Location Data. In The 39th ACM/SIGAPP Symposium on Applied Computing (Avila, Spain) (SAC ’24). " \
              "Association for Computing Machinery, New York, NY, USA, 8 pages."
    DATA_TYPE_ID = [LocationData.DATA_TYPE_ID]
    METRIC_ID = [QualityLoss.METRIC_ID]

    def __init__(self, epsilon: float, r: float = None, p: float = 0.95, search_space: str = 'circle'):
        """
        Initializes the Privacy-Aware Remapping mechanism by defining the privacy parameter epsilon

        :param float epsilon: privacy parameter
        :param float r: obfuscation radius. By default, if r = None, r will be computed as r = lambertw^(-1)(p, epsilon) + s/√2.
        :param float p: probability of the obfuscated location falling within the radius r. By default, r = 0.95.
        :param str search_space: flag which indicates which search space to consider around each cell, 'circle' for the
                             circle with radius r or 'square' for the square in which the circle is inscribed. Use
                            'square' for a faster runtime.
        """

        self.epsilon = epsilon
        if r is None:
            if p == 0.95 or p < 0 or p > 1:  # check the range of p
                du.warn("The default r will be considered by computing r = lambertw^(-1)(p, epsilon) + s/√2 with a default probability p = 0.95.")
            else:
                du.warn(f"The default r will be considered by computing r = lambertw^(-1)(p, epsilon) + s/√2 with a probability defined by user as {p}.")
        self.r = r
        self.p = p
        self.search_space = search_space

    def execute(self, location_data: LocationData, apply_pl: bool = True):
        """
        Applies the Remapping Mechanism to the obfuscated data given as parameter

        :param privkit.LocationData location_data: location data where the Privacy-Aware Remap should be applied
        :param bool apply_pl: boolean to specify if Planar Laplace should be applied or if it is applied a priori. By default, it is True.
        :return: location data with remapped latitude and longitude through the Remapping function which minimizes the
                 weighted quality loss
        """

        if not hasattr(location_data, "grid"):
            du.warn("Grid-Map is not defined. It will be defined from locations.")
            location_data.create_grid(*location_data.get_bounding_box_range(), spacing=2 / self.epsilon)
            location_data.filter_outside_points(*location_data.get_bounding_box_range())

        grid = location_data.grid
        grid_size = grid.get_size()

        if apply_pl or (not {constants.OBF_LATITUDE, constants.OBF_LONGITUDE}.issubset(location_data.data.columns)):
            location_data = PlanarLaplace(epsilon=self.epsilon).execute(location_data)

        R = self.get_remapping_function(location_data)

        gop = np.vectorize(self.apply_remapping_function_to_point, excluded=['R', 'grid', 'grid_size'], otypes=['O'])
        values = gop(obf_latitude=location_data.data[constants.OBF_LATITUDE],
                     obf_longitude=location_data.data[constants.OBF_LONGITUDE],
                     R=R, grid=grid, grid_size=grid_size)
        obf_data = pd.DataFrame(np.row_stack(values), index=location_data.data.index)
        location_data.data[[constants.OBF_LATITUDE, constants.OBF_LONGITUDE]] = obf_data

        return location_data

    @staticmethod
    def apply_remapping_function_to_point(obf_latitude: float, obf_longitude: float, R: dict, grid: GridMap,
                                          grid_size: [int, int]):
        """
        Applies the R remapping function to the cell where the given point is contained.
        :param obf_latitude: obfuscated latitude
        :param obf_longitude: obfuscated longitude
        :param R: Remapping function
        :param grid: grid discretization
        :param grid_size: size of the grid
        :returns: updated obfuscated location
        """
        lat_off, lon_off = grid.get_cell_with_point_within(obf_latitude, obf_longitude)
        cell_id = lat_off * grid_size[1] + lon_off
        new_cell_id = R[cell_id]
        new_lat_off = int(int(new_cell_id) / grid_size[1])
        new_lon_off = int(int(new_cell_id) % grid_size[1])
        (remapped_obf_lat, remapped_obf_lon) = grid.cells[new_lat_off][new_lon_off].center

        return remapped_obf_lat, remapped_obf_lon

    def get_remapping_function(self, location_data: LocationData):
        """
        Computes the Remapping function to the data given as parameter

        :param privkit.LocationData location_data: location data where the Privacy-Aware Remapping should be executed
        :return: Remapping function R which maps individual cells within a grid to themselves
        """

        if not hasattr(location_data, 'grid'):
            raise Exception('Grid-Map is not defined')

        grid = location_data.grid
        spacing = location_data.grid.spacing
        grid_size = grid.get_size()
        nr_cells = grid_size[0] * grid_size[1]

        w = self.generate_cells_weights(location_data, grid_size)

        if self.r is None:
            self.r = PlanarLaplace(epsilon=self.epsilon).inverse_cumulative_gamma(p=self.p) + spacing/np.sqrt(2)

        r_ = int(np.ceil(self.r / spacing))
        r_offsets = np.linspace(-r_, r_, r_ * 2 + 1)

        gop = np.vectorize(self.get_cell_remap, excluded=['w', 'grid', 'grid_size', 'r_offsets'], otypes=['O'])
        values = gop(c=range(nr_cells), w=w, grid=grid, grid_size=grid_size, r_offsets=r_offsets)
        R = {index: value for index, value in enumerate(values)}

        return R

    def get_cell_remap(self, c: int, w: dict, grid: GridMap, grid_size: [int, int], r_offsets: [int]):
        """
        Calculates the remapped cell which minimizes the weighted quality loss of a given cell

        :param int c: current cell
        :param dict w: Map between cell ID's and respective weights
        :param GridMap grid: map discretization
        :param grid_size: grid dimensions
        :param r_offsets: search space centered at the cell c
        """

        def is_cell_outside_the_circle(p_i: int, p_j: int):
            """
            Tests if the input cell p is inside the circle of obfuscation of radius r centered at cell c
            :param int p_i: row index from the cell matrix of the cell p
            :param int p_j: column index from the cell matrix of the cell p
            :returns: True if p is inside the circle of radius r centered at c, False otherwise
            """
            p_ul = grid.cells[p_i][p_j].upper_left_corner
            p_ur = grid.cells[p_i][p_j].upper_right_corner
            p_ll = grid.cells[p_i][p_j].lower_left_corner
            p_lr = grid.cells[p_i][p_j].lower_right_corner

            return gu.great_circle_distance(c_center[0], c_center[1], p_ul[0], p_ul[1]) > self.r and \
                   gu.great_circle_distance(c_center[0], c_center[1], p_ur[0], p_ur[1]) > self.r and \
                   gu.great_circle_distance(c_center[0], c_center[1], p_ll[0], p_ll[1]) > self.r and \
                   gu.great_circle_distance(c_center[0], c_center[1], p_lr[0], p_lr[1]) > self.r

        c_i = int(c / grid_size[1])
        c_j = c % grid_size[1]
        c_center = grid.cells[c_i][c_j].center

        b_i, b_j = -1, -1
        b_error = np.inf
        for nc_i, nc_j in [(c_i + int(x), c_j + int(y)) for x in r_offsets for y in r_offsets]:
            if nc_i < 0 or nc_j < 0 or nc_i >= grid_size[0] or nc_j >= grid_size[1]:
                continue

            if self.search_space == 'circle' and is_cell_outside_the_circle(nc_i, nc_j):
                continue

            error = 0
            for t_i, t_j in [(c_i + int(x_), c_j + int(y_)) for x_ in r_offsets for y_ in r_offsets]:
                if t_i < 0 or t_j < 0 or t_i >= grid_size[0] or t_j >= grid_size[1]:
                    continue

                if self.search_space == 'circle' and is_cell_outside_the_circle(t_i, t_j):
                    continue

                tc = t_i * grid_size[1] + t_j
                error += w[tc] * np.sqrt((nc_i - t_i) ** 2 + (nc_j - t_j) ** 2)

            if error < b_error:
                b_error = error
                b_i, b_j = nc_i, nc_j

        nc = b_i * grid_size[1] + b_j

        return nc

    @staticmethod
    def generate_cells_weights(location_data, grid_size: [int, int]):
        """
        Creates a map which associates a cell ID from a grid map to the number of reports which lay on the cell

        :param privkit.LocationData location_data: location data where cells reports will get counted
        :param grid_size: number of rows and columns on the grid
        :return: Dictionary which maps cell's ID into their weights
        """

        w = {}

        for i in range(grid_size[0] * grid_size[1]):
            w[i] = 0

        for userID in location_data.data[constants.UID].unique():
            user_data = location_data.data[location_data.data[constants.UID] == userID]

            for index, lat, lon in zip(user_data.index, user_data[constants.LATITUDE], user_data[constants.LONGITUDE]):
                i, j = location_data.grid.get_cell_with_point_within(lat, lon)
                if i == -1 or j == -1:
                    continue

                cell_id = grid_size[1] * i + j

                w[cell_id] += 1

        return w
