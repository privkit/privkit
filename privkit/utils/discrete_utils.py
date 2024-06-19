"""Discrete utility classes and methods."""

import math
import numpy as np

from privkit.utils import geo_utils as gu


class BoundingBox:
    def __init__(self, min_latitude: float, max_latitude: float, min_longitude: float, max_longitude: float):
        """
        Generates a box bounded by the min and max latitude and longitude.

        :param float min_latitude: minimum latitude coordinate
        :param float max_latitude: maximum latitude coordinate
        :param float min_longitude: minimum longitude coordinate
        :param float max_longitude: maximum longitude coordinate
        """

        self.min_latitude = min_latitude
        self.max_latitude = max_latitude
        self.min_longitude = min_longitude
        self.max_longitude = max_longitude

        self.lower_left_corner = (min_latitude, min_longitude)
        self.lower_right_corner = (min_latitude, max_longitude)
        self.upper_left_corner = (max_latitude, min_longitude)
        self.upper_right_corner = (max_latitude, max_longitude)

    def _point_within_bounding_box(self, latitude: float, longitude: float):
        """
        Verifies if a given point is in the bounding box.

        :param float latitude: point's latitude
        :param float longitude: point's latitude
        :return: True if the point is within the bound box, False otherwise
        """

        return self.lower_right_corner[0] <= latitude <= self.upper_right_corner[0] and \
               self.upper_left_corner[1] <= longitude <= self.upper_right_corner[1]

    def get_bounding_box_array(self):
        """
        Returns the latitudes and longitudes which bound the bounding box.
        :return: min and max latitude and longitude which bound the bounding box
        """
        return self.min_latitude, self.max_latitude, self.min_longitude, self.max_longitude

    def get_bounding_box_corners(self):
        """
        Returns the bounding corners which bound the bounding box.
        :return: lower left and upper right corners which bound the bounding box.
        """
        return self.lower_left_corner, self.upper_right_corner


class Cell:
    def __init__(self, lower_latitude: float, upper_latitude: float, leftmost_longitude: float,
                 rightmost_longitude: float, cell_id: int):
        """
        Generates a cell characterized by the min and max latitude and longitude and assigns it an ID.

        :param float lower_latitude: minimum latitude coordinate
        :param float upper_latitude: maximum latitude coordinate
        :param float leftmost_longitude: minimum longitude coordinate
        :param float rightmost_longitude: maximum longitude coordinate
        :param int cell_id: ID to assign to the cell
        """

        self.lower_left_corner = (lower_latitude, leftmost_longitude)
        self.lower_right_corner = (lower_latitude, rightmost_longitude)
        self.upper_left_corner = (upper_latitude, leftmost_longitude)
        self.upper_right_corner = (upper_latitude, rightmost_longitude)
        self.center = ((upper_latitude + lower_latitude) / 2, (rightmost_longitude + leftmost_longitude) / 2)

        self.id = cell_id

    def get_width_height(self):
        """
        Computes the dimensions of the cell.
        :return: width and height of the cell
        """
        width = gu.great_circle_distance(self.upper_left_corner[0], self.upper_left_corner[1],
                                         self.upper_right_corner[0], self.upper_right_corner[1])
        height = gu.great_circle_distance(self.upper_left_corner[0], self.upper_left_corner[1],
                                          self.lower_left_corner[0], self.lower_left_corner[0])
        return width, height


class GridMap:
    def __init__(self, min_lat: float, max_lat: float, min_lon: float, max_lon: float, spacing: float):
        """
        Discretizes the space defined by the min and max latitude and longitude as a grid. Spacing is the distance in
        meters between two subsequent horizontal or vertical centers of the grid.

        :param float min_lat: minimum latitude coordinate
        :param float max_lat: maximum latitude coordinate
        :param float min_lon: minimum longitude coordinate
        :param float max_lon: maximum longitude coordinate
        :param float spacing: grid cell spacing
        """

        self.cells = []
        self.spacing = spacing
        self.__create_grid_by_spacing(min_lat, max_lat, min_lon, max_lon, spacing)

    def __create_grid_by_spacing(self, min_lat: float, max_lat: float, min_lon: float, max_lon: float, spacing: float):
        """
        Discretizes the space defined by the min and max latitude and longitude of the location data by cell spacing.

        :param float min_lat: minimum latitude coordinate
        :param float max_lat: maximum latitude coordinate
        :param float min_lon: minimum longitude coordinate
        :param float max_lon: maximum longitude coordinate
        :param float spacing: grid cell spacing
        """

        # Geodesic distance
        nr_horizontal_cells = math.ceil(gu.great_circle_distance(max_lat, min_lon, max_lat, max_lon) / spacing)
        nr_vertical_cells = math.ceil(gu.great_circle_distance(max_lat, min_lon, min_lat, min_lon) / spacing)
        lon_cell_length = (max_lon - min_lon) / nr_horizontal_cells
        lat_cell_length = (max_lat - min_lat) / nr_vertical_cells

        # nr_vertical_cells = math.ceil((max_lat - min_lat)/spacing)
        # nr_horizontal_cells = math.ceil((max_lon - min_lon)/spacing)

        # Top left corner is used as reference.
        cur_lat = max_lat
        cur_row = 0
        cell_nr = 1  # locationstamps start at 1 -- required by LPM^2 (bayesian inference)
        while cur_lat > min_lat:
            self.cells.append([])
            cur_lon = min_lon  # reset longitude
            while cur_lon < max_lon:
                self.cells[cur_row].append(
                    Cell(cur_lat - lat_cell_length, cur_lat, cur_lon, cur_lon + lon_cell_length, cell_nr))
                cur_lon += lon_cell_length
                cell_nr += 1

            cur_lat -= lat_cell_length  # latitude decreases as we start on the top
            cur_row += 1

        self.lon_cell_length = lon_cell_length
        self.lat_cell_length = lat_cell_length

        self.upper_left_corner = self.cells[0][0].upper_left_corner
        self.upper_right_corner = self.cells[0][-1].upper_right_corner
        self.lower_left_corner = self.cells[-1][0].lower_left_corner
        self.lower_right_corner = self.cells[-1][-1].lower_right_corner

    def get_size(self):
        """
        Computes the number of vertical and horizontal cells.
        :return: list containing the number of vertical and horizontal cells
        """
        return [len(self.cells), len(self.cells[0])]

    def get_corners(self):
        """
        Computes min and max latitude and longitude which bound the grid.
        :return: list of min and max latitude and longitude of the grid
        """
        # min_lat, max_lat, min_lon, max_lon
        return [self.lower_right_corner[0], self.upper_right_corner[0], self.upper_left_corner[1],
                self.upper_right_corner[1]]

    def get_cartesian_cell(self, i: int, j: int):
        """
        Computes the center of the respective grid map cell in cartesian coordinates.

        :param int i: row position of the cell
        :param int j: column position of the cell
        :return: center of the cell in cartesian coordinates
        """

        center = self.cells[i][j].center
        center_cartesian = gu.geodetic2cartesian(center[0], center[1], 6371009 / 1000)
        return center_cartesian

    def get_cartesian_cells(self):
        """
        Builds a matrix of the same size as the grid where each element represents the center of the respective grid map
        cell in cartesian coordinates.

        :return: matrix of the centers of the grid map in cartesian coordinate system
        """

        cartesian_cells = []
        size = self.get_size()

        for i in range(size[0]):
            cartesian_cells.append([])
            for j in range(size[1]):
                center_cartesian = self.get_cartesian_cell(i, j)
                cartesian_cells[i].append(np.array(list(center_cartesian)))

        return cartesian_cells

    def _print_summary_statistics(self):
        """
        Returns useful information about the grid.
        :return: string containing the size of the grid as well as the bounding corners
        """

        return "Grid of {}x{} cells starting at {} and finishing at {}".format(len(self.cells), len(self.cells[0]),
                                                                               self.upper_left_corner,
                                                                               self.lower_right_corner)

    def get_locationstamp(self, latitude: float, longitude: float):
        """
        Given a point finds in which cell it is contained.

        :param latitude: point's latitude
        :param longitude: point's longitude
        :return: cell ID where point is contained
        """

        if not self.point_within_grid(latitude, longitude):
            raise ValueError(f"The point ({latitude}, {longitude}) is not within the grid.")

        for row in self.cells:
            row_lower_latitude = row[0].lower_left_corner[0]
            if latitude < row_lower_latitude:
                continue
            else:  # Now we navigate the columns
                for cell in row:
                    if longitude <= cell.upper_right_corner[1]:
                        return cell.id

        raise RuntimeError('Grid error')

    def point_within_grid(self, latitude: float, longitude: float):
        """
        Given a point verifies if it is within the grid.

        :param float latitude: point's latitude
        :param float longitude: point's longitude
        :return: True if the point is within the grid, False otherwise
        """

        return self.lower_right_corner[0] <= latitude <= self.upper_right_corner[0] and \
               self.upper_left_corner[1] <= longitude <= self.upper_right_corner[1]

    def get_cell_with_point_within(self, latitude: float, longitude: float):
        """
        Computes the cell indexes where the given point is contained.

        :param float latitude: location latitude
        :param float longitude: location longitude
        :return: indexes of the cell where point is contained
        """

        if not self.point_within_grid(latitude, longitude):
            return -1, -1

        latitude_offset = int(np.floor((self.upper_left_corner[0] - latitude) / self.lat_cell_length))
        longitude_offset = int(np.floor((longitude - self.upper_left_corner[1]) / self.lon_cell_length))

        return latitude_offset, longitude_offset

    def export_grid(self, output_file_path: str):
        """
        Exports the grid to the file specified by :output_file_path: in CSV format, where each line is the ordered
        centers of the cells as: <latitude>, <longitude>

        :param output_file_path: path where grip will be exported to
        """

        with open(output_file_path, 'w') as output_file:
            for row in self.cells:
                for cell in row:
                    output_file.write("{}, {}\n".format(cell.center[0], cell.center[1]))
