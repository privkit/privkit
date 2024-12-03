"""Geospatial utility methods."""

import math
import pyproj
import numpy as np
import osmnx as ox
import networkx as nx

from typing import List

EARTH_RADIUS = 6371009
""" Earth radius a constant in meters """


def euclidean_distance(x1: float, y1: float, x2: float, y2: float) -> float:
    """
    Computes the Euclidean distance between two points. For accurate
    results, use projected coordinates rather than decimal degrees.

    :param float x1: first point's x coordinate
    :param float y1: first point's y coordinate
    :param float x2: second point's x coordinate
    :param float y2: second point's y coordinate
    :return: distance from point (x1, y1) to (x2, y2) in coordinate's units
    :rtype: float
    """
    return ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5


def great_circle_distance(lat1: float, lon1: float, lat2: float, lon2: float, earth_radius: float = EARTH_RADIUS) -> float:
    """
    Computes the great-circle distance between two points' coordinates. Expects coordinates in decimal degrees.

    :param float lat1: first point's latitude coordinate
    :param float lon1: first point's longitude coordinate
    :param float lat2: second point's latitude coordinate
    :param float lon2: second point's longitude coordinate
    :param float earth_radius: radius of earth in units in which distance will be returned (default meters)
    :return: distance from point (lat1, lon1) to (lat2, lon2) in units of earth_radius (default meters)
    :rtype: float
    """
    y1, x1, y2, x2 = map(math.radians, [lat1, lon1, lat2, lon2])

    dy = y2 - y1
    dx = x2 - x1

    h = math.sin(dy / 2) ** 2 + math.cos(y1) * math.cos(y2) * math.sin(dx / 2) ** 2
    h = np.minimum(1, h)  # protect against floating point errors
    arc = 2 * np.arcsin(math.sqrt(h))

    return arc * earth_radius


def compute_trace_distance(latitudes: List, longitudes: List):
    """
    Computes trace distance in meters given the list of latitudes and longitudes.

    :param List latitudes: list of latitudes
    :param List longitudes: list of longitudes
    :return: trace distance in meters
    """
    trace_distance = 0
    for i in range(1, len(longitudes)):
        trace_distance += great_circle_distance(latitudes[i-1], longitudes[i-1], latitudes[i], longitudes[i])
    return trace_distance


def geodetic2cartesian(latitude: float, longitude: float, earth_radius: float = EARTH_RADIUS) -> [float, float, float]:
    """
    Converts geodetic coordinates to cartesian

    :param float latitude: geodetic coordinate
    :param float longitude: geodetic coordinate
    :param float earth_radius: radius of earth in units in which distance will be returned (default meters)
    :return: cartesian coordinates x, y, and z
    """
    theta = math.pi / 2 - math.radians(latitude)
    phi = math.radians(longitude)
    x = earth_radius * math.sin(theta) * math.cos(phi)
    y = earth_radius * math.sin(theta) * math.sin(phi)
    z = earth_radius * math.cos(theta)
    return x, y, z


def cartesian2geodetic(x: float, y: float, z: float) -> [float, float]:
    """
    Converts cartesian to geodetic coordinates

    :param float x: cartesian coordinate
    :param float y: cartesian coordinate
    :param float z: cartesian coordinate
    :return: geodetic coordinates latitude and longitude
    """
    r = math.sqrt(x ** 2 + y ** 2 + z ** 2)
    theta = math.asin(z / r)
    phi = math.atan2(y, x)
    latitude = math.degrees(theta)
    longitude = math.degrees(phi)
    return latitude, longitude


def compute_bearing(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Computes the bearing between two points' coordinates. Expects coordinates in decimal degrees.

    :param float lat1: first point's latitude coordinate
    :param float lon1: first point's longitude coordinate
    :param float lat2: second point's latitude coordinate
    :param float lon2: second point's longitude coordinate
    :return: bearing from point (lat1, lon1) to (lat2, lon2) in degrees
    :rtype: float
    """
    y1, x1, y2, x2 = map(math.radians, [lat1, lon1, lat2, lon2])

    bearing = math.atan2(math.cos(y2) * math.sin(x2 - x1), math.cos(y1) * math.sin(y2) - math.sin(y1) * math.cos(y2) * math.cos(x2 - x1))

    return math.degrees(bearing)


def project2crs(latitude: float, longitude: float, crs: int or str or dict or pyproj.CRS) -> [float, float]:
    """
    Projects a point (latitude, longitude) to a different CRS.

    :param float latitude: latitude to be projected
    :param float longitude: longitude to be projected
    :param crs: the new coordinate reference system (CRS)
    :type crs: int or str or dict or pyproj.CRS
    :return: projected point (x, y) in the new CRS
    """
    project = pyproj.Proj(crs)
    x, y = project(latitude=latitude, longitude=longitude)
    return x, y


# ######################## Road network utils ######################## #

def get_dijkstra_length_path(G: nx.MultiDiGraph, start_node: int, end_node: int, weight: str = 'length') -> [float, List]:
    """
    Computes the dijkstra path and respective length

    :param networkx.MultiDiGraph G: road network represented as a directed graph
    :param int start_node: id of the node where the path should start
    :param int end_node: id of the node where the path should end
    :param str weight: weight to compute the dijkstra path. The default is `'length'`.
    :return: path from the start node to the end node and respective length, computed according to the weight parameter
    """
    try:
        length, path = nx.bidirectional_dijkstra(G, start_node, end_node, weight=weight)
    except nx.NetworkXNoPath:
        length = 0
        path = []
    return length, path


def compute_highway_speeds_default(G: nx.MultiDiGraph) -> dict:
    """
    Computes highway types and typical speeds from a given graph G.

    :param networkx.MultiDiGraph G: road network represented as a directed graph
    :return: dictionary where keys are the highway types and the values are their typical speeds
    """
    edges = ox.graph_to_gdfs(G, nodes=False, fill_edge_geometry=False)
    hwy_speeds = {}
    for hwy, group in edges.groupby("highway"):
        if "speed_kph" in group:
            hwy_speeds[hwy] = np.mean(group["speed_kph"])
        else:
            hwy_speeds[hwy] = 50  # default value
    return hwy_speeds
