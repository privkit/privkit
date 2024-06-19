import difflib
import osmnx as ox
import networkx as nx

from typing import List
from privkit.data import LocationData
from privkit.metrics import Metric
from privkit.utils import (
    geo_utils as gu,
    constants,
    plot_utils
)


class F1ScoreMM(Metric):
    METRIC_ID = "f1_score_mm"
    METRIC_NAME = "F1 Score MM"
    METRIC_INFO = "F1 Score can be defined as the harmonic mean between precision and recall calculated as follows: " \
                  "F1 = 2 * (precision * recall) / (precision + recall). In the context of Map-Matching (MM), " \
                  "precision is defined as precision = Length_correct / Length_matched and Recall is defined as " \
                  "recall = Length_correct / Length_truth. "
    DATA_TYPE_ID = [LocationData.DATA_TYPE_ID]

    def __init__(self):
        super().__init__()
        self.values = []

    def execute(self, location_data: LocationData, G: nx.MultiDiGraph = None):
        """
        Executes the F1 Score metric.

        :param privkit.LocationData location_data: data where F1 Score will be computed
        :param networkx.MultiDiGraph G: road network represented as a directed graph
        :return: data with the computed metric
        """
        if self.METRIC_ID not in location_data.data.columns or location_data.data.get(self.METRIC_ID).isna().any():
            try:
                if G is None:
                    raise KeyError("There is no available data to compute {}. The road network should be provided".format(self.METRIC_NAME))

                Gp = ox.project_graph(G)
                Gp = ox.add_edge_speeds(Gp)
                Gp = ox.add_edge_travel_times(Gp)

                trajectories = location_data.get_trajectories()
                for trajectory_id, trajectory in trajectories:
                    if {constants.GT_STATE, constants.MM_STATE}.issubset(trajectory.columns):
                        i = 0
                        gt_nearest_nodes = []
                        mm_nearest_nodes = []
                        for gt_state, mm_state in zip(trajectory[constants.GT_STATE],
                                                      trajectory[constants.MM_STATE]):
                            gt_nearest_nodes.append(gt_state.nearest_node_id)
                            mm_nearest_nodes.append(mm_state.nearest_node_id)

                        gt_path, gt_length = self.get_length_path(Gp, gt_nearest_nodes)
                        mm_path, mm_length = self.get_length_path(Gp, mm_nearest_nodes)

                        location_data.data.loc[trajectories.groups[trajectory_id], self.METRIC_ID] = self.compute_f1_score(Gp, gt_path, mm_path, gt_length, mm_length)
                        i += 1
                    else:
                        raise Exception
            except Exception:
                raise KeyError("There is no available data to compute {}.".format(self.METRIC_NAME))

        self.values = location_data.data.get(self.METRIC_ID)
        self._print_statistics()

        return location_data

    @staticmethod
    def get_length_path(G: nx.MultiDiGraph, nearest_nodes: List[int]):
        """
        Compute the length of the path

        :param networkx.MultiDiGraph G: road network represented as a directed graph
        :param List[int] nearest_nodes: list with the identifiers of the nearest nodes
        :return: path and length of the path
        """
        final_length = 0
        final_path = []
        for i in range(1, len(nearest_nodes)):
            start_node = nearest_nodes[i - 1]
            end_node = nearest_nodes[i]
            try:
                length, path = gu.get_dijkstra_length_path(G, start_node, end_node)
                final_path.extend(path)
            except nx.NetworkXNoPath:
                length = 0
            except nx.NodeNotFound:
                continue
            final_length += length
        return final_path, final_length

    @staticmethod
    def matches(list1: List, list2: List):
        """
        Computes where two lists match

        :param List list1: list to match
        :param List list2: list to match
        """
        while True:
            mbs = difflib.SequenceMatcher(None, list1, list2).get_matching_blocks()
            if len(mbs) == 1: break
            for i, j, n in mbs[::-1]:
                if n > 0: yield list1[i: i + n]
                del list1[i: i + n]
                del list2[j: j + n]

    def compute_f1_score(self, G: nx.MultiDiGraph, gt_path: List, mm_path: List, length_truth: int, length_matched: int):
        """
        Compute the F1 Score metric.

        :param networkx.MultiDiGraph G: road network represented as a directed graph
        :param List gt_path: ground-truth path
        :param List mm_path: map-matched path
        :param int length_truth: length of the ground-truth path
        :param int length_matched: length of the map-matched path
        :return: f1 score value
        """
        aux_gt_path = gt_path[:]
        aux_mm_path = mm_path[:]
        list_length_correct = list(self.matches(aux_mm_path, aux_gt_path))
        length_correct = 0
        path_correct = []
        for partial_correct in list_length_correct:
            if len(partial_correct) > 1:
                partial_path_correct, partial_length_correct = self.get_length_path(G, partial_correct)
                length_correct += partial_length_correct
                path_correct.extend(partial_path_correct)

        recall = 0
        precision = 0
        if length_truth != 0:
            recall = length_correct / length_truth
        if length_matched != 0:
            precision = length_correct / length_matched
        if length_correct != 0:
            f1_score = 2 * ((precision * recall) / (precision + recall))
        else:
            f1_score = 0

        return f1_score

    def plot(self):
        """Plot F1 Score metric"""
        plot_utils.boxplot(self.values, title=self.METRIC_NAME)
