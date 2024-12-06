import numpy as np
import osmnx as ox
import networkx as nx

from typing import List
from scipy.stats import expon
from math import pi, sqrt, pow, exp

from privkit.attacks import Attack
from privkit.data import LocationData
from privkit.metrics import AdversaryError, F1ScoreMM
from privkit.utils import (
    geo_utils as gu,
    constants,
    dev_utils as du
)


class MapMatching(Attack):
    """
    Class to execute a Map-Matching attack based on papers [1,2].

    References
    ----------
    [1] Newson, P., & Krumm, J. (2009, November). Hidden Markov map matching through noise and sparseness. In
    Proceedings of the 17th ACM SIGSPATIAL international conference on advances in geographic information systems
    (pp. 336-343).
    [2] Jagadeesh, G. R., & Srikanthan, T. (2017). Online map-matching of noisy and sparse location data with hidden
    Markov and route choice models. IEEE Transactions on Intelligent Transportation Systems, 18(9), 2423-2434.
    """
    ATTACK_ID = "map_matching"
    ATTACK_NAME = "Map-Matching"
    ATTACK_INFO = "Map-Matching is a mechanism that allows to continuously identify the position of a vehicle on a " \
                  "road network, given noisy location readings."
    ATTACK_REF = "Jagadeesh, G. R., & Srikanthan, T. (2017). Online map-matching of noisy and sparse location data " \
                 "with hidden Markov and route choice models. IEEE Transactions on Intelligent Transportation " \
                 "Systems, 18(9), 2423-2434."
    DATA_TYPE_ID = [LocationData.DATA_TYPE_ID]
    METRIC_ID = [AdversaryError.METRIC_ID,
                 F1ScoreMM.METRIC_ID]

    def __init__(self, G: nx.MultiDiGraph, sigma: float = 382, measurement_errors: List = None,
                 error_range: float = None, scalar: float = 4, lambda_y: float = 0.69, lambda_z: float = 13.35):
        """
        Initializes the Map-Matching attack by defining the road network and the remaining parameters

        :param networkx.MultiDiGraph G: road network represented as a directed graph
        :param float sigma: standard deviation of the location measurement noise in meters
        :param List measurement_errors: list of the measurement errors/noise in meters to compute the sigma value
        :param float error_range: defines the range to search for states for each observation point
        :param float scalar: if error_range is not defined, compute the error range as scalar x sigma
        :param float lambda_y: multiplier of circuitousness used to compute the probability of transition
        :param float lambda_z: multiplier of temporal implausibility used to compute the probability of transition
        """
        super().__init__()

        try:
            for node_id, node_data in G.nodes(data=True):
                node_data[constants.LATITUDE] = node_data['y']
                node_data[constants.LONGITUDE] = node_data['x']
            Gp = ox.project_graph(G)
            Gp = ox.add_edge_speeds(Gp, fallback=50)
            # self.hwy_speeds = gu.compute_highway_speeds_default(Gp)
            Gp = ox.add_edge_travel_times(Gp)
            self.Gp = Gp
        except:
            raise KeyError(f"Road network is not properly configured.")

        self.sigma = sigma or 1.4826 * np.median(measurement_errors)
        self.error_range = error_range or scalar * self.sigma

        if lambda_y == 0.69 and lambda_z == 13.35:
            du.warn("Default values of lambdas are being used. These parameters should be estimated according to the "
                    "used data.")

        self.lambda_y = lambda_y
        self.lambda_z = lambda_z

    def execute(self, location_data: LocationData, estimation: bool = False, data_processing: bool = False):
        """
        Executes the Map-Matching attack. If estimation is True, it executes the estimation of lambdas parameters

        :param privkit.LocationData location_data: data where Map-Matching will be performed
        :param bool estimation: boolean to specify if it executes the lambdas estimation. By default, estimation = False
        :param bool data_processing: boolean to specify if it should be applied as a data processing mechanism. By default, it is False
        :return: (if estimation is False) data updated with the MM estimation and MM state
                 (if estimation is True) lambda_y and lambda_z are returned
        """
        if estimation:
            estimated_lambda_y = []
            estimated_lambda_z = []

        if data_processing or estimation:
            latitude_column = constants.LATITUDE
            longitude_column = constants.LONGITUDE
        else:
            latitude_column = constants.OBF_LATITUDE
            longitude_column = constants.OBF_LONGITUDE

            if not {constants.OBF_LATITUDE, constants.OBF_LONGITUDE}.issubset(location_data.data.columns):
                raise KeyError(f"Obfuscated locations <{constants.OBF_LATITUDE}, {constants.OBF_LONGITUDE}> are "
                               f"missing, this attack should be applied after applying a privacy-preserving mechanism.")

        trajectories = location_data.get_trajectories()
        for _, trajectory in trajectories:
            observations = {}
            back_pointers = {}
            joint_probability = {}
            time_of_convergence = 0
            final_solution = []
            observation_id = 1
            num_observations = len(trajectory)
            for latitude, longitude, datetime in zip(trajectory[latitude_column], trajectory[longitude_column],
                                                     trajectory[constants.DATETIME]):
                du.log(f"{observation_id}/{num_observations}")
                observation = Observation(observation_id, [latitude, longitude], datetime)
                observation.get_states(self.Gp, self.sigma, self.error_range)
                observations[observation_id] = observation

                if not estimation:
                    time_of_convergence, solution = self.viterbi(observation.id, observations, back_pointers,
                                                                 joint_probability, time_of_convergence,
                                                                 num_observations)
                    final_solution.extend(solution)
                observation_id += 1

            if estimation:
                lambdas_y, lambdas_z = self.compute_lambdas(observations)
                estimated_lambda_y.extend(lambdas_y)
                estimated_lambda_z.extend(lambdas_z)
            else:
                for i, solution_id in enumerate(final_solution, start=1):
                    state = observations[i].states[solution_id]
                    index = trajectory.iloc[i - 1].name
                    if data_processing:
                        location_data.data.loc[index, constants.GT_STATE] = state
                        location_data.data.loc[index, constants.GT_LATITUDE] = state.point[0]
                        location_data.data.loc[index, constants.GT_LONGITUDE] = state.point[1]
                    else:
                        location_data.data.loc[index, constants.MM_STATE] = state
                        location_data.data.loc[index, constants.ADV_LATITUDE] = state.point[0]
                        location_data.data.loc[index, constants.ADV_LONGITUDE] = state.point[1]
        if estimation:
            self.lambda_y, self.lambda_z = self.lambdas_estimation(estimated_lambda_y, estimated_lambda_z)
            return self.lambda_y, self.lambda_z

        return location_data

    def viterbi(self, t: int, observations: dict, back_pointers: dict, joint_probability: dict,
                time_of_convergence: int, number_of_observations: int) -> [int, List]:
        """
        Computes the viterbi algorithm

        :param int t: represents the timestamp
        :param dict observations: dictionary of the observations
        :param dict back_pointers: dictionary of back pointers
        :param dict joint_probability: dictionary of joint probability
        :param int time_of_convergence: current time of convergence
        :param int number_of_observations: total number of observations
        :return: current time of convergence and current solution
        """
        solution = []
        if t == 1:
            joint_probability[t] = {}
            back_pointers[t] = dict(zip(observations[t].states.keys(), observations[t].states.keys()))

            for state in observations[t].states.values():
                joint_probability[t][state.id] = state.emission_probability

            if number_of_observations == 1:
                max_index = max(joint_probability[t], key=joint_probability[t].get)
                time_of_convergence = t
                solution.append(max_index)
        else:
            if t not in joint_probability:
                joint_probability[t] = {}
                back_pointers[t] = {}

            for current_state in observations[t].states.values():
                max_probability = -1
                max_index = -1

                for previous_state in observations[t - 1].states.values():
                    transition_probability = self.compute_transition_probability(previous_state, current_state)

                    temp_probability = (joint_probability[t - 1][previous_state.id] * transition_probability)
                    temp_index = previous_state.id

                    if temp_probability > max_probability:
                        max_probability = temp_probability
                        max_index = temp_index

                joint_probability[t][current_state.id] = current_state.emission_probability * max_probability
                back_pointers[t][current_state.id] = max_index

            if t < number_of_observations:
                states = list(np.unique(list(back_pointers[t].values())))
                time_of_convergence, solution = self.online_viterbi_decoder(t, states, back_pointers,
                                                                            time_of_convergence)
            else:
                max_index = max(joint_probability[t], key=joint_probability[t].get)
                states = [back_pointers[t][max_index]]
                time_of_convergence, solution = self.online_viterbi_decoder(t, states, back_pointers,
                                                                            time_of_convergence, True)
                solution.append(max_index)

        return time_of_convergence, solution

    def online_viterbi_decoder(self, t: int, states: List, back_pointers: dict, time_of_convergence: int,
                               last: bool = False) -> [int, List]:
        """
        Computes the online viterbi decoder

        :param int t: represents the timestamp
        :param List states: list of states
        :param dict back_pointers: dictionary of back pointers
        :param int time_of_convergence: current time of convergence
        :param bool last: boolean that specifies if it is the last timestamp
        :return: current time of convergence and current solution
        """
        solution = []
        if not last:
            t, states = self.find_convergence_point(t - 1, states, back_pointers, time_of_convergence)
        else:
            t = t - 1

        if len(states) == 1 and (t != time_of_convergence):
            temp_t = t
            state = states[0]
            solution.append(state)
            while t > (time_of_convergence + 1):
                state = back_pointers[t][state]
                solution.append(state)
                t -= 1
            time_of_convergence = temp_t
            solution = list(reversed(solution))
        return time_of_convergence, solution

    @staticmethod
    def find_convergence_point(t: int, states: List, back_pointers: dict, time_of_convergence: int) -> [int, List]:
        """
        Finds a convergence point

        :param int t: represents the timestamp
        :param List states: list of states
        :param dict back_pointers: dictionary of back pointers
        :param int time_of_convergence: current time of convergence
        :return: timestamp and list of states
        """
        while len(states) != 1 and t > time_of_convergence:
            temp = []
            for state in states:
                temp.append(back_pointers[t][state])
            states = list(np.unique(temp))
            t -= 1
        return t, states

    def compute_transition_probability(self, previous_state: "State", current_state: "State") -> float:
        """
        Computes the transition probability between the previous state and the current state

        :param State previous_state: state of the time step t-1
        :param State current_state: state of the time step t
        :return: transition probability value
        """
        delta_t = abs((previous_state.datetime - current_state.datetime) / np.timedelta64(1, 's'))
        free_flow_travel_time, minimum_travel_time_path = gu.get_dijkstra_length_path(self.Gp,
                                                                                      previous_state.nearest_node_id,
                                                                                      current_state.nearest_node_id,
                                                                                      weight="travel_time")

        circuitousness = self.compute_circuitousness(delta_t, previous_state, current_state, minimum_travel_time_path)
        temporal_implausibility = self.compute_temporal_implausibility(delta_t, free_flow_travel_time)

        return self.lambda_y * exp(-self.lambda_y * circuitousness) * self.lambda_z * exp(
            -self.lambda_z * temporal_implausibility)

    def compute_circuitousness(self, delta_t: float, previous_state: "State", current_state: "State",
                               minimum_travel_time_path: List) -> float:
        """
        Computes the circuitousness for the optimal path between s_t-1,j and st_k

        :param float delta_t: time interval between time steps t-1 and t (in seconds)
        :param State previous_state: state of the time step t-1
        :param State current_state: state of the time step t
        :param List minimum_travel_time_path: minimum-travel-time path between the given states
        :return: circuitousness value
        """
        driving_distance = 0
        for i in range(0, len(minimum_travel_time_path) - 1):
            edge = self.Gp[minimum_travel_time_path[i]][minimum_travel_time_path[i + 1]][0]
            driving_distance += edge["length"]

        great_circle_distance = gu.great_circle_distance(previous_state.point[0], previous_state.point[1],
                                                         current_state.point[0], current_state.point[1])

        return (driving_distance - great_circle_distance) / delta_t

    @staticmethod
    def compute_temporal_implausibility(delta_t: float, free_flow_travel_time: float) -> float:
        """
        Computes the temporal implausibility for the optimal path between s_t-1,j and st_k

        :param float delta_t: time interval between time steps t-1 and t (in seconds)
        :param float free_flow_travel_time: free-flow travel time of the optimal path (in seconds)
        :return: temporal implausibility (in seconds)
        """
        return max((free_flow_travel_time - delta_t), 0) / delta_t

    def compute_lambdas(self, observations: dict) -> [List[float], List[float]]:
        """
        Computes the circuitousness and temporal implausibility to estimate lambdas parameters

        :param dict observations: Dict of the observations
        :return: estimated lambda_y and lambda_z
        """
        estimated_lambda_y = []
        estimated_lambda_z = []
        for t in range(2, len(list(observations.keys()))):
            for current_state in observations[t].states.values():
                for previous_state in observations[t - 1].states.values():
                    delta_t = abs((previous_state.datetime - current_state.datetime) / np.timedelta64(1, 's'))
                    free_flow_travel_time, minimum_travel_time_path = gu.get_dijkstra_length_path(self.Gp,
                                                                                                  previous_state.nearest_node_id,
                                                                                                  current_state.nearest_node_id,
                                                                                                  weight="travel_time")

                    circuitousness = self.compute_circuitousness(delta_t, previous_state, current_state,
                                                                 minimum_travel_time_path)
                    temporal_implausibility = self.compute_temporal_implausibility(delta_t, free_flow_travel_time)

                    estimated_lambda_y.append(circuitousness)
                    if temporal_implausibility > 0:
                        estimated_lambda_z.append(temporal_implausibility)

        return estimated_lambda_y, estimated_lambda_z

    @staticmethod
    def lambdas_estimation(estimated_lambda_y: List[float], estimated_lambda_z: List[float]) -> [float, float]:
        """
        Computes the estimation of the lambdas parameters

        :param List estimated_lambda_y: list of estimated lambda_y
        :param List estimated_lambda_z: list of estimated lambda_z
        :return: estimation of lambda_y and lambda_z
        """
        _, scale = expon.fit(estimated_lambda_y, floc=0)
        lambda_y = 1 / scale
        _, scale = expon.fit(estimated_lambda_z, floc=0)
        lambda_z = 1 / scale

        return lambda_y, lambda_z

    @staticmethod
    def sigma_estimation(location_data: LocationData) -> [float, List[float]]:
        """
        Estimates the sigma value given the location data with ground truth and test data

        :param privkit.LocationData location_data: location data used as ground truth
        :return: value of sigma and list of measurement errors between ground truth and test data
        """
        measurement_errors = []
        for i in range(0, len(location_data.data)):
            measurement_errors.append(gu.great_circle_distance(location_data.data[constants.LATITUDE][i],
                                                               location_data.data[constants.LONGITUDE][i],
                                                               location_data.data[constants.OBF_LATITUDE][i],
                                                               location_data.data[constants.OBF_LONGITUDE][i]))

        sigma = 1.4826 * np.median(measurement_errors)
        return sigma, measurement_errors


class Observation:
    """
    Observation Class represents an observation point (i.e. a location point) that should be matched in the road network
    """

    def __init__(self, observation_id: int, point: [float, float], datetime: np.datetime64):
        self.id = observation_id
        self.point = point  # [latitude, longitude]
        self.datetime = datetime
        self.states = {}

    def get_states(self, Gp: nx.MultiDiGraph, sigma: float, error_range: float):
        """
        Computes the states, i.e. the match the observation points in the road network within the defined error_range

        :param networkx.MultiDiGraph Gp: road network represented as a projected graph
        :param float sigma: standard deviation of the location measurement noise in meters
        :param float error_range: defines the range to search for states for each observation point
        """
        point_proj = gu.project2crs(self.point[0], self.point[1], Gp.graph['crs'])

        try:
            G_aux = Gp.copy()
            find_states = True
        except:
            find_states = False

        state_id = 1
        while find_states:
            edge, distance = ox.nearest_edges(G_aux, point_proj[0], point_proj[1], return_dist=True)

            start_node_id = edge[0]
            end_node_id = edge[1]

            if distance <= error_range:
                start_node = [G_aux.nodes[start_node_id][constants.LATITUDE], G_aux.nodes[start_node_id][constants.LONGITUDE]]
                dist_start_node = gu.great_circle_distance(self.point[0], self.point[1], start_node[0], start_node[1])

                end_node = [G_aux.nodes[end_node_id][constants.LATITUDE], G_aux.nodes[end_node_id][constants.LONGITUDE]]
                dist_end_node = gu.great_circle_distance(self.point[0], self.point[1], end_node[0], end_node[1])

                if dist_start_node < dist_end_node:
                    nearest_node = start_node
                    nearest_node_id = start_node_id
                else:
                    nearest_node = end_node
                    nearest_node_id = end_node_id

                state = State(state_id, nearest_node, self.datetime, nearest_node_id, distance)
                state.set_emission_probability(sigma)

                if state.emission_probability <= pow(10, -4):  # ignore emission probabilities below 10^(-4)
                    break

                self.states[state_id] = state
                state_id += 1

                try:
                    G_aux.remove_edge(start_node_id, end_node_id)
                    G_aux.remove_edge(end_node_id, start_node_id)
                except nx.NetworkXError:
                    pass
            else:
                break

        # If there is no state, the nearest node is used
        if len(self.states) == 0:
            nearest_node_id, distance = ox.nearest_nodes(Gp, point_proj[0], point_proj[1], return_dist=True)
            nearest_node = [Gp.nodes[nearest_node_id][constants.LATITUDE], Gp.nodes[nearest_node_id][constants.LONGITUDE]]

            state = State(state_id, nearest_node, self.datetime, nearest_node_id, distance)
            state.set_emission_probability(sigma)

            self.states[state_id] = state


class State:
    """
    State Class represents a state from the HMM and is related to an observation point
    """

    def __init__(self, state_id: int, point: [float, float], datetime: np.datetime64, nearest_node_id: int,
                 distance: float):
        self.id = state_id
        self.point = point  # [latitude, longitude]
        self.datetime = datetime
        self.nearest_node_id = nearest_node_id
        self.distance_to_observation = distance
        self.emission_probability = 0

    def set_emission_probability(self, sigma: float):
        self.emission_probability = (1 / (sigma * sqrt(2 * pi))) * exp(
            -(pow(self.distance_to_observation, 2) / (2 * pow(sigma, 2))))
