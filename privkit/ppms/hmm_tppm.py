import numpy as np
import pandas as pd
from typing import List, Tuple, Optional
from hmmlearn import hmm

from privkit.data import LocationData
from privkit.metrics import QualityLoss
from privkit.ppms import PPM, PlanarLaplace
from privkit.utils import geo_utils as gu, constants
import config


class TrajectoryPrivacyProtectionHMM(PPM):
    """
    HMM-based Trajectory Privacy Protection Method (TPPM) class to apply the mechanism

    Trains a Hidden Markov Model over discretized location cells to estimate the predictability of
    each point. The privacy budget epsilon is then allocated dynamically per point (more privacy to
    less predictable, more important points) under a sliding-window budget cap, and the obfuscated
    point is chosen among Planar Laplace candidates using the HMM transition probabilities.

    References
    ----------
    Qiu, Shuyuan & Pi, Dechang & Wang, Yanxue & Liu, Yufei. (2022). Novel trajectory privacy protection method against prediction attacks. Expert Systems with Applications. 213. 118870. 10.1016/j.eswa.2022.118870. 
    """

    PPM_ID = "hmm-tppm"
    PPM_NAME = "Trajectory Privacy Protection (HMM-based)"
    PPM_INFO = "Uses an HMM to estimate location predictability, then allocates privacy budget dynamically."
    PPM_REF = "Qiu, Shuyuan & Pi, Dechang & Wang, Yanxue & Liu, Yufei. (2022). Novel trajectory privacy protection method against prediction attacks. Expert Systems with Applications. 213. 118870. 10.1016/j.eswa.2022.118870."
    DATA_TYPE_ID = [LocationData.DATA_TYPE_ID]
    METRIC_ID = [QualityLoss.METRIC_ID]

    def __init__(self, epsilon: float, ws: int = 5, delta_e: float = None, beta1: float = 0.8, beta2: float = 0.2, candidate_k: int = 10):
        """
        Initializes the HMM-based TPPM mechanism.

        :param float epsilon: total privacy budget
        :param int ws: sliding-window size used for prediction and budget allocation
        :param float delta_e: epsilon adjustment magnitude (defaults to 0.8 * epsilon)
        :param float beta1: weight of the predictability term in the epsilon adjustment
        :param float beta2: weight of the importance term in the epsilon adjustment
        :param int candidate_k: number of Planar Laplace candidates evaluated per point
        """
        super().__init__()
        self.epsilon = epsilon
        self.ws = ws
        self.delta_e = float(delta_e) if delta_e is not None else 0.8 * float(epsilon)
        self.beta1 = beta1
        self.beta2 = beta2
        self.model = None  # trained HMM model
        self.pair_to_symbol = {}
        self.symbol_to_pair = {}
        self.candidate_k = candidate_k
        self.cell_size = 0.0005

    # Model training

    def fit(self, location_data: LocationData, n_states: Optional[int] = None, n_iter: int = 50):
        """
        Trains the HMM on the discretized trajectories.

        :param privkit.LocationData location_data: training data
        :param Optional[int] n_states: kept for API compatibility; the number of hidden states is the number of unique cells
        :param int n_iter: maximum number of Baum-Welch iterations
        """
        trajectories = location_data.get_trajectories()

        # Build discretized sequences and collect all unique cells
        all_cell_seqs = []
        lengths = []
        unique_pairs_set = set()
        for _, traj in trajectories:
            lat_cells = np.floor(traj[constants.LATITUDE].values / 0.0005).astype(int)
            lon_cells = np.floor(traj[constants.LONGITUDE].values / 0.0005).astype(int)
            snapped_lat = (lat_cells + 0.5) * self.cell_size
            snapped_lon = (lon_cells + 0.5) * self.cell_size
            cell_seq = list(zip(snapped_lat, snapped_lon))
            unique_pairs_set.update(cell_seq)
            all_cell_seqs.append(cell_seq)
            lengths.append(len(cell_seq))

        # Sort for a deterministic cell <-> symbol mapping
        unique_pairs = sorted(list(unique_pairs_set))
        self.pair_to_symbol = {tuple(p): idx for idx, p in enumerate(unique_pairs)}
        self.symbol_to_pair = {idx: tuple(p) for idx, p in enumerate(unique_pairs)}
        n_symbols = len(unique_pairs)

        # Convert sequences to compact symbol IDs
        all_symbol_seqs = []
        for seq in all_cell_seqs:
            symbols = np.array([self.pair_to_symbol[pair] for pair in seq], dtype=int)
            all_symbol_seqs.append(symbols)

        # One hidden state per unique cell
        n_components = n_symbols

        # Stack sequences for the HMM
        X = np.concatenate([s.reshape(-1, 1) for s in all_symbol_seqs], axis=0)

        # Initialize counts for start, transitions, and emissions
        A_counts = np.zeros((n_components, n_components), dtype=float)
        B_counts = np.zeros((n_components, n_symbols), dtype=float)
        pi_counts = np.zeros(n_components, dtype=float)

        for symbols in all_symbol_seqs:
            if len(symbols) == 0:
                continue
            pi_counts[symbols[0]] += 1
            for t in range(len(symbols) - 1):
                A_counts[symbols[t], symbols[t + 1]] += 1
            for sym in symbols:
                B_counts[sym, sym] += 1

        # Add small smoothing to avoid divide-by-zero
        A_counts += 1e-8
        B_counts += 1e-8
        pi_counts += 1e-8

        # Normalize into probability matrices
        A0 = (A_counts.T / A_counts.sum(axis=1)).T
        B0 = (B_counts.T / B_counts.sum(axis=1)).T
        pi0 = pi_counts / pi_counts.sum()

        # Assign the initial parameters directly (no uniform fallback / reinit)
        model = hmm.CategoricalHMM(
            n_components=n_components,
            n_iter=n_iter,
            tol=1e-4,
            init_params="",  # prevent reinit
            params="ste"     # allow updates of start, transition, emission
        )
        model.n_features = n_symbols
        model.startprob_ = pi0
        model.transmat_ = A0
        model.emissionprob_ = B0

        model.fit(X, lengths)
        self.model = model

    # Main execution

    def execute(self, location_data: LocationData):
        """
        Executes the TPPM mechanism to the data given as parameter.

        :param privkit.LocationData location_data: location data where the mechanism should be executed
        :return: location data with obfuscated latitude and longitude, the quality loss metric and predicted positions
        """
        if self.model is None:
            raise RuntimeError("HMM model is not trained. Call fit() first.")

        trajectories = location_data.get_trajectories()
        location_data.data["pred_lat"] = np.nan
        location_data.data["pred_lon"] = np.nan

        for _, trajectory in trajectories:
            eps_history: List[float] = []
            for i in range(len(trajectory)):
                obf_lat, obf_lon, quality_loss, pred_lat, pred_lon, eps_i = \
                    self.get_obfuscated_point(i, trajectory, eps_history)

                idx = trajectory.iloc[i].name
                location_data.data.loc[idx, constants.OBF_LATITUDE] = obf_lat
                location_data.data.loc[idx, constants.OBF_LONGITUDE] = obf_lon
                location_data.data.loc[idx, QualityLoss.METRIC_ID] = quality_loss
                location_data.data.loc[idx, "pred_lat"] = pred_lat
                location_data.data.loc[idx, "pred_lon"] = pred_lon

                eps_history.append(eps_i)

        return location_data

    # Per-point obfuscation

    def get_obfuscated_point(self, i: int, trajectory: pd.DataFrame, eps_history: List[float]) -> Tuple[float, float, float, float, float, float]:
        """
        Allocates the per-point epsilon and obfuscates a single point.

        :param int i: timestamp (row position within the trajectory)
        :param pd.DataFrame trajectory: trajectory with snapped coordinates (lat/lon are already cell centers)
        :param List[float] eps_history: previously assigned epsilons for this trajectory (chronological)
        :return: obfuscated lat, obfuscated lon, quality loss, predicted lat, predicted lon, assigned epsilon
        """
        curr_lat = float(trajectory[constants.LATITUDE].iloc[i])
        curr_lon = float(trajectory[constants.LONGITUDE].iloc[i])

        epsilon_r = self.epsilon / self.ws
        pred_lat, pred_lon = np.nan, np.nan
        eps_i = epsilon_r

        if i >= self.ws:
            prev_lat, prev_lon = self._get_windowed_trajectory(trajectory, i)
            pred_cell = self.predict_next_cell(prev_lat, prev_lon)
            if pred_cell is not None:
                pred_lat, pred_lon = self._cell_to_coords(pred_cell)

            pp = self.location_predictability(curr_lat, curr_lon, pred_lat, pred_lon)
            importance = self.compute_importance(curr_lat, curr_lon, prev_lat, prev_lon, pred_lat, pred_lon)

            lam = self.beta1 * pp - self.beta2 * importance
            candidate_eps = epsilon_r - lam * self.delta_e

            # Sliding-window cap: budget left after the previous (w-1) epsilons in this window
            if len(eps_history) >= (self.ws - 1):
                prev_sum = sum(eps_history[-(self.ws - 1):])
            else:
                prev_sum = sum(eps_history)
            eps_max = max(self.epsilon - prev_sum, config.min_eps)
            eps_i = min(candidate_eps, eps_max)
            eps_i = float(max(eps_i, config.min_eps))
        else:
            # Not enough previous points: still must respect the window cap
            prev_sum = sum(eps_history[-(self.ws - 1):]) if len(eps_history) >= (self.ws - 1) else sum(eps_history)
            eps_max = max(self.epsilon - prev_sum, config.min_eps)
            eps_i = float(min(epsilon_r, eps_max))
            eps_i = float(max(eps_i, config.min_eps))

        # Candidate perturbation selection (Algorithm 3) using PlanarLaplace k samples
        best_lat, best_lon, best_quality = self._select_best_candidate(curr_lat, curr_lon, eps_i)

        return best_lat, best_lon, best_quality, pred_lat, pred_lon, eps_i

    # Helper methods

    def _select_best_candidate(self, lat_p: float, lon_p: float, eps_i: float):
        """
        Generates candidate_k perturbed points and selects the one with the minimum score.
        Score = Manhattan distance if the candidate stays in the same cell, else the transition probability a_mn.
        """
        real_cell = self._coords_to_cell_single(lat_p, lon_p)
        real_sym = self.pair_to_symbol.get(real_cell, None)

        candidates = []
        for _ in range(self.candidate_k):
            obf_lat, obf_lon, _ = PlanarLaplace(eps_i).get_obfuscated_point(lat_p, lon_p)
            candidates.append((obf_lat, obf_lon))

        best_score = float("inf")
        best_point = (lat_p, lon_p)
        best_quality = 0.0

        for lat_z, lon_z in candidates:
            cand_cell = self._coords_to_cell_single(lat_z, lon_z)
            cand_sym = self.pair_to_symbol.get(cand_cell, None)

            if real_sym is not None and cand_sym is not None:
                if cand_sym == real_sym:
                    # Same cell → score = Manhattan distance
                    score = TrajectoryPrivacyProtectionHMM.manhattan_distance((lat_p, lon_p), (lat_z, lon_z))
                else:
                    # Different cell → score = transition probability a_mn
                    score = float(self.model.transmat_[real_sym, cand_sym])
            else:
                score = 1.0  # fallback if unknown

            if score < best_score:
                best_score = score
                best_point = (lat_z, lon_z)
                best_quality = gu.great_circle_distance(lat_p, lon_p, lat_z, lon_z)

        return best_point[0], best_point[1], best_quality

    def _get_windowed_trajectory(self, trajectory: pd.DataFrame, i: int):
        """Returns the lat/lon of the ws points preceding position i."""
        lat = trajectory[constants.LATITUDE].iloc[i-self.ws:i].values
        lon = trajectory[constants.LONGITUDE].iloc[i-self.ws:i].values
        return lat, lon

    def predict_next_cell(self, prev_lat, prev_lon):
        """Predicts the next cell symbol from the previous window using Viterbi decoding."""
        cell_seq = self._coords_to_cells(prev_lat, prev_lon)
        X = np.array(cell_seq).reshape(-1, 1)
        _, states = self.model.decode(X, algorithm="viterbi")
        last_state = states[-1]
        next_state = np.argmax(self.model.transmat_[last_state])
        return next_state

    def _coords_to_cells(self, lat_arr, lon_arr):
        """Maps arrays of coordinates to their compact cell-symbol IDs (nearest known cell if unseen)."""
        lat_cells = np.floor(lat_arr / self.cell_size).astype(int)
        lon_cells = np.floor(lon_arr / self.cell_size).astype(int)
        snapped_lat = (lat_cells + 0.5) * self.cell_size
        snapped_lon = (lon_cells + 0.5) * self.cell_size
        compact_ids = []
        for lat, lon in zip(snapped_lat, snapped_lon):
            pair = (lat, lon)
            if pair in self.pair_to_symbol:
                compact_ids.append(self.pair_to_symbol[pair])
            else:
                known_pairs = np.array(list(self.pair_to_symbol.keys()))
                dists = np.linalg.norm(known_pairs - np.array([lat, lon]), axis=1)
                nearest_idx = np.argmin(dists)
                compact_ids.append(self.pair_to_symbol[tuple(known_pairs[nearest_idx])])
        return np.array(compact_ids, dtype=int)

    def _cell_to_coords(self, cell_id):
        """Maps a cell symbol ID back to its cell-center coordinates."""
        if cell_id not in self.symbol_to_pair:
            lat_cell, lon_cell = next(iter(self.symbol_to_pair.values()))
        else:
            lat_cell, lon_cell = self.symbol_to_pair[cell_id]
        lat = (lat_cell + 0.5) * self.cell_size
        lon = (lon_cell + 0.5) * self.cell_size
        return lat, lon

    @staticmethod
    def location_predictability(true_lat, true_lon, pred_lat, pred_lon):
        """
        Computes predictability as 1 / (1 + Manhattan distance) between the true and predicted points.
        """
        if np.isnan(pred_lat) or np.isnan(pred_lon):
            return 0.0
        d = TrajectoryPrivacyProtectionHMM.manhattan_distance((true_lat, true_lon), (pred_lat, pred_lon))
        return 1.0 / (1.0 + d)

    @staticmethod
    def manhattan_distance(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
        """
        Computes the Manhattan distance between two lat/lon points in meters.
        """
        lat1, lon1 = p1
        lat2, lon2 = p2
        R = 6371000.0  # earth radius in meters

        # Approximate meters per degree latitude & longitude at the given latitude
        lat_rad = np.radians((lat1 + lat2) / 2.0)
        m_per_deg_lat = 111132.92 - 559.82 * np.cos(2 * lat_rad) + 1.175 * np.cos(4 * lat_rad)
        m_per_deg_lon = (np.pi / 180) * R * np.cos(lat_rad)

        dx = abs(lon1 - lon2) * m_per_deg_lon
        dy = abs(lat1 - lat2) * m_per_deg_lat
        return dx + dy

    def compute_importance(self, curr_lat, curr_lon, prev_lat, prev_lon, next_lat, next_lon):
        """
        Computes a point's importance from the turning angle between the incoming and outgoing segments.
        Returns a value in [0, 1] that is larger when the trajectory changes direction.
        """
        x_prev, y_prev = self._latlon_to_meters(np.array([prev_lat]), np.array([prev_lon]))
        x_curr, y_curr = self._latlon_to_meters(np.array([curr_lat]), np.array([curr_lon]))
        x_next, y_next = self._latlon_to_meters(np.array([next_lat]), np.array([next_lon]))
        a = np.array([x_curr[0] - x_prev[0], y_curr[0] - y_prev[0]])
        b = np.array([x_next[0] - x_curr[0], y_next[0] - y_curr[0]])
        norm_a, norm_b = np.linalg.norm(a), np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        cos_theta = np.dot(a, b) / (norm_a * norm_b)
        cos_theta = float(np.clip(cos_theta, -1.0, 1.0))
        return abs(cos_theta) if cos_theta <= 0.0 else 0.0

    def _latlon_to_meters(self, lat_arr, lon_arr):
        """Projects lat/lon arrays to a local planar (meters) frame centered on their mean."""
        R = 6371000.0
        lat0, lon0 = np.mean(lat_arr), np.mean(lon_arr)
        lat_rad, lon_rad = np.radians(lat_arr), np.radians(lon_arr)
        lat0_rad, lon0_rad = np.radians(lat0), np.radians(lon0)
        x = (lon_rad - lon0_rad) * np.cos(lat0_rad) * R
        y = (lat_rad - lat0_rad) * R
        return x, y

    def _coords_to_cell_single(self, lat, lon):
        """Snaps a single coordinate to its cell-center (lat, lon)."""
        lat_cell = int(np.floor(lat / self.cell_size))
        lon_cell = int(np.floor(lon / self.cell_size))
        return ((lat_cell + 0.5) * self.cell_size, (lon_cell + 0.5) * self.cell_size)
