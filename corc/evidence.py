"""Six evidence probes for CERP-QoS conflict analysis."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import config
from .similarity import AxisSimilarity, compute_axis_similarity, leave_one_out_state


PROBE_NAMES = ("ucf", "scf", "geo_ucf", "geo_scf", "as_ucf", "as_scf")


@dataclass(frozen=True)
class ProbeResult:
    prediction: float
    fallback_prediction: float
    available: bool
    neighbor_count: int
    effective_neighbor_count: int
    similarity_sum: float
    max_similarity: float
    neighbor_qos_variance: float
    fallback_type: str


class EvidenceProbeModel:
    def __init__(
        self,
        train_matrix: np.ndarray,
        users: pd.DataFrame,
        services: pd.DataFrame,
        top_k: int = config.TOP_K_NEIGHBORS,
        min_co_rated: int = config.MIN_CO_RATED,
        sim_shrinkage: float = config.SIM_SHRINKAGE,
        leave_one_out: bool = False,
    ):
        # Training entries sit inside the matrix their own features are computed
        # from. The value never enters the neighbour aggregation (the target row
        # and column are dropped there), but it does enter the similarities, and
        # that is not negligible at low density: at 5% two services share a
        # median of 2 co-rated users, so one observation can carry half of the
        # similarity that picks their neighbours. With this flag the entry is
        # removed from the similarity state before its features are read.
        self.leave_one_out = leave_one_out
        self.train_matrix = train_matrix
        self.mask = ~np.isnan(train_matrix)
        self.users = users
        self.services = services
        self.top_k = top_k
        self.min_co_rated = min_co_rated
        self.sim_shrinkage = sim_shrinkage

        self.global_mean = float(np.nanmean(train_matrix))
        self.user_sim: AxisSimilarity = compute_axis_similarity(train_matrix, dtype=np.float64)
        self.service_sim: AxisSimilarity = compute_axis_similarity(train_matrix.T, dtype=np.float32)

        user_deg = self.mask.sum(axis=1)
        service_deg = self.mask.sum(axis=0)
        # Raw entry-level support, kept as a structure feature. Distinct from
        # `neighbor_count`, which is measured after the similarity filter.
        self.user_degree = user_deg
        self.service_degree = service_deg
        user_sum = np.nansum(train_matrix, axis=1)
        service_sum = np.nansum(train_matrix, axis=0)
        self.user_mean = np.where(user_deg > 0, user_sum / np.maximum(user_deg, 1), self.global_mean)
        self.service_mean = np.where(service_deg > 0, service_sum / np.maximum(service_deg, 1), self.global_mean)

        self.user_country = users["country"].astype("string").to_numpy()
        self.service_country = services["country"].astype("string").to_numpy()
        self.user_as = users["as_no"].astype("string").to_numpy()
        self.service_as = services["as_no"].astype("string").to_numpy()
        self.user_lat = users["lat"].to_numpy(dtype=float)
        self.user_lon = users["lon"].to_numpy(dtype=float)
        self.service_lat = services["lat"].to_numpy(dtype=float)
        self.service_lon = services["lon"].to_numpy(dtype=float)

        self._user_loo = None
        self._service_loo = None
        if leave_one_out:
            self._user_loo = leave_one_out_state(train_matrix, self.user_sim, np.float64)
            self._service_loo = leave_one_out_state(train_matrix.T, self.service_sim, np.float32)
        # Per-entry overrides, populated by `_enter_target` while leave-one-out
        # is active. None means "use the full-matrix state".
        self._u_sim = self._u_co = self._s_sim = self._s_co = None
        self._u_mean = self._s_mean = None
        self._u_deg = self._s_deg = None

    def _enter_target(self, u: int, s: int) -> None:
        if not self.leave_one_out or not self.mask[u, s]:
            self._u_sim = self._u_co = self._s_sim = self._s_co = None
            self._u_mean = self._s_mean = self._u_deg = self._s_deg = None
            return
        self._u_sim, self._u_co, self._u_mean, self._u_deg = self._user_loo.drop(u, s)
        self._s_sim, self._s_co, self._s_mean, self._s_deg = self._service_loo.drop(s, u)

    def user_support(self, u: int, s: int) -> int:
        """Observation count for the user, excluding the entry itself when training."""
        return int(self._u_deg) if self._u_deg is not None else int(self.user_degree[u])

    def service_support(self, u: int, s: int) -> int:
        return int(self._s_deg) if self._s_deg is not None else int(self.service_degree[s])

    def predict_all(self, u: int, s: int) -> dict[str, ProbeResult]:
        self._enter_target(u, s)
        return {
            "ucf": self.predict_ucf(u, s),
            "scf": self.predict_scf(u, s),
            "geo_ucf": self.predict_geo_ucf(u, s),
            "geo_scf": self.predict_geo_scf(u, s),
            "as_ucf": self.predict_as_ucf(u, s),
            "as_scf": self.predict_as_scf(u, s),
        }

    def predict_ucf(self, u: int, s: int) -> ProbeResult:
        cand, weights = self._user_candidates(u, s)
        return self._predict_user_side(u, s, cand, weights, require_context=False)

    def predict_scf(self, u: int, s: int) -> ProbeResult:
        cand, weights = self._service_candidates(u, s)
        return self._predict_service_side(u, s, cand, weights, require_context=False)

    def predict_geo_ucf(self, u: int, s: int) -> ProbeResult:
        cand, weights = self._user_candidates(u, s)
        multipliers, context_valid = self._geo_multipliers(
            u,
            cand,
            self.user_country,
            self.user_lat,
            self.user_lon,
        )
        return self._predict_user_side(u, s, cand, weights * multipliers, require_context=True, context_valid=context_valid)

    def predict_geo_scf(self, u: int, s: int) -> ProbeResult:
        cand, weights = self._service_candidates(u, s)
        multipliers, context_valid = self._geo_multipliers(
            s,
            cand,
            self.service_country,
            self.service_lat,
            self.service_lon,
        )
        return self._predict_service_side(u, s, cand, weights * multipliers, require_context=True, context_valid=context_valid)

    def predict_as_ucf(self, u: int, s: int) -> ProbeResult:
        cand, weights = self._user_candidates(u, s)
        multipliers, context_valid = self._as_multipliers(u, cand, self.user_as)
        return self._predict_user_side(u, s, cand, weights * multipliers, require_context=True, context_valid=context_valid)

    def predict_as_scf(self, u: int, s: int) -> ProbeResult:
        cand, weights = self._service_candidates(u, s)
        multipliers, context_valid = self._as_multipliers(s, cand, self.service_as)
        return self._predict_service_side(u, s, cand, weights * multipliers, require_context=True, context_valid=context_valid)

    def _user_candidates(self, u: int, s: int) -> tuple[np.ndarray, np.ndarray]:
        cand = np.nonzero(self.mask[:, s])[0]
        cand = cand[cand != u]
        if cand.size == 0:
            return cand, np.array([], dtype=np.float64)
        sim_row = self._u_sim if self._u_sim is not None else self.user_sim.sim[u]
        co_row = self._u_co if self._u_co is not None else self.user_sim.co_count[u]
        sims = sim_row[cand].astype(np.float64)
        co = co_row[cand].astype(np.float64)
        valid = (co >= self.min_co_rated) & (sims > 0)
        cand = cand[valid]
        sims = sims[valid]
        co = co[valid]
        if cand.size == 0:
            return cand, sims
        shrink = co / (co + self.sim_shrinkage)
        return cand, sims * shrink

    def _service_candidates(self, u: int, s: int) -> tuple[np.ndarray, np.ndarray]:
        cand = np.nonzero(self.mask[u, :])[0]
        cand = cand[cand != s]
        if cand.size == 0:
            return cand, np.array([], dtype=np.float64)
        sim_row = self._s_sim if self._s_sim is not None else self.service_sim.sim[s]
        co_row = self._s_co if self._s_co is not None else self.service_sim.co_count[s]
        sims = sim_row[cand].astype(np.float64)
        co = co_row[cand].astype(np.float64)
        valid = (co >= self.min_co_rated) & (sims > 0)
        cand = cand[valid]
        sims = sims[valid]
        co = co[valid]
        if cand.size == 0:
            return cand, sims
        shrink = co / (co + self.sim_shrinkage)
        return cand, sims * shrink

    def _predict_user_side(
        self,
        u: int,
        s: int,
        cand: np.ndarray,
        weights: np.ndarray,
        require_context: bool,
        context_valid: np.ndarray | None = None,
    ) -> ProbeResult:
        fallback_prediction, fallback_type = self._fallback_prediction(u, s)
        if require_context:
            if context_valid is None:
                context_valid = np.zeros(cand.size, dtype=bool)
            cand = cand[context_valid]
            weights = weights[context_valid]
        return self._predict_adjusted(
            base_mean=float(self._u_mean if self._u_mean is not None else self.user_mean[u]),
            values=self.train_matrix[cand, s] if cand.size else np.array([], dtype=float),
            neighbor_means=self.user_mean[cand] if cand.size else np.array([], dtype=float),
            weights=weights,
            fallback_prediction=fallback_prediction,
            fallback_type=fallback_type,
        )

    def _predict_service_side(
        self,
        u: int,
        s: int,
        cand: np.ndarray,
        weights: np.ndarray,
        require_context: bool,
        context_valid: np.ndarray | None = None,
    ) -> ProbeResult:
        fallback_prediction, fallback_type = self._fallback_prediction(u, s)
        if require_context:
            if context_valid is None:
                context_valid = np.zeros(cand.size, dtype=bool)
            cand = cand[context_valid]
            weights = weights[context_valid]
        return self._predict_adjusted(
            base_mean=float(self._s_mean if self._s_mean is not None else self.service_mean[s]),
            values=self.train_matrix[u, cand] if cand.size else np.array([], dtype=float),
            neighbor_means=self.service_mean[cand] if cand.size else np.array([], dtype=float),
            weights=weights,
            fallback_prediction=fallback_prediction,
            fallback_type=fallback_type,
        )

    def _predict_adjusted(
        self,
        base_mean: float,
        values: np.ndarray,
        neighbor_means: np.ndarray,
        weights: np.ndarray,
        fallback_prediction: float,
        fallback_type: str,
    ) -> ProbeResult:
        neighbor_count = int(values.size)
        if neighbor_count == 0:
            return self._unavailable(fallback_prediction, fallback_type)

        order = np.argsort(-weights)
        if len(order) > self.top_k:
            order = order[:self.top_k]
        values = values[order]
        neighbor_means = neighbor_means[order]
        weights = weights[order]
        effective = np.abs(weights) > 1e-12
        if not np.any(effective):
            return self._unavailable(fallback_prediction, fallback_type)

        values = values[effective]
        neighbor_means = neighbor_means[effective]
        weights = weights[effective]
        weight_sum = float(np.sum(np.abs(weights)))
        pred = base_mean + float(np.sum(weights * (values - neighbor_means)) / weight_sum)
        pred = max(0.0, pred)
        return ProbeResult(
            prediction=pred,
            fallback_prediction=pred,
            available=True,
            neighbor_count=neighbor_count,
            effective_neighbor_count=int(values.size),
            similarity_sum=weight_sum,
            max_similarity=float(np.max(weights)),
            neighbor_qos_variance=float(np.var(values)),
            fallback_type="none",
        )

    def _unavailable(self, fallback_prediction: float, fallback_type: str) -> ProbeResult:
        return ProbeResult(
            prediction=float("nan"),
            fallback_prediction=float(fallback_prediction),
            available=False,
            neighbor_count=0,
            effective_neighbor_count=0,
            similarity_sum=0.0,
            max_similarity=0.0,
            neighbor_qos_variance=0.0,
            fallback_type=fallback_type,
        )

    def _fallback_prediction(self, u: int, s: int) -> tuple[float, str]:
        service_deg = int(self.mask[:, s].sum())
        user_deg = int(self.mask[u, :].sum())
        if service_deg > 0:
            return float(self.service_mean[s]), "service_mean"
        if user_deg > 0:
            return float(self.user_mean[u]), "user_mean"
        return self.global_mean, "global_mean"

    def _geo_multipliers(
        self,
        target: int,
        cand: np.ndarray,
        countries: np.ndarray,
        latitudes: np.ndarray,
        longitudes: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        if cand.size == 0:
            return np.array([], dtype=np.float64), np.array([], dtype=bool)

        multipliers = np.ones(cand.size, dtype=np.float64)
        context_valid = np.zeros(cand.size, dtype=bool)

        target_country = countries[target]
        country_known = not pd.isna(target_country)
        cand_country_known = ~pd.isna(countries[cand])
        if country_known:
            same_country = np.zeros(cand.size, dtype=bool)
            same_country[cand_country_known] = countries[cand][cand_country_known] == target_country
            multipliers += same_country.astype(np.float64) * config.GEO_SAME_COUNTRY_BONUS
            context_valid |= np.asarray(cand_country_known, dtype=bool)

        target_lat = latitudes[target]
        target_lon = longitudes[target]
        target_coord_known = np.isfinite(target_lat) and np.isfinite(target_lon)
        cand_coord_known = np.isfinite(latitudes[cand]) & np.isfinite(longitudes[cand])
        if target_coord_known:
            valid = cand_coord_known
            if np.any(valid):
                distances = haversine_km(target_lat, target_lon, latitudes[cand][valid], longitudes[cand][valid])
                multipliers[valid] += config.GEO_DISTANCE_BONUS * np.exp(-distances / config.GEO_DISTANCE_SCALE_KM)
                context_valid |= valid

        return multipliers, context_valid

    def _as_multipliers(self, target: int, cand: np.ndarray, as_numbers: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if cand.size == 0:
            return np.array([], dtype=np.float64), np.array([], dtype=bool)
        multipliers = np.ones(cand.size, dtype=np.float64)
        target_as = as_numbers[target]
        target_known = not pd.isna(target_as)
        cand_known = ~pd.isna(as_numbers[cand])
        context_valid = np.asarray(cand_known, dtype=bool) & bool(target_known)
        if target_known:
            same_as = np.zeros(cand.size, dtype=bool)
            same_as[cand_known] = as_numbers[cand][cand_known] == target_as
            multipliers += same_as.astype(np.float64) * config.AS_SAME_BONUS
        return multipliers, context_valid


def haversine_km(lat1: float, lon1: float, lat2: np.ndarray, lon2: np.ndarray) -> np.ndarray:
    radius_km = 6371.0
    lat1_rad = np.radians(lat1)
    lon1_rad = np.radians(lon1)
    lat2_rad = np.radians(lat2)
    lon2_rad = np.radians(lon2)
    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon / 2.0) ** 2
    return 2.0 * radius_km * np.arcsin(np.sqrt(a))
