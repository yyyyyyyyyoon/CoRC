"""Feature groups used by the residual corrector.

The module defines the full structure-search input and the final 17-column
CoRC input, together with their ablation conditions."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from . import config

PROBES = ("ucf", "scf", "geo_ucf", "geo_scf", "as_ucf", "as_scf")
DIAGNOSTICS = ("neighbor_count", "effective_neighbor_count", "similarity_sum",
               "max_similarity", "available", "neighbor_qos_variance")
OBSERVATION = ("user_observation_count", "service_observation_count")

# Columns whose range is fixed by construction rather than estimated from the
# data: a flag is 0 or 1, and the effective neighbour count is capped at the
# probe's `top_k`.
BOUNDS = {"available": 1.0, "effective_neighbor_count": float(config.TOP_K_NEIGHBORS)}

CORE = ("pred",)
DIRECTION = tuple(f"dev_{probe}" for probe in PROBES)
RELIABILITY = tuple(f"{probe}_{stat}" for probe in PROBES
                    for stat in DIAGNOSTICS) + OBSERVATION
FULL = CORE + DIRECTION + RELIABILITY

# The four conditions of the first comparison. Each names the columns it keeps;
# the rest are present but zeroed.
CONDITIONS = {
    "A_base": CORE,
    "B_direction": CORE + DIRECTION,
    "C_reliability": CORE + RELIABILITY,
    "D_full": FULL,
}

# Chosen on validation and not revisited; Table 6 is that search.
FINAL_PROBES = ("ucf", "scf", "geo_ucf", "geo_scf")
FINAL_REMOVED_PROBES = ("as_ucf", "as_scf")
FINAL_REMOVED_KINDS = ("support",)


# Final-17 against Final-17 without the geography-weighted probes.
CONTEXT_OVERLAP_PROBES = ("geo_ucf", "geo_scf")


def final_conditions() -> dict:
    """Return the final CoRC input and its direction/reliability ablations."""
    final = set(_final())
    return {
        "core_only": [c for c in FULL if c in CORE],
        "core_direction": [c for c in FULL if c in final
                           and (c in CORE or c in DIRECTION)],
        "core_reliability": [c for c in FULL if c in final
                             and (c in CORE or c in RELIABILITY)],
        "final": list(_final()),
    }


def _final() -> tuple[str, ...]:
    return tuple(without(FINAL_REMOVED_PROBES, FINAL_REMOVED_KINDS))


# Removing a probe takes its direction and all six of its diagnostics with it.
ABLATIONS = {
    "full": (),
    "wo_user_side": ("ucf", "geo_ucf", "as_ucf"),
    "wo_service_side": ("scf", "geo_scf", "as_scf"),
    "wo_general_cf": ("ucf", "scf"),
    "wo_geography": ("geo_ucf", "geo_scf"),
    "wo_as": ("as_ucf", "as_scf"),
}
# Reliability by kind rather than by probe, keeping all six directions.
RELIABILITY_KINDS = {
    "support": ("neighbor_count", "effective_neighbor_count", "available"),
    "similarity": ("similarity_sum", "max_similarity"),
    "dispersion": ("neighbor_qos_variance",),
}


def kind_columns(kind: str) -> list[str]:
    """Every column of one reliability kind, across all six probes."""
    stats = RELIABILITY_KINDS[kind]
    columns = [f"{probe}_{stat}" for probe in PROBES for stat in stats]
    if kind == "support":
        # How many observations a user or a service has is a support measure
        # like the neighbour counts, and belongs with them.
        columns += list(OBSERVATION)
    return columns


def without_kind(kind: str) -> list[str]:
    dropped = set(kind_columns(kind))
    return [column for column in FULL if column not in dropped]


def without(probes: tuple[str, ...] = (), kinds: tuple[str, ...] = ()) -> list[str]:
    """Return columns remaining after probe and reliability-kind removals."""
    dropped: set[str] = set()
    for probe in probes:
        dropped.update(probe_columns(probe))
    for kind in kinds:
        dropped.update(kind_columns(kind))
    return [column for column in FULL if column not in dropped]


# The observation counts belong to a side, so dropping that side drops them.
SIDE_OBSERVATION = {"wo_user_side": ("user_observation_count",),
                    "wo_service_side": ("service_observation_count",)}


def bound_of(column: str) -> float:
    """Return a fixed diagnostic range, or zero for data-estimated scaling."""
    for stat, limit in BOUNDS.items():
        if column.endswith(stat):
            return limit
    return 0.0


def probe_columns(probe: str) -> list[str]:
    return [f"dev_{probe}"] + [f"{probe}_{stat}" for stat in DIAGNOSTICS]


def ablation_columns(name: str) -> list[str]:
    """What `name` keeps, in the fixed FULL order."""
    dropped = set()
    for probe in ABLATIONS[name]:
        dropped.update(probe_columns(probe))
    dropped.update(SIDE_OBSERVATION.get(name, ()))
    return [column for column in FULL if column not in dropped]


def attach(frame: pd.DataFrame, base: str = "pred") -> pd.DataFrame:
    """Add signed probe-to-base deviations while preserving missing probes."""
    out = frame.copy()
    for probe in PROBES:
        prediction = out[f"{probe}_prediction"].to_numpy(dtype=float)
        out[f"dev_{probe}"] = prediction - out[base].to_numpy(dtype=float)
    return out


def matrix(frame: pd.DataFrame, keep: list[str] | tuple[str, ...],
           mean: np.ndarray | None = None,
           std: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build fixed-width inputs with transforms appropriate to each feature type."""
    values = frame[list(FULL)].to_numpy(dtype=float)
    signed = np.array([column in DIRECTION for column in FULL])
    bound = np.array([bound_of(column) for column in FULL])
    bounded = bound > 0
    magnitude = ~(signed | bounded)
    values[:, magnitude] = np.log1p(np.clip(values[:, magnitude], 0.0, None))
    values[:, signed] = np.sign(values[:, signed]) * np.log1p(np.abs(values[:, signed]))
    # An unavailable probe leaves its deviation missing, and the scale is fitted
    # on the entries that have one. The diagnostics are not missing in that
    # case -- zero neighbours and zero similarity are what the probe found --
    # so they enter the statistics as they stand.
    absent = ~np.isfinite(values)
    if mean is None:
        # Centred only where centring is meaningful: not for the bounded
        # columns, which are already on a fixed scale, and not for the signed
        # deviations, whose zero carries meaning.
        mean = np.where(bounded | signed, 0.0, np.nanmean(values, axis=0))
        spread = np.nanstd(values, axis=0)
        std = np.where(bounded, np.where(bounded, bound, 1.0),
                       np.where(spread > 1e-12, spread, 1.0))
    standardised = (values - mean) / std
    standardised[absent] = 0.0
    mask = np.array([column in set(keep) for column in FULL], dtype=float)
    return standardised * mask, mean, std


def load_probe_frame(evidence_dir: Path, qos: str, seed: int, density_tag: str,
                     role: str, train_entries: np.ndarray | None = None) -> pd.DataFrame:
    """Load probe predictions and diagnostics by user-service key."""
    columns = (["user_id", "service_id"] + list(OBSERVATION)
               + [f"{probe}_prediction" for probe in PROBES]
               + [f"{probe}_{stat}" for probe in PROBES for stat in DIAGNOSTICS])
    path = evidence_dir / qos / f"seed{seed}" / density_tag / f"evidence_{role}_predictions.csv"
    frame = pd.read_csv(path, usecols=columns)
    if train_entries is not None and role == "train":
        for column, axis in (("user_observation_count", 0),
                             ("service_observation_count", 1)):
            degree = pd.Series(train_entries[:, axis]).value_counts()
            key = frame["user_id" if axis == 0 else "service_id"]
            frame[column] = (key.map(degree).fillna(0).to_numpy() - 1).clip(min=0)
    return frame


FINAL = _final()
# The seventeen columns the paper calls CoRC. Aliased rather than renamed
# because the ablations above are built from the same definition, and calling
# those CoRC too would be wrong.
CORC = FINAL
