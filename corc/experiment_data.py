# Load and join experiment data by QoS type, seed, density, and split role.

# This module joins base predictions with evidence and loads candidate-level data.
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

KEY = ["user_id", "service_id"]

# (output directory, file prefix) for the five base predictors the paper reports on.
PREDICTORS = {
    "UIPCC": ("uipcc", "uipcc"),
    "PMF": ("pmf", "pmf"),
    "BiasMF": ("biasmf", "biasmf"),
    "LDCF": ("ldcf", "ldcf"),
    # Joint multi-task: one fit per cell answers both QoS types, but the files
    # it writes follow the same per-QoS layout as every other predictor.
    "NFMF": ("nfmf", "nfmf"),
}

# The evidence columns `load_role` carries alongside the prediction. Chosen per
# QoS type, because the two differ in which probe's dispersion is informative.
STRUCTURE3 = {
    "rt": ["user_observation_count", "ucf_neighbor_qos_variance",
           "geo_scf_neighbor_qos_variance"],
    "tp": ["geo_scf_neighbor_qos_variance", "scf_max_similarity",
           "ucf_neighbor_qos_variance"],
}


def density_tag(density: float) -> str:
    return f"density{int(round(density * 100)):02d}"


def load_role(root: Path, evidence_dir: Path, version: str, name: str, qos: str,
              seed: int, density: float, role: str) -> pd.DataFrame:
    # Join predictions and evidence for one split role.
    # Training predictions must be out-of-fold or leave-one-entry-out.
    family, prefix = PREDICTORS[name]
    pred = pd.read_csv(root / family / version / qos / f"seed{seed}" / density_tag(density)
                       / f"{prefix}_{role}_predictions.csv", usecols=KEY + ["y_true", "pred"])
    probe = pd.read_csv(evidence_dir / qos / f"seed{seed}" / density_tag(density)
                        / f"evidence_{role}_predictions.csv", usecols=KEY + STRUCTURE3[qos])
    return pred.merge(probe, on=KEY, how="inner")


def entry_predictions(directory: Path, predictor: str, condition: str, tag: str,
                      seed: int, role: str, ordered: pd.DataFrame) -> dict:
    # One corrector's predictions, in the candidate frame's row order.
    out = {}
    for qos in config.QOS_TYPES:
        path = directory / f"{predictor}_{qos}_{condition}_{tag}_seed{seed}_{role}.csv"
        lookup = pd.read_csv(path).drop_duplicates(KEY).set_index(KEY)["pred"]
        values = ordered.set_index(KEY).index.map(lookup).to_numpy(float)
        assert not np.isnan(values).any(), f"{path.name}: 후보쌍 누락"
        out[qos] = values
    return out


def load_candidates(role: str, density: float, seed: int, k: int,
                    directory: Path) -> pd.DataFrame:
    # One role's candidate sets: K rows per request, every predictor's column.

    # pipeline.build_selection_dataset` writes these, and the regret in Table 1, the decomposition in Figure 3 and the ladders in Tables 3 and 4 all start here.
    # Test carries no suffix: it was the first role written and the name stuck.
    suffix = "" if role == "test" else f"_{role}"
    return pd.read_csv(directory / f"candidates_k{k}_{density_tag(density)}"
                                   f"{suffix}_seed{seed}.csv")
