"""Candidate scoring and server-selection regret.

Scores use within-set min-max normalisation for RT, TP, load, and failure rate."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch

WEIGHT_COLUMNS = {"rt": "weight_rt", "tp": "weight_tp",
                  "ld": "weight_ld", "fr": "weight_fr"}
KEY_COLUMNS = ["user_id", "service_id"]



def minmax(values: torch.Tensor, lower_is_better: bool) -> torch.Tensor:
    """Min-max normalise within each candidate set; map constant features to zero."""
    lo = values.min(dim=1, keepdim=True).values
    hi = values.max(dim=1, keepdim=True).values
    span = hi - lo
    numerator = hi - values if lower_is_better else values - lo
    return torch.where(span > 0, numerator / span.clamp(min=1e-12),
                       torch.zeros_like(values))


def score(rt: torch.Tensor, tp: torch.Tensor, ld: torch.Tensor, fr: torch.Tensor,
          weights: dict[str, torch.Tensor]) -> torch.Tensor:
    return (weights["rt"] * minmax(rt, True) + weights["tp"] * minmax(tp, False)
            + weights["ld"] * minmax(ld, True) + weights["fr"] * minmax(fr, True))




@dataclass
class Instances:
    """One role's selection instances, shaped (n, K, ...) and ready to batch."""
    features: torch.Tensor          # (n, K, F)
    raw: torch.Tensor               # (n, K, 2)  base RT and TP predictions
    truth: torch.Tensor             # (n, K, 2)  measured RT and TP
    ld: torch.Tensor                # (n, K)
    fr: torch.Tensor                # (n, K)
    weights: dict[str, torch.Tensor]
    user_weight: torch.Tensor       # (n,)  1 / that user's request count
    scale: torch.Tensor             # (2,)  residual std, to put RT and TP on one scale

    def __len__(self) -> int:
        return self.features.shape[0]

    def observed_score(self) -> torch.Tensor:
        return score(self.truth[..., 0], self.truth[..., 1], self.ld, self.fr, self.weights)


def build_instances(frame, predictor: str, structure_columns: list[str],
                    k: int, scale: torch.Tensor | None = None,
                    mean: torch.Tensor | None = None,
                    std: torch.Tensor | None = None) -> tuple[Instances, dict]:
    """Reshape candidate rows using scaling derived from training data."""
    # Sorted here rather than in the caller. `grid` reshapes by row order while
    # the user ids below are taken by grouping on sample_id, which pandas sorts;
    # sample_id is a string, so the two orders are not the same unless the frame
    # is sorted first. Leaving it to the caller once attached the wrong user to
    # 15,566 of 39,426 instances, which silently skews the user weights and so
    # the user-level regret computed from them.
    frame = frame.sort_values(["sample_id", "candidate_rank"])
    n = frame["sample_id"].nunique()
    grid = lambda col: torch.as_tensor(
        frame[col].to_numpy(dtype=np.float64).reshape(n, k), dtype=torch.float32)

    raw = torch.stack([grid(f"pred_{predictor}_rt"), grid(f"pred_{predictor}_tp")], dim=-1)
    truth = torch.stack([grid("true_rt"), grid("true_tp")], dim=-1)
    columns = [raw[..., 0], raw[..., 1]] + [grid(c) for c in structure_columns]
    features = torch.stack(columns, dim=-1)

    # log1p before standardising. The neighbour-dispersion columns are
    # variances and run from a median of 135 to a maximum of 106,379 on
    # throughput, a skew of 5.9; standardising alone leaves |z| above 10 on a
    # fraction of rows, and in a linear first layer those few rows dominate the
    # gradient. A tree never sees this because it splits on order, which is why
    # the same columns helped the boosted corrector and hurt this one. Every
    # column here is a non-negative magnitude, so the transform is defined
    # throughout and order-preserving.
    features = torch.log1p(features.clamp(min=0.0))

    if mean is None:
        mean = features.reshape(-1, features.shape[-1]).mean(0)
        std = features.reshape(-1, features.shape[-1]).std(0).clamp(min=1e-6)
    features = (features - mean) / std

    if scale is None:
        residual = truth - raw
        scale = residual.reshape(-1, 2).std(0).clamp(min=1e-6)

    users = frame.groupby("sample_id", sort=True)["user_id"].first().to_numpy()
    counts = pd.Series(users).map(pd.Series(users).value_counts()).to_numpy()
    instances = Instances(
        features=features, raw=raw, truth=truth,
        ld=grid("LD"), fr=grid("FR"),
        weights={key: grid(col) for key, col in WEIGHT_COLUMNS.items()},
        user_weight=torch.as_tensor(1.0 / counts, dtype=torch.float32),
        scale=scale)
    return instances, {"mean": mean, "std": std, "scale": scale, "users": users}


def user_mean(per_request: np.ndarray, instances: Instances) -> float:
    """Average per-user means so each user receives equal weight."""
    weight = instances.user_weight.numpy()
    return float((per_request * weight).sum() / weight.sum())


def regret(instances: Instances, corrected: torch.Tensor | None = None) -> np.ndarray:
    """Per-instance regret, for the base predictions or the corrected ones."""
    source = instances.raw if corrected is None else corrected
    predicted = score(source[..., 0], source[..., 1], instances.ld, instances.fr,
                      instances.weights)
    observed = instances.observed_score()
    picked = predicted.argmax(dim=1)
    taken = observed.gather(1, picked[:, None]).squeeze(1)
    return (observed.max(dim=1).values - taken).detach().numpy()


