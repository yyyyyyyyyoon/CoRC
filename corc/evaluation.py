"""Evaluation metrics for QoS prediction and server selection.

Includes MAE, pairwise error, ranking measures, regret summaries, bootstrap
intervals, and multiple-testing correction."""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from scipy import stats

from . import config
from .server_selection import WEIGHT_COLUMNS, score


def per_user_regret(frame: pd.DataFrame, pred: dict, k: int) -> pd.Series:
    """Mean regret for each of the 256 users, the unit every test here uses."""
    frame = frame.sort_values(["sample_id", "candidate_rank"])
    n = frame["sample_id"].nunique()
    grid = lambda col: torch.as_tensor(
        frame[col].to_numpy(dtype=np.float64).reshape(n, k), dtype=torch.float32)
    weights = {key: grid(col) for key, col in WEIGHT_COLUMNS.items()}
    ld, fr = grid("LD"), grid("FR")
    truth = score(grid("true_rt"), grid("true_tp"), ld, fr, weights)
    guess = score(torch.as_tensor(pred["rt"].reshape(n, k), dtype=torch.float32),
                  torch.as_tensor(pred["tp"].reshape(n, k), dtype=torch.float32),
                  ld, fr, weights)
    row = torch.arange(n)
    best = truth.argmax(1)
    gap = (truth[row, best] - truth[row, guess.argmax(1)]).numpy()
    users = frame.groupby("sample_id", sort=True)["user_id"].first().to_numpy()
    return pd.Series(gap, index=users).groupby(level=0).mean()


def interval(delta: np.ndarray, draws: int, rng) -> tuple[float, float, float, float]:
    """Return percentile and BCa intervals for paired per-user differences."""
    index = rng.integers(0, len(delta), size=(draws, len(delta)))
    replicates = delta[index].mean(axis=1)
    low, high = np.quantile(replicates, [0.025, 0.975])
    observed = float(delta.mean())
    share = float((replicates < observed).mean())
    if share in (0.0, 1.0):
        return float(low), float(high), float(low), float(high)
    bias = stats.norm.ppf(share)
    total, n = delta.sum(), len(delta)
    jackknife = (total - delta) / (n - 1)
    centred = jackknife.mean() - jackknife
    accel = (centred ** 3).sum() / (6 * ((centred ** 2).sum() ** 1.5) + 1e-300)
    bounds = []
    for q in (0.025, 0.975):
        z = bias + stats.norm.ppf(q)
        bounds.append(float(np.quantile(
            replicates, np.clip(stats.norm.cdf(bias + z / (1 - accel * z)), 0, 1))))
    return float(low), float(high), bounds[0], bounds[1]


def pair_error(pred: np.ndarray, true: np.ndarray) -> float:
    """Measure errors in pairwise prediction gaps within each candidate set."""
    dp = pred[:, :, None] - pred[:, None, :]
    dt = true[:, :, None] - true[:, None, :]
    k = pred.shape[1]
    off = ~np.eye(k, dtype=bool)
    return float(np.abs(dp - dt)[:, off].mean())


def kendall(pred: np.ndarray, true: np.ndarray) -> float:
    """Mean Kendall tau-a between the two orderings, instance by instance."""
    dp = np.sign(pred[:, :, None] - pred[:, None, :])
    dt = np.sign(true[:, :, None] - true[:, None, :])
    k = pred.shape[1]
    off = ~np.eye(k, dtype=bool)
    return float((dp * dt)[:, off].mean())


def holm(pvalues: dict[str, float]) -> dict[str, float]:
    """Holm step-down adjustment over a pre-declared family of tests."""
    ordered = sorted(pvalues.items(), key=lambda kv: kv[1])
    n, adjusted, running = len(ordered), {}, 0.0
    for i, (key, p) in enumerate(ordered):
        running = max(running, min((n - i) * p, 1.0))
        adjusted[key] = running
    return adjusted


def measure(frame: pd.DataFrame, pred_rt: np.ndarray, pred_tp: np.ndarray,
            k: int) -> tuple[dict, dict]:
    """Every quantity for one set of predictions on one candidate frame."""
    frame = frame.sort_values(["sample_id", "candidate_rank"])
    n = frame["sample_id"].nunique()
    grid = lambda col: frame[col].to_numpy(dtype=np.float64).reshape(n, k)
    weights = {key: grid(col) for key, col in WEIGHT_COLUMNS.items()}
    ld, fr = grid("LD"), grid("FR")
    true = {"rt": grid("true_rt"), "tp": grid("true_tp")}
    pred = {"rt": pred_rt.reshape(n, k), "tp": pred_tp.reshape(n, k)}

    as_t = lambda x: torch.as_tensor(x, dtype=torch.float32)
    truth_score = score(as_t(true["rt"]), as_t(true["tp"]),
                        as_t(ld), as_t(fr),
                        {key: as_t(value) for key, value in weights.items()}).numpy()
    pred_score = score(as_t(pred["rt"]), as_t(pred["tp"]),
                       as_t(ld), as_t(fr),
                       {key: as_t(value) for key, value in weights.items()}).numpy()

    best = truth_score.argmax(1)
    pick = pred_score.argmax(1)
    row = np.arange(n)
    per_request = truth_score[row, best] - truth_score[row, pick]

    # The true runner-up: the best candidate once the true best is removed.
    masked = truth_score.copy()
    masked[row, best] = -np.inf
    second = masked.argmax(1)

    best_pred = pred_score[row, best]
    # OPA: of the K-1 comparisons the true best takes part in, how many does it
    # win under the prediction. CPA is the one against the runner-up.
    # The comparison against itself is never strict, so it drops out and the
    # count already runs over the other K-1 candidates.
    opa = (best_pred[:, None] > pred_score).sum(1) / (pred_score.shape[1] - 1)
    cpa = (best_pred > pred_score[row, second])
    margin_pred = best_pred - pred_score[row, second]
    margin_true = truth_score[row, best] - truth_score[row, second]

    out = {"top1": float((pick == best).mean()),
           "kendall": kendall(pred_score, truth_score),
           "opa": float(opa.mean()),
           "cpa": float(cpa.mean()),
           "critical_margin_error": float(np.abs(margin_pred - margin_true).mean()),
           "score_absolute": float(np.abs(pred_score - truth_score).mean()),
           "score_relative": pair_error(pred_score, truth_score),
           "regret_instance": float(per_request.mean())}
    for qos in config.QOS_TYPES:
        out[f"absolute_{qos}"] = float(np.abs(pred[qos] - true[qos]).mean())
        out[f"relative_{qos}"] = pair_error(pred[qos], true[qos])
    return out, {"regret": per_request, "margin_true": margin_true,
                 "hit": pick == best}
