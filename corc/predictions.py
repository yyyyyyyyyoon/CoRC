"""Shared output schema and metrics for all base predictors.

Using one schema lets every downstream stage load predictor results uniformly."""
from __future__ import annotations

import numpy as np
import pandas as pd


def values_at(matrix: np.ndarray, entries: np.ndarray) -> np.ndarray:
    """The observed QoS for a list of (user, service) pairs."""
    return matrix[entries[:, 0], entries[:, 1]].astype(np.float64)


def metric_row(qos_type: str, seed: int, density: float, split_role: str,
               y_true: np.ndarray, y_pred: np.ndarray, elapsed_seconds: float,
               n_epochs_run: int, predictor: str) -> dict:
    """One row of a runner's metrics file: a cell summarised."""
    abs_error = np.abs(y_pred - y_true)
    return {
        "predictor": predictor,
        "qos_type": qos_type,
        "seed": seed,
        "density": density,
        "split_role": split_role,
        "n_entries": int(len(y_true)),
        "mae": float(np.mean(abs_error)),
        "rmse": float(np.sqrt(np.mean((y_pred - y_true) ** 2))),
        "median_abs_error": float(np.median(abs_error)),
        "p95_abs_error": float(np.quantile(abs_error, 0.95)),
        "worst5_mean_abs_error": float(np.mean(abs_error[abs_error >= np.quantile(abs_error, 0.95)])),
        "negative_fraction": float(np.mean(y_pred < 0)),
        "elapsed_seconds": elapsed_seconds,
        "n_epochs_run": n_epochs_run,
    }


def prediction_frame(qos_type: str, seed: int, density: float, split_role: str,
                     entries: np.ndarray, y_true: np.ndarray,
                     y_pred: np.ndarray, predictor: str) -> pd.DataFrame:
    """Per-entry predictions for one cell, in the layout everything downstream reads."""
    # `predictor` is passed rather than defaulted: this helper serves every
    # baseline, and a default here once labelled PMF's files "mf".
    return pd.DataFrame({
        "predictor": predictor,
        "qos_type": qos_type,
        "seed": seed,
        "density": density,
        "split_role": split_role,
        "user_id": entries[:, 0].astype(np.int32),
        "service_id": entries[:, 1].astype(np.int32),
        "y_true": y_true,
        "pred": y_pred,
        "abs_error": np.abs(y_pred - y_true),
    })
