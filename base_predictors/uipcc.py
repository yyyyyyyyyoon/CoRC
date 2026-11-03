"""UPCC, IPCC, and UIPCC as implemented by LanternYing/WSRec.

The implementation preserves positive-neighbour filtering, row/column-mean
fallbacks, and the published weighted combination without output clamping."""
from __future__ import annotations

import numpy as np

# `para` in WSRec's UIPCC/run_rt.py and run_tp.py.
TOP_K = 10
LAMBDA = {"rt": 0.1, "tp": 0.9}
MIN_CO_RATED = 2
# Neighbours scored per pass while walking them in similarity order. Speed
# only.
NEIGHBOUR_BLOCK = 64


def row_means(matrix: np.ndarray, observed: np.ndarray
              ) -> tuple[np.ndarray, np.ndarray]:
    """Return observed-entry row means and a mask of nonempty rows.

    Empty rows retain the published zero fallback and are filtered upstream.
    """
    count = observed.sum(axis=1)
    total = np.where(observed, matrix, 0.0).sum(axis=1)
    has_data = count > 0
    return np.where(has_data, total / np.maximum(count, 1), 0.0), has_data


def pearson(matrix: np.ndarray, observed: np.ndarray, means: np.ndarray,
            has_data: np.ndarray) -> np.ndarray:
    """Compute row-wise Pearson similarity over co-observed columns."""
    deviation = np.where(observed, matrix - means[:, None], 0.0)
    squared = deviation ** 2
    mask = observed.astype(np.float64)

    upper = deviation @ deviation.T
    # Only the columns both rows observed contribute to either denominator.
    lower_a = squared @ mask.T
    lower_b = mask @ squared.T
    shared = mask @ mask.T

    denominator = np.sqrt(lower_a * lower_b)
    with np.errstate(divide="ignore", invalid="ignore"):
        similarity = np.where(denominator > 0, upper / denominator, 0.0)
    similarity[shared < MIN_CO_RATED] = 0.0
    # A row with no observation of its own is skipped entirely by the original.
    similarity[~has_data, :] = 0.0
    similarity[:, ~has_data] = 0.0
    np.fill_diagonal(similarity, 0.0)
    return similarity


def predict_axis(matrix: np.ndarray, observed: np.ndarray, top_k: int = TOP_K
                 ) -> np.ndarray:
    """UPCC over the rows of `matrix`; pass a transpose to get IPCC."""
    means, has_data = row_means(matrix, observed)
    similarity = pearson(matrix, observed, means, has_data)
    similarity = np.where(similarity > 0, similarity, 0.0)

    deviation = np.where(observed, matrix - means[:, None], 0.0)
    # Preserve the published top-k positive-neighbour rule per column.
    order = np.argsort(-similarity, axis=1)
    prediction = np.repeat(means[:, None], matrix.shape[1], axis=1)

    for i in range(matrix.shape[0]):
        if not has_data[i]:
            continue
        neighbours = order[i][similarity[i, order[i]] > 0]
        if neighbours.size == 0:
            continue
        weighted = np.zeros(matrix.shape[1])
        total = np.zeros(matrix.shape[1])
        filled = np.zeros(matrix.shape[1], dtype=np.int64)
        for start in range(0, neighbours.size, NEIGHBOUR_BLOCK):
            block = neighbours[start:start + NEIGHBOUR_BLOCK]
            usable = observed[block]
            rank = filled + np.cumsum(usable, axis=0)
            taken = usable & (rank <= top_k)
            weight = similarity[i, block][:, None]
            weighted += (weight * deviation[block] * taken).sum(axis=0)
            total += (weight * taken).sum(axis=0)
            filled = np.minimum(rank[-1], top_k)
            if filled.min() >= top_k:
                break
        with np.errstate(divide="ignore", invalid="ignore"):
            value = np.where(total > 0, weighted / np.maximum(total, 1e-12) + means[i],
                             means[i])
        # "will become worst, no need" -- the original's own comment.
        prediction[i] = np.where(value > 0, value, means[i])
    return prediction


def uipcc(matrix: np.ndarray, observed: np.ndarray, qos_type: str,
          top_k: int = TOP_K) -> dict[str, np.ndarray]:
    """UPCC, IPCC and their fixed-weight combination."""
    upcc = predict_axis(matrix, observed, top_k)
    ipcc = predict_axis(matrix.T, observed.T, top_k).T
    lam = LAMBDA[qos_type]
    return {"upcc": upcc, "ipcc": ipcc, "uipcc": lam * upcc + (1.0 - lam) * ipcc}
