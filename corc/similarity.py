"""Collaborative similarity and leave-one-entry-out updates.

Similarities use mean-centred cosine values with co-rating counts."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class AxisSimilarity:
    sim: np.ndarray          # (n, n) 유사도
    co_count: np.ndarray     # (n, n) 공통 관측 수
    entity_mean: np.ndarray  # (n,) 관측값 평균 (관측 없으면 global mean으로 대체)
    deg: np.ndarray          # (n,) 관측 수
    centered: np.ndarray | None = None  # (n, m) 평균중심화, 결측 0
    norm: np.ndarray | None = None      # (n,) 행 노름


def compute_axis_similarity(mat: np.ndarray, dtype=np.float64) -> AxisSimilarity:
    """mat: (n_entities, n_features), NaN=미관측. 행(entity) 간 유사도를 계산한다."""
    mask = ~np.isnan(mat)
    deg = mask.sum(axis=1)

    global_mean = np.nanmean(mat)
    # deg==0인 행에서 nanmean이 "Mean of empty slice" 경고를 내므로
    # sum/count로 직접 계산해 빈 슬라이스를 피한다.
    row_sum = np.nansum(mat, axis=1)
    entity_mean = np.where(deg > 0, row_sum / np.maximum(deg, 1), global_mean)

    centered = np.where(mask, mat - entity_mean[:, None], 0.0).astype(dtype)
    norm = np.sqrt((centered ** 2).sum(axis=1))

    sim = centered @ centered.T
    denom = np.outer(norm, norm)
    sim = np.divide(sim, denom, out=np.zeros_like(sim), where=denom > 0)
    np.fill_diagonal(sim, 0.0)

    mask_f = mask.astype(dtype)
    co_count = mask_f @ mask_f.T
    np.fill_diagonal(co_count, 0.0)

    return AxisSimilarity(sim=sim, co_count=co_count, entity_mean=entity_mean, deg=deg,
                          centered=centered, norm=norm)


@dataclass
class LeaveOneOutState:
    """State for removing one observation without recomputing all similarities."""
    sim: np.ndarray            # (n, n) similarity, original
    co_count: np.ndarray       # (n, n) co-observation counts, original
    centered: np.ndarray       # (n, m) mean-centred, zero-filled
    norm: np.ndarray           # (n,) row norms
    entity_mean: np.ndarray    # (n,) row means
    deg: np.ndarray            # (n,) observation counts
    row_sum: np.ndarray        # (n,) sum of observed values
    row_sqsum: np.ndarray      # (n,) sum of squared observed values
    mask: np.ndarray           # (n, m) observed
    overlap: np.ndarray        # (n, n) T: centred mass of k over i's observations

    def drop(self, i: int, j: int) -> tuple[np.ndarray, np.ndarray, float, int]:
        """Row `i` of (sim, co_count) plus its mean and degree, with (i, j) removed."""
        n = int(self.deg[i])
        if n <= 1 or not self.mask[i, j]:
            return self.sim[i], self.co_count[i], float(self.entity_mean[i]), n

        x = float(self.centered[i, j] + self.entity_mean[i])
        n2 = n - 1
        s2 = float(self.row_sum[i]) - x
        mean2 = s2 / n2
        norm2 = float(np.sqrt(max(float(self.row_sqsum[i]) - x * x - s2 * s2 / n2, 0.0)))

        shift = float(self.entity_mean[i]) - mean2
        dot = self.centered @ self.centered[i]
        dot = dot + shift * self.overlap[i] - (x - mean2) * self.centered[:, j]
        denom = norm2 * self.norm
        sim = np.divide(dot, denom, out=np.zeros_like(dot), where=denom > 0)
        sim[i] = 0.0

        co = self.co_count[i] - self.mask[:, j]
        co[i] = 0.0
        return sim, co, mean2, n2


def leave_one_out_state(mat: np.ndarray, axis: AxisSimilarity,
                        dtype=np.float64) -> LeaveOneOutState:
    mask = ~np.isnan(mat)
    mask_f = mask.astype(dtype)
    return LeaveOneOutState(
        sim=axis.sim, co_count=axis.co_count, centered=axis.centered, norm=axis.norm,
        entity_mean=axis.entity_mean, deg=axis.deg,
        row_sum=np.nansum(mat, axis=1), row_sqsum=np.nansum(mat ** 2, axis=1),
        mask=mask_f, overlap=mask_f @ axis.centered.T,
    )
