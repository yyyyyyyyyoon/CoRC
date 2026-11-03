"""Load WS-DREAM data and create fixed nested train/validation/test splits.

Missing values, response-time filtering, and shared RT/TP partitions follow the
experimental protocol used by every predictor and corrector."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import config


@dataclass
class QosDataset:
    rt: np.ndarray          # (n_users, n_services), NaN = 결측
    tp: np.ndarray          # (n_users, n_services), NaN = 결측
    users: pd.DataFrame     # index = user id
    services: pd.DataFrame  # index = service id

    @property
    def n_users(self) -> int:
        return self.rt.shape[0]

    @property
    def n_services(self) -> int:
        return self.rt.shape[1]

    def matrix(self, qos_type: str) -> np.ndarray:
        if qos_type == "rt":
            return self.rt
        if qos_type == "tp":
            return self.tp
        raise ValueError(f"unknown qos_type: {qos_type}")


def _load_matrix(path) -> np.ndarray:
    mat = np.loadtxt(path)
    mat = np.where(mat == config.MISSING_VALUE, np.nan, mat)
    return mat


def _load_userlist(path) -> pd.DataFrame:
    df = pd.read_csv(
        path,
        sep="\t",
        skiprows=2,
        header=None,
        names=["user_id", "ip", "country", "ip_no", "as_str", "lat", "lon"],
        na_values=["null", "NULL", ""],
        encoding="latin-1",
    )
    # rtMatrix/tpMatrix 행 순서는 파일 내 행 순서(위치)와 대응한다. wslist.txt에는
    # 알려진 id 중복(예: service_id 4702가 3행에 반복)이 있어 id 컬럼을 인덱스로
    # 쓰면 안 된다 -> 위치 기반(RangeIndex)으로 정렬한다.
    df = df.reset_index(drop=True)
    df["as_no"] = df["as_str"].astype(str).str.extract(r"(AS\d+)")
    return df


def _load_wslist(path) -> pd.DataFrame:
    df = pd.read_csv(
        path,
        sep="\t",
        skiprows=2,
        header=None,
        names=[
            "service_id", "wsdl", "provider", "ip", "country",
            "ip_no", "as_str", "lat", "lon",
        ],
        na_values=["null", "NULL", ""],
        encoding="latin-1",
    )
    df = df.reset_index(drop=True)
    df["as_no"] = df["as_str"].astype(str).str.extract(r"(AS\d+)")
    return df


def load_dataset() -> QosDataset:
    rt = _load_matrix(config.RT_PATH)
    tp = _load_matrix(config.TP_PATH)
    users = _load_userlist(config.USERLIST_PATH)
    services = _load_wslist(config.WSLIST_PATH)
    assert rt.shape == (len(users), len(services)), (rt.shape, len(users), len(services))
    assert tp.shape == rt.shape
    return QosDataset(rt=rt, tp=tp, users=users, services=services)


@dataclass
class Split:
    qos_type: str
    density: float
    seed: int
    train_matrix: np.ndarray   # 관측되지 않은 칸은 NaN (테스트/미관측 모두 마스킹됨)
    train_mask: np.ndarray     # bool, True = 학습에 쓰는 관측값
    test_pool_mask: np.ndarray  # bool, True = train에 안 쓰인 원래 관측값 (density 분할 후 남은 것)


@dataclass(frozen=True)
class FixedSplitBundle:
    qos_type: str
    seed: int
    observed_entries: np.ndarray
    model_pool_entries: np.ndarray
    validation_entries: np.ndarray
    test_entries: np.ndarray
    density_train_entries: dict[float, np.ndarray]
    density_meta_folds: dict[float, list[np.ndarray]]


def density_split(full_matrix: np.ndarray, density: float, seed: int) -> Split:
    """전체 관측값 중 density 비율만 학습에 사용하고 나머지는 테스트 풀로 남긴다."""
    observed_mask = ~np.isnan(full_matrix)
    observed_idx = np.argwhere(observed_mask)

    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(observed_idx))
    observed_idx = observed_idx[perm]

    n_train = int(round(density * len(observed_idx)))
    train_idx = observed_idx[:n_train]
    test_pool_idx = observed_idx[n_train:]

    train_mask = np.zeros_like(observed_mask)
    train_mask[train_idx[:, 0], train_idx[:, 1]] = True

    test_pool_mask = np.zeros_like(observed_mask)
    test_pool_mask[test_pool_idx[:, 0], test_pool_idx[:, 1]] = True

    train_matrix = np.where(train_mask, full_matrix, np.nan)

    return Split(
        qos_type="",
        density=density,
        seed=seed,
        train_matrix=train_matrix,
        train_mask=train_mask,
        test_pool_mask=test_pool_mask,
    )


def mask_from_entries(shape: tuple[int, int], entries: np.ndarray) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    if entries.size:
        mask[entries[:, 0], entries[:, 1]] = True
    return mask


def split_from_train_entries(
    full_matrix: np.ndarray,
    train_entries: np.ndarray,
    test_entries: np.ndarray,
    qos_type: str,
    density: float,
    seed: int,
) -> Split:
    train_mask = mask_from_entries(full_matrix.shape, train_entries)
    test_pool_mask = mask_from_entries(full_matrix.shape, test_entries)
    train_matrix = np.where(train_mask, full_matrix, np.nan)
    return Split(
        qos_type=qos_type,
        density=density,
        seed=seed,
        train_matrix=train_matrix,
        train_mask=train_mask,
        test_pool_mask=test_pool_mask,
    )


def fixed_research_split(
    full_matrix: np.ndarray,
    qos_type: str,
    seed: int = config.SEED,
    densities: list[float] | tuple[float, ...] = tuple(config.DENSITIES),
    validation_fraction: float = config.VALIDATION_FRACTION,
    test_fraction: float = config.TEST_FRACTION,
    meta_folds: int = config.META_FOLDS,
    observed_mask: np.ndarray | None = None,
) -> FixedSplitBundle:
    """Create nested density masks after a fixed validation/test holdout."""
    if validation_fraction < 0 or test_fraction < 0:
        raise ValueError("validation_fraction and test_fraction must be non-negative")
    if validation_fraction + test_fraction >= 1:
        raise ValueError("validation_fraction + test_fraction must be < 1")
    if meta_folds < 2:
        raise ValueError("meta_folds must be at least 2")

    mask = ~np.isnan(full_matrix) if observed_mask is None else observed_mask
    if mask.shape != full_matrix.shape:
        raise ValueError(f"observed_mask shape {mask.shape} != matrix shape {full_matrix.shape}")
    if np.any(mask & np.isnan(full_matrix)):
        raise ValueError("observed_mask selects entries that are missing in this matrix")
    observed_entries = np.argwhere(mask).astype(np.int32)
    rng = np.random.default_rng(seed)
    permuted = observed_entries[rng.permutation(len(observed_entries))]

    n_observed = len(permuted)
    n_test = int(round(test_fraction * n_observed))
    n_validation = int(round(validation_fraction * n_observed))

    test_entries = permuted[:n_test]
    validation_entries = permuted[n_test:n_test + n_validation]
    model_pool_entries = permuted[n_test + n_validation:]

    train_perm = model_pool_entries[rng.permutation(len(model_pool_entries))]
    density_train_entries: dict[float, np.ndarray] = {}
    density_meta_folds: dict[float, list[np.ndarray]] = {}
    for density in sorted(densities):
        n_train = int(round(float(density) * n_observed))
        if n_train > len(train_perm):
            raise ValueError(
                f"density={density} requests {n_train} train entries, "
                f"but only {len(train_perm)} remain after holdout"
            )
        train_entries = train_perm[:n_train].astype(np.int32)
        density_train_entries[float(density)] = train_entries

        fold_perm = train_entries[rng.permutation(len(train_entries))]
        density_meta_folds[float(density)] = [
            fold.astype(np.int32) for fold in np.array_split(fold_perm, meta_folds)
        ]

    return FixedSplitBundle(
        qos_type=qos_type,
        seed=seed,
        observed_entries=observed_entries,
        model_pool_entries=model_pool_entries.astype(np.int32),
        validation_entries=validation_entries.astype(np.int32),
        test_entries=test_entries.astype(np.int32),
        density_train_entries=density_train_entries,
        density_meta_folds=density_meta_folds,
    )


def stratified_sample_test_entries(
    split: Split,
    sample_size: int = config.TEST_SAMPLE_SIZE,
    n_quantiles: int = config.STRATA_QUANTILES,
    seed: int = config.SEED,
) -> np.ndarray:
    """Sample test entries across user- and service-degree strata."""
    candidates = np.argwhere(split.test_pool_mask)
    if len(candidates) <= sample_size:
        return candidates

    u_deg = split.train_mask.sum(axis=1)
    s_deg = split.train_mask.sum(axis=0)

    cand_u_deg = u_deg[candidates[:, 0]]
    cand_s_deg = s_deg[candidates[:, 1]]

    u_bin = pd.qcut(cand_u_deg, q=n_quantiles, labels=False, duplicates="drop")
    s_bin = pd.qcut(cand_s_deg, q=n_quantiles, labels=False, duplicates="drop")
    strata = pd.Series(list(zip(u_bin, s_bin)))

    rng = np.random.default_rng(seed)
    groups = strata.groupby(strata).indices  # stratum -> positional indices into candidates
    n_strata = len(groups)
    target_per_stratum = int(np.ceil(sample_size / n_strata))

    picked = []
    for _, idx in groups.items():
        idx = np.asarray(idx)
        if len(idx) <= target_per_stratum:
            picked.append(idx)
        else:
            picked.append(rng.choice(idx, size=target_per_stratum, replace=False))
    picked_idx = np.concatenate(picked)

    if len(picked_idx) > sample_size:
        picked_idx = rng.choice(picked_idx, size=sample_size, replace=False)

    return candidates[picked_idx]
