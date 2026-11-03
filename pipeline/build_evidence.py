"""Build collaborative evidence features for each split role.

Training evidence can be computed leave-one-entry-out to prevent leakage."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from corc import config
from corc.dataset import load_dataset, mask_from_entries
from corc.evidence import PROBE_NAMES, EvidenceProbeModel, ProbeResult


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def density_tag(density: float) -> str:
    return f"density{int(round(density * 100)):02d}"


def values_at(matrix: np.ndarray, entries: np.ndarray) -> np.ndarray:
    return matrix[entries[:, 0], entries[:, 1]].astype(np.float64)


def result_columns(prefix: str, result: ProbeResult) -> dict:
    return {
        f"{prefix}_prediction": result.prediction,
        f"{prefix}_fallback_prediction": result.fallback_prediction,
        f"{prefix}_available": result.available,
        f"{prefix}_neighbor_count": result.neighbor_count,
        f"{prefix}_effective_neighbor_count": result.effective_neighbor_count,
        f"{prefix}_similarity_sum": result.similarity_sum,
        f"{prefix}_max_similarity": result.max_similarity,
        f"{prefix}_neighbor_qos_variance": result.neighbor_qos_variance,
        f"{prefix}_fallback_type": result.fallback_type,
    }


def predict_entries(
    model: EvidenceProbeModel,
    entries: np.ndarray,
    y_true: np.ndarray,
    qos_type: str,
    seed: int,
    density: float,
    split_role: str,
    progress_every: int,
) -> pd.DataFrame:
    rows = []
    started = time.perf_counter()
    for idx, ((u, s), y) in enumerate(zip(entries, y_true, strict=True), start=1):
        probe_results = model.predict_all(int(u), int(s))
        row = {
            "qos_type": qos_type,
            "seed": seed,
            "density": density,
            "split_role": split_role,
            "user_id": int(u),
            "service_id": int(s),
            "true_qos": float(y),
            # Read through the model so that a training entry does not count
            # itself: its own observation is part of the matrix its features
            # come from, while validation/test entries are not.
            "user_observation_count": model.user_support(int(u), int(s)),
            "service_observation_count": model.service_support(int(u), int(s)),
        }
        for probe_name in PROBE_NAMES:
            row.update(result_columns(probe_name, probe_results[probe_name]))
        rows.append(row)
        if progress_every and idx % progress_every == 0:
            elapsed = time.perf_counter() - started
            log(f"{split_role}: predicted {idx}/{len(entries)} entries in {elapsed:.1f}s")
    return pd.DataFrame(rows)


def summarize_probe_metrics(df: pd.DataFrame, qos_type: str, seed: int, density: float, split_role: str) -> list[dict]:
    rows = []
    for probe_name in PROBE_NAMES:
        pred_col = f"{probe_name}_prediction"
        available_col = f"{probe_name}_available"
        available = df[available_col].astype(bool)
        if available.any():
            errors = np.abs(df.loc[available, pred_col].to_numpy(dtype=float) - df.loc[available, "true_qos"].to_numpy(dtype=float))
            mae = float(np.mean(errors))
            rmse = float(np.sqrt(np.mean(errors ** 2)))
            p95 = float(np.quantile(errors, 0.95))
        else:
            mae = rmse = p95 = float("nan")
        rows.append({
            "qos_type": qos_type,
            "seed": seed,
            "density": density,
            "split_role": split_role,
            "probe": probe_name,
            "n_entries": int(len(df)),
            "available_entries": int(available.sum()),
            "coverage": float(available.mean()),
            "fallback_rate": float((~available).mean()),
            "mae_on_available": mae,
            "rmse_on_available": rmse,
            "p95_abs_error_on_available": p95,
            "mean_effective_neighbor_count": float(df[f"{probe_name}_effective_neighbor_count"].mean()),
            "mean_similarity_sum": float(df[f"{probe_name}_similarity_sum"].mean()),
        })
    return rows


def maybe_sample_entries(entries: np.ndarray, sample_size: int, seed: int) -> np.ndarray:
    if sample_size <= 0 or len(entries) <= sample_size:
        return entries
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(entries), size=sample_size, replace=False)
    return entries[np.sort(idx)]


def run_one(
    matrix: np.ndarray,
    users: pd.DataFrame,
    services: pd.DataFrame,
    split_npz: Path,
    qos_type: str,
    seed: int,
    density: float,
    out_dir: Path,
    args: argparse.Namespace,
) -> list[dict]:
    tag = density_tag(density)
    split = np.load(split_npz)
    train_entries = split[f"{tag}_train_entries"]
    train_mask = mask_from_entries(matrix.shape, train_entries)
    train_matrix = np.where(train_mask, matrix, np.nan)

    log(f"building evidence model qos_type={qos_type} seed={seed} density={density:.0%}")
    model = EvidenceProbeModel(
        train_matrix=train_matrix,
        users=users,
        services=services,
        top_k=args.top_k,
        min_co_rated=args.min_co_rated,
        sim_shrinkage=args.sim_shrinkage,
        leave_one_out=args.leave_one_out,
    )

    combo_dir = out_dir / qos_type / f"seed{seed}" / tag
    combo_dir.mkdir(parents=True, exist_ok=True)

    metrics: list[dict] = []
    for split_role in args.split_roles:
        key = f"{tag}_train_entries" if split_role == "train" else f"{split_role}_entries"
        entries = split[key]
        # Density must NOT enter the sample seed. The fixed splits guarantee that
        # every density shares the same validation/test entries, and sampling per
        # density would break that: the intersection across four densities would
        # collapse to a few hundred entries, making paired density comparisons
        # impossible.
        sample_seed = seed + {"validation": 0, "test": 1_000_000}.get(split_role, 2_000_000)
        entries = maybe_sample_entries(entries, args.sample_size, sample_seed)
        y_true = values_at(matrix, entries)
        log(f"predicting {split_role} entries={len(entries)}")
        df = predict_entries(
            model,
            entries,
            y_true,
            qos_type,
            seed,
            density,
            split_role,
            progress_every=args.progress_every,
        )
        df.to_csv(combo_dir / f"evidence_{split_role}_predictions.csv", index=False)
        metrics.extend(summarize_probe_metrics(df, qos_type, seed, density, split_role))
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--split-dir",
        type=Path,
        default=config.OUTPUT_DIR / "splits" / config.SPLIT_VERSION,
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=config.OUTPUT_DIR / "evidence_probes" / config.SPLIT_VERSION,
    )
    parser.add_argument("--qos-types", nargs="+", choices=["rt", "tp"], default=config.QOS_TYPES)
    parser.add_argument("--densities", nargs="+", type=float, default=config.DENSITIES)
    parser.add_argument("--seeds", nargs="+", type=int, default=config.SPLIT_SEEDS)
    parser.add_argument("--split-roles", nargs="+", choices=["validation", "test", "train"],
                        default=["validation", "test"])
    parser.add_argument("--leave-one-out", action="store_true",
                        help="drop each entry from the similarity state before reading its "
                             "features; required for the train split, where the entry is "
                             "inside the matrix its own features come from")
    parser.add_argument("--top-k", type=int, default=config.TOP_K_NEIGHBORS)
    parser.add_argument("--min-co-rated", type=int, default=config.MIN_CO_RATED)
    parser.add_argument("--sim-shrinkage", type=float, default=config.SIM_SHRINKAGE)
    parser.add_argument("--sample-size", type=int, default=0, help="0 means all entries")
    parser.add_argument("--progress-every", type=int, default=50_000)
    parser.add_argument("--quick", action="store_true")
    return parser.parse_args()


def main() -> int:
    if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")

    args = parse_args()
    if args.quick:
        args.qos_types = ["rt"]
        args.densities = [0.05]
        args.seeds = [config.SPLIT_SEEDS[0]]
        args.split_roles = ["validation"]
        args.sample_size = 1_000
        args.progress_every = 250
        args.out_dir = args.out_dir / "quick_check"

    dataset = load_dataset()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "probe_names": PROBE_NAMES,
        "split_version": config.SPLIT_VERSION,
        "split_dir": str(args.split_dir),
        "qos_types": args.qos_types,
        "densities": args.densities,
        "seeds": args.seeds,
        "split_roles": args.split_roles,
        "sample_size": args.sample_size,
        "parameters": {
            "top_k": args.top_k,
            "min_co_rated": args.min_co_rated,
            "sim_shrinkage": args.sim_shrinkage,
            "geo_same_country_bonus": config.GEO_SAME_COUNTRY_BONUS,
            "geo_distance_bonus": config.GEO_DISTANCE_BONUS,
            "geo_distance_scale_km": config.GEO_DISTANCE_SCALE_KM,
            "as_same_bonus": config.AS_SAME_BONUS,
        },
    }
    (args.out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    metrics: list[dict] = []
    for qos_type in args.qos_types:
        matrix = dataset.matrix(qos_type)
        for seed in args.seeds:
            split_npz = args.split_dir / f"{qos_type}_seed{seed}_{config.SPLIT_VERSION}.npz"
            if not split_npz.exists():
                raise FileNotFoundError(split_npz)
            for density in args.densities:
                metrics.extend(
                    run_one(matrix, dataset.users, dataset.services, split_npz, qos_type, seed, density, args.out_dir, args)
                )

    metrics_df = pd.DataFrame(metrics)
    metrics_df.to_csv(args.out_dir / "evidence_probe_metrics.csv", index=False)
    log(f"metrics -> {args.out_dir / 'evidence_probe_metrics.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
