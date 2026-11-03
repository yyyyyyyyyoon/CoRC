"""Run UIPCC on the fixed experimental splits and write standard prediction files
and run metadata."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from corc import config
from corc.experiment_data import density_tag
from corc.dataset import load_dataset
from base_predictors.uipcc import LAMBDA, MIN_CO_RATED, TOP_K, uipcc
from corc.predictions import metric_row, prediction_frame, values_at


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def dense(shape: tuple[int, int], entries: np.ndarray,
          values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    matrix = np.zeros(shape, dtype=np.float64)
    observed = np.zeros(shape, dtype=bool)
    matrix[entries[:, 0], entries[:, 1]] = values
    observed[entries[:, 0], entries[:, 1]] = True
    return matrix, observed


def main() -> int:
    if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split-dir", type=Path,
                        default=config.OUTPUT_DIR / "splits" / config.SPLIT_VERSION)
    parser.add_argument("--out-dir", type=Path,
                        default=config.OUTPUT_DIR / "base_predictors" / "uipcc" / config.SPLIT_VERSION)
    parser.add_argument("--prefix", default="uipcc")
    parser.add_argument("--qos-types", nargs="+", choices=["rt", "tp"], default=config.QOS_TYPES)
    parser.add_argument("--densities", nargs="+", type=float, default=sorted(config.DENSITIES))
    parser.add_argument("--seeds", nargs="+", type=int, default=config.SPLIT_SEEDS)
    parser.add_argument("--top-k", type=int, default=TOP_K)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--roles", nargs="+", default=["validation", "test", "train"],
                        choices=["validation", "test", "train"])
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    dataset = load_dataset()
    metrics = []
    for qos in args.qos_types:
        matrix = dataset.matrix(qos)
        for seed in args.seeds:
            split = np.load(args.split_dir / f"{qos}_seed{seed}_{config.SPLIT_VERSION}.npz")
            holdout = {role: split[f"{role}_entries"]
                       for role in ("validation", "test") if role in args.roles}
            for density in args.densities:
                tag = density_tag(density)
                train = split[f"{tag}_train_entries"]
                y = values_at(matrix, train)

                started = time.perf_counter()
                full, observed = dense(matrix.shape, train, y)
                predicted = uipcc(full, observed, qos, args.top_k)["uipcc"]
                elapsed = time.perf_counter() - started

                combo = args.out_dir / qos / f"seed{seed}" / tag
                combo.mkdir(parents=True, exist_ok=True)
                for role, entries in holdout.items():
                    truth = values_at(matrix, entries)
                    pred = predicted[entries[:, 0], entries[:, 1]]
                    metrics.append(metric_row(qos, seed, density, role, truth, pred,
                                              elapsed, 0, predictor="uipcc"))
                    frame = prediction_frame(qos, seed, density, role, entries,
                                             truth, pred, "uipcc")
                    frame.to_csv(combo / f"{args.prefix}_{role}_predictions.csv", index=False)

                oof_mae = None
                if "train" in args.roles:
                    # Folds come from the split seed, as in build_train_oof, so
                    # every predictor's corrector sees the same partition.
                    folds = np.random.default_rng(seed).integers(0, args.folds, len(train))
                    oof = np.empty(len(train), dtype=np.float64)
                    for fold in range(args.folds):
                        held = folds == fold
                        if not held.any():
                            continue
                        part, seen = dense(matrix.shape, train[~held], y[~held])
                        out = uipcc(part, seen, qos, args.top_k)["uipcc"]
                        oof[held] = out[train[held, 0], train[held, 1]]
                    oof_mae = float(np.mean(np.abs(y - oof)))
                    metrics.append(metric_row(qos, seed, density, "train", y, oof,
                                              time.perf_counter() - started, 0,
                                              predictor="uipcc"))
                    frame = prediction_frame(qos, seed, density, "train", train,
                                             y, oof, "uipcc")
                    frame.to_csv(combo / f"{args.prefix}_train_predictions.csv", index=False)

                test_mae = next((m["mae"] for m in metrics[-3:]
                                 if m["split_role"] == "test"), float("nan"))
                log(f"  {qos} seed{seed} {tag}: test MAE {test_mae:.4f}"
                    + (f"  OOF MAE {oof_mae:.4f}" if oof_mae is not None else "")
                    + f"  ({time.perf_counter() - started:.0f}초)")

    target = args.out_dir / f"{args.prefix}_metrics.csv"
    pd.DataFrame(metrics).to_csv(target, index=False)
    (args.out_dir / "manifest.json").write_text(json.dumps({
        "predictor": "uipcc",
        "source": "LanternYing/WSRec, UIPCC/ (Jamie Zhu)",
        "split_version": config.SPLIT_VERSION,
        "hyperparameters": {
            "top_k": args.top_k,
            "lambda": LAMBDA,
            "min_co_rated": MIN_CO_RATED,
            "similarity": "Pearson over co-observed entries, deviations from each "
                          "row's overall mean",
            "shrinkage": None,
            "fallback": "row mean when no neighbour contributes or the value "
                        "lands at or below zero",
            "clip_negative": False,
        },
        "deviations": {
            "oof": f"{args.folds}-fold, matching the other predictors, rather "
                   f"than the probes' leave-one-entry-out",
            "cold_start": "services with no training observation keep the "
                          "original's zero; they are removed from the candidate "
                          "pool by the selection dataset builder instead",
            "splits": "fixed nested density splits, shared across predictors",
            "repeats": "3 seeds instead of 20 rounds",
        },
        "qos_types": args.qos_types, "densities": args.densities, "seeds": args.seeds,
    }, indent=2), encoding="utf-8")
    log(f"metrics -> {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
