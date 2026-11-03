"""Train LDCF on the fixed experimental splits and write standard prediction
files and run metadata."""
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
from base_predictors.ldcf import (BATCH_SIZE, EPOCHS, HUBER_DELTA, LAYERS,
                                      LEARNING_RATE, LdcfPredictor)
from corc.predictions import metric_row, prediction_frame, values_at


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def codes(series: pd.Series) -> np.ndarray:
    """Integer codes with missing values kept as their own category."""
    return pd.factorize(series.astype("object").fillna("__missing__"))[0]


def location_codes(dataset) -> tuple[np.ndarray, np.ndarray]:
    user = np.stack([codes(dataset.users["country"]),
                     codes(dataset.users["as_no"])], axis=1).astype(np.int64)
    service = np.stack([codes(dataset.services["country"]),
                        codes(dataset.services["as_no"])], axis=1).astype(np.int64)
    for name, table, limit in (("user", user, len(dataset.users)),
                               ("service", service, len(dataset.services))):
        if table.max() >= limit:
            raise ValueError(
                f"{name} location codes reach {table.max()} but the original "
                f"sizes that embedding table at {limit}")
    return user, service


def main() -> int:
    if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split-dir", type=Path,
                        default=config.OUTPUT_DIR / "splits" / config.SPLIT_VERSION)
    parser.add_argument("--out-dir", type=Path,
                        default=config.OUTPUT_DIR / "base_predictors" / "ldcf" / config.SPLIT_VERSION)
    parser.add_argument("--prefix", default="ldcf")
    parser.add_argument("--qos-types", nargs="+", choices=["rt", "tp"], default=config.QOS_TYPES)
    parser.add_argument("--densities", nargs="+", type=float, default=sorted(config.DENSITIES))
    parser.add_argument("--seeds", nargs="+", type=int, default=config.SPLIT_SEEDS)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--learning-rate", type=float, default=LEARNING_RATE)
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    dataset = load_dataset()
    user_lc, service_lc = location_codes(dataset)

    (args.out_dir / "manifest.json").write_text(json.dumps({
        "predictor": "ldcf",
        "source": "yinchunhui-ahu/Location-aware_Deep_Collaborative_Filtering, "
                  "commit 3934e06 (2020-11-10), run_rt.py / run_tp.py",
        "split_version": config.SPLIT_VERSION,
        "hyperparameters": {
            "layers": list(LAYERS),
            "embedding_dim": LAYERS[0] // 4,
            "optimizer": "Adam", "learning_rate": args.learning_rate,
            "decay": 0.0, "batch_size": args.batch_size, "epochs": args.epochs,
            "loss": f"Huber(delta={HUBER_DELTA})",
            "reg_layers": [0, 0, 0, 0, 0],
            "dropout": 0.0,
            "ac_cos": True,
            "shared_location_table": True,
            "clip_negative": False,
        },
        "deviations": {
            "best_epoch": "validation MAE on a held-out tenth of the training "
                          "entries; the original uses the test set",
            "train_fraction": 1.0 - args.validation_fraction,
            "splits": "fixed nested density splits, shared across predictors",
            "repeats": "3 seeds instead of 20 rounds",
        },
        "qos_types": args.qos_types,
        "densities": args.densities,
        "seeds": args.seeds,
    }, indent=2), encoding="utf-8")

    metrics: list[dict] = []
    for qos_type in args.qos_types:
        matrix = dataset.matrix(qos_type)
        for seed in args.seeds:
            split = np.load(args.split_dir
                            / f"{qos_type}_seed{seed}_{config.SPLIT_VERSION}.npz")
            validation_entries = split["validation_entries"]
            test_entries = split["test_entries"]
            v_true = values_at(matrix, validation_entries)
            t_true = values_at(matrix, test_entries)
            for density in args.densities:
                tag = density_tag(density)
                train_entries = split[f"{tag}_train_entries"]
                y = values_at(matrix, train_entries)

                started = time.perf_counter()
                model = LdcfPredictor(
                    matrix.shape[0], matrix.shape[1], user_lc, service_lc,
                    layers=LAYERS, learning_rate=args.learning_rate,
                    batch_size=args.batch_size, epochs=args.epochs,
                    validation_fraction=args.validation_fraction,
                    seed=seed, threads=args.threads)
                model.fit(train_entries, y)
                elapsed = time.perf_counter() - started

                combo = args.out_dir / qos_type / f"seed{seed}" / tag
                combo.mkdir(parents=True, exist_ok=True)
                pd.DataFrame([row.__dict__ for row in model.history]).to_csv(
                    combo / f"{args.prefix}_training_history.csv", index=False)

                for role, entries, truth in (("validation", validation_entries, v_true),
                                             ("test", test_entries, t_true)):
                    pred = model.predict_entries(entries)
                    metrics.append(metric_row(qos_type, seed, density, role, truth,
                                              pred, elapsed, model.best_epoch,
                                              predictor="ldcf"))
                    frame = prediction_frame(qos_type, seed, density, role,
                                             entries, truth, pred, "ldcf")
                    frame.to_csv(combo / f"{args.prefix}_{role}_predictions.csv",
                                 index=False)
                at_cap = model.best_epoch == args.epochs
                log(f"  {qos_type} seed{seed} {tag}: best epoch "
                    f"{model.best_epoch}/{args.epochs}{' (상한)' if at_cap else ''}  "
                    f"val MAE {metrics[-2]['mae']:.4f}  test MAE {metrics[-1]['mae']:.4f}  "
                    f"neg {metrics[-1]['negative_fraction']:.2%}  ({elapsed:.0f}초)")

    target = args.out_dir / f"{args.prefix}_metrics.csv"
    pd.DataFrame(metrics).to_csv(target, index=False)
    capped = sum(1 for m in metrics if m["split_role"] == "test"
                 and m["n_epochs_run"] == args.epochs)
    log(f"best epoch가 상한({args.epochs})인 조합: {capped}/{len(metrics)//2}")
    log(f"metrics -> {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
