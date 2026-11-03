"""Train PMF or BiasMF on the fixed experimental splits and write standard
prediction files and run metadata."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from corc import config
from corc.dataset import load_dataset
from corc.experiment_data import density_tag
from corc.predictions import (metric_row, prediction_frame,
                                   values_at)
from base_predictors.mf import LINE_SEARCH_STEPS, WsrecMatrixFactorization


# `para` in WSRec's run_rt.py / run_tp.py, per variant and QoS type.
WSREC = {
    "pmf":    {"use_bias": False,
               "rt": {"lam": 40.0, "eta_init": 0.01},
               "tp": {"lam": 800.0, "eta_init": 0.001}},
    "biasmf": {"use_bias": True,
               "rt": {"lam": 20.0, "eta_init": 0.01},
               "tp": {"lam": 500.0, "eta_init": 0.001}},
}
DIMENSION = 10
MAX_ITER = 300


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def train_one(matrix: np.ndarray, split_npz: Path, qos_type: str, seed: int,
              density: float, out_dir: Path, args: argparse.Namespace) -> list[dict]:
    tag = density_tag(density)
    split = np.load(split_npz)
    train_entries = split[f"{tag}_train_entries"]
    validation_entries = split["validation_entries"]
    test_entries = split["test_entries"]

    train_values = values_at(matrix, train_entries)
    validation_values = values_at(matrix, validation_entries)
    test_values = values_at(matrix, test_entries)

    setting = WSREC[args.variant]
    log(f"{args.variant} qos={qos_type} seed={seed} density={density:.0%} "
        f"train={len(train_entries)} lambda={setting[qos_type]['lam']} "
        f"eta={setting[qos_type]['eta_init']}")

    started = time.perf_counter()
    model = WsrecMatrixFactorization(
        n_users=matrix.shape[0], n_services=matrix.shape[1],
        n_factors=args.dimension, max_iter=args.max_iter, seed=seed,
        use_bias=setting["use_bias"], **setting[qos_type])
    # Validation never selects anything here -- there is no early stopping and
    # the model is whatever iteration 300 produced. It is recorded so the
    # appendix can show where the curve was by then.
    model.fit(train_entries, train_values,
              validation_entries=validation_entries,
              validation_values=validation_values)
    elapsed = time.perf_counter() - started

    combo_dir = out_dir / qos_type / f"seed{seed}" / tag
    combo_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([row.__dict__ for row in model.history]).to_csv(
        combo_dir / f"{args.prefix}_training_history.csv", index=False)

    rows: list[dict] = []
    for split_role, entries, y_true in (("validation", validation_entries, validation_values),
                                        ("test", test_entries, test_values)):
        y_pred = model.predict_entries(entries)
        rows.append(metric_row(qos_type, seed, density, split_role, y_true, y_pred,
                               elapsed, args.max_iter, predictor=args.variant))
        prediction_frame(qos_type, seed, density, split_role, entries, y_true,
                         y_pred, args.variant).to_csv(
            combo_dir / f"{args.prefix}_{split_role}_predictions.csv", index=False)
        log(f"  {split_role} MAE={rows[-1]['mae']:.6g} RMSE={rows[-1]['rmse']:.6g} "
            f"neg={rows[-1]['negative_fraction']:.2%}")
    best = min(model.history, key=lambda r: r.validation_mae)
    log(f"  수렴 기록: validation MAE 최저 {best.validation_mae:.4f} @ {best.iteration}회, "
        f"300회 시점 {model.history[-1].validation_mae:.4f} ({elapsed:.0f}초)")
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=sorted(WSREC), required=True)
    parser.add_argument("--split-dir", type=Path,
                        default=config.OUTPUT_DIR / "splits" / config.SPLIT_VERSION)
    parser.add_argument("--out-dir", type=Path, default=None,
                        help="defaults to baselines/<variant>/<split version>")
    parser.add_argument("--prefix", default=None, help="file prefix; defaults to the variant")
    parser.add_argument("--qos-types", nargs="+", choices=["rt", "tp"], default=config.QOS_TYPES)
    parser.add_argument("--densities", nargs="+", type=float, default=sorted(config.DENSITIES))
    parser.add_argument("--seeds", nargs="+", type=int, default=config.SPLIT_SEEDS)
    parser.add_argument("--dimension", type=int, default=DIMENSION)
    parser.add_argument("--max-iter", type=int, default=MAX_ITER)
    args = parser.parse_args()
    if args.out_dir is None:
        args.out_dir = config.OUTPUT_DIR / "base_predictors" / args.variant / config.SPLIT_VERSION
    if args.prefix is None:
        args.prefix = args.variant
    return args


def main() -> int:
    if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")

    args = parse_args()
    dataset = load_dataset()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    setting = WSREC[args.variant]
    (args.out_dir / "manifest.json").write_text(json.dumps({
        "predictor": args.variant,
        "source": "LanternYing/WSRec, PMF/ and Biased_MF/ (Jamie Zhu)",
        "split_version": config.SPLIT_VERSION,
        "split_dir": str(args.split_dir),
        "hyperparameters": {
            "dimension": args.dimension,
            "max_iter": args.max_iter,
            "lambda": {q: setting[q]["lam"] for q in ("rt", "tp")},
            "eta_init": {q: setting[q]["eta_init"] for q in ("rt", "tp")},
            "line_search_steps": LINE_SEARCH_STEPS,
            "use_bias": setting["use_bias"],
            "optimizer": "full-batch gradient descent with halving line search",
            "loss": "0.5 * sum squared error + 0.5 * lambda * squared L2 norms",
            "init": "Uniform[0,1) for every parameter",
            "early_stopping": False,
            "clip_negative": False,
            "target_transform": "none",
            "calibrate": False,
        },
        "deviations": {
            "splits": "fixed nested density splits instead of WSRec's per-round "
                      "random removal, so every predictor sees the same entries",
            "repeats": "3 seeds instead of 20 rounds",
            "validation": "recorded per iteration but never used for selection",
        },
        "qos_types": args.qos_types,
        "densities": args.densities,
        "seeds": args.seeds,
    }, indent=2), encoding="utf-8")

    metrics: list[dict] = []
    for qos_type in args.qos_types:
        matrix = dataset.matrix(qos_type)
        for seed in args.seeds:
            split_npz = args.split_dir / f"{qos_type}_seed{seed}_{config.SPLIT_VERSION}.npz"
            if not split_npz.exists():
                raise FileNotFoundError(split_npz)
            for density in args.densities:
                metrics.extend(train_one(matrix, split_npz, qos_type, seed,
                                         density, args.out_dir, args))

    target = args.out_dir / f"{args.prefix}_metrics.csv"
    pd.DataFrame(metrics).to_csv(target, index=False)
    log(f"metrics -> {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
