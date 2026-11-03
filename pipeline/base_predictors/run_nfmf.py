"""Train NFMF on the fixed experimental splits and write joint RT/TP predictions
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
from base_predictors.nfmf import (BATCH_SIZE, BIAS_WEIGHT, CONTEXT_LAYERS,
                                      EARLY_STOPPING_ROUNDS, EMBEDDING_DIM, EPOCHS,
                                      LEARNING_RATE, TASK_LAYERS, TASKS, WEIGHT_DECAY,
                                      NfmfPredictor, build_context)
from corc.predictions import metric_row, prediction_frame, values_at

PREFIX = "nfmf"


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main() -> int:
    if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split-dir", type=Path,
                        default=config.OUTPUT_DIR / "splits" / config.SPLIT_VERSION)
    parser.add_argument("--out-dir", type=Path,
                        default=config.OUTPUT_DIR / "base_predictors" / PREFIX
                                / config.SPLIT_VERSION)
    parser.add_argument("--densities", nargs="+", type=float,
                        default=sorted(config.DENSITIES))
    parser.add_argument("--seeds", nargs="+", type=int, default=config.SPLIT_SEEDS)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--target-scaling", choices=["none", "standardise"],
                        default="none")
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--device", default=None,
                        help="cuda / cpu; default picks cuda when visible")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--prefix", default=None)
    args = parser.parse_args()
    prefix = args.prefix or PREFIX
    args.out_dir.mkdir(parents=True, exist_ok=True)

    dataset = load_dataset()
    matrices = {qos: dataset.matrix(qos) for qos in TASKS}
    context = build_context(dataset.users, dataset.services)
    import torch
    log(f"device={args.device or ('cuda' if torch.cuda.is_available() else 'cpu')}")
    log(f"{context['n_users']}x{context['n_services']}  "
        f"context {sum(context['sizes'].values())}개 범주, 목표 스케일 {args.target_scaling}")

    (args.out_dir / "manifest.json").write_text(json.dumps({
        "predictor": "NFMF",
        "joint": "one fit per cell answers both QoS types",
        "hyperparameters": {
            "embedding_dim": EMBEDDING_DIM, "task_layers": list(TASK_LAYERS),
            "context_layers": list(CONTEXT_LAYERS), "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY, "batch_size": BATCH_SIZE,
            "epochs": args.epochs, "early_stopping_rounds": EARLY_STOPPING_ROUNDS,
            "bias_weight": BIAS_WEIGHT, "loss": "L1, tasks weighted equally",
            "target_scaling": args.target_scaling,
            # The joint model stops on `validation_score`, the mean of the two
            # task MAEs, not on either one alone. With `target_scaling="none"`
            # -- the paper's setting -- that mean is on the raw scale, so
            # throughput, thirty-odd times larger, effectively picks the epoch.
            # Both tasks reached their best at the same epoch in every pilot
            # cell, so this is recorded rather than worked around.
            "select_by": "score",
            "validation_fraction": args.validation_fraction,
        },
        "from_paper": ["embedding dimension", "task network shape", "L1 loss",
                       "equal task weights", "Adam", "learning rate",
                       "weight decay", "batch size", "context encoder with "
                       "batch normalisation", "NFCF concat/add/product",
                       "context-aware bias interaction"],
        "reimplementation_decisions": {
            "context_encoder_widths": "452 -> 64 -> 64 -> 64, from the figure's "
                                      "three layers; widths are not stated",
            "task_activations": "ReLU between layers, linear output, no batch "
                                "normalisation outside the encoder",
            "bias_weight": "not reported, set to 1.0",
            "stopping": f"internal 9:1 split, patience {EARLY_STOPPING_ROUNDS}, "
                        f"cap {args.epochs}",
            "missing_context": "a reserved category for absent AS, IP and "
                               "provider; coordinates filled with the "
                               "sub-training mean",
            "clipping": "none; negative and out-of-range rates are reported",
        },
        "densities": args.densities, "seeds": args.seeds,
    }, indent=2, ensure_ascii=False), encoding="utf-8")

    metrics: list[dict] = []
    for seed in args.seeds:
        splits = {qos: np.load(args.split_dir / f"{qos}_seed{seed}_{config.SPLIT_VERSION}.npz")
                  for qos in TASKS}
        roles = {role: splits["rt"][f"{role}_entries"]
                 for role in ("validation", "test")}
        for role, entries in roles.items():
            assert np.array_equal(entries, splits["tp"][f"{role}_entries"]), \
                f"{role} 엔트리가 QoS별로 다릅니다"
        for density in args.densities:
            tag = density_tag(density)
            entries = splits["rt"][f"{tag}_train_entries"]
            assert np.array_equal(entries, splits["tp"][f"{tag}_train_entries"])
            combo = {qos: args.out_dir / qos / f"seed{seed}" / tag for qos in TASKS}
            if args.resume and all((path / f"{prefix}_test_predictions.csv").exists()
                                   for path in combo.values()):
                log(f"  seed{seed} {tag}: 건너뜀")
                continue

            started = time.perf_counter()
            model = NfmfPredictor(context, epochs=args.epochs, seed=seed,
                                  target_scaling=args.target_scaling,
                                  validation_fraction=args.validation_fraction,
                                  device=args.device, threads=args.threads)
            model.fit(entries, {qos: values_at(matrices[qos], entries) for qos in TASKS})
            elapsed = time.perf_counter() - started

            history = pd.DataFrame([row.__dict__ for row in model.history])
            for qos in TASKS:
                combo[qos].mkdir(parents=True, exist_ok=True)
                history.to_csv(combo[qos] / f"{prefix}_training_history.csv", index=False)
            for role, role_entries in roles.items():
                predicted = model.predict_entries(role_entries)
                for qos in TASKS:
                    truth = values_at(matrices[qos], role_entries)
                    frame = prediction_frame(qos, seed, density, role, role_entries,
                                             truth, predicted[qos], prefix)
                    frame.to_csv(combo[qos] / f"{prefix}_{role}_predictions.csv",
                                 index=False)
                    row = metric_row(qos, seed, density, role, truth, predicted[qos],
                                     elapsed, len(model.history), predictor=prefix)
                    train_values = values_at(matrices[qos], entries)
                    row.update({
                        "best_epoch": model.best_epoch,
                        "epochs_run": len(model.history),
                        "at_cap": len(model.history) >= args.epochs,
                        # Recorded, not acted on: nothing is clipped here or in
                        # any other baseline, so the tail is the behaviour of an
                        # unbounded head rather than a fault.
                        "negative_rate": float((predicted[qos] < 0).mean()),
                        "over_range_rate": float(
                            ((predicted[qos] > train_values.max())
                             | (predicted[qos] < train_values.min())).mean()),
                        "pred_std": float(predicted[qos].std()),
                        "true_std": float(truth.std()),
                    })
                    metrics.append(row)
            pd.DataFrame(metrics).to_csv(args.out_dir / f"{prefix}_metrics.csv", index=False)
            shown = {qos: next(m["mae"] for m in reversed(metrics)
                               if m["qos_type"] == qos and m["split_role"] == "validation")
                     for qos in TASKS}
            log(f"  seed{seed} {tag}: val MAE rt {shown['rt']:.4f} tp {shown['tp']:.2f}  "
                f"epoch {model.best_epoch}/{len(model.history)}  ({elapsed:.0f}초)")

    pd.DataFrame(metrics).to_csv(args.out_dir / f"{prefix}_metrics.csv", index=False)
    table = pd.DataFrame(metrics)
    capped = int(table[table.split_role == "validation"].at_cap.sum())
    log(f"\n상한({args.epochs}) 도달: {capped}/{len(table[table.split_role == 'validation'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
