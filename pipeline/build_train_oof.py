"""Build out-of-fold predictions for corrector training.

Each training entry is predicted by a base model that did not observe it;
fixed deployed hyperparameters are reused without consulting the test split."""
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
from base_predictors.nfmf import NfmfPredictor, build_context
from base_predictors.ldcf import LdcfPredictor
from base_predictors.mf import WsrecMatrixFactorization
from corc.predictions import prediction_frame, values_at


FOLDS = 5
# Read back from each baseline's manifest.json rather than restated, so a change
# there cannot silently desynchronise the fold models from the deployed one.
SPECS = {
    "PMF": {"family": "pmf", "prefix": "pmf", "kind": "wsrec_mf"},
    "BiasMF": {"family": "biasmf", "prefix": "biasmf", "kind": "wsrec_mf"},
    "LDCF": {"family": "ldcf", "prefix": "ldcf", "kind": "ldcf"},
    # Joint: one fit per cell answers both QoS types, so this one is driven by
    # its own loop rather than the per-QoS one below.
    "NFMF": {"family": "nfmf", "prefix": "nfmf", "kind": "nfmf", "joint": True},
}


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def read_hyperparameters(root: Path, family: str, version: str) -> dict:
    manifest = json.loads((root / family / version / "manifest.json").read_text(encoding="utf-8"))
    return manifest["hyperparameters"]


def best_epoch(root: Path, family: str, prefix: str, version: str, qos: str,
               seed: int, density: float, select_by: str) -> tuple[int, int] | None:
    """Return the selected epoch and final recorded epoch from run history."""
    path = (root / family / version / qos / f"seed{seed}" / density_tag(density)
            / f"{prefix}_training_history.csv")
    if not path.exists():
        return None
    history = pd.read_csv(path)
    column = f"validation_{select_by}"
    if column not in history or history[column].isna().all():
        return None
    return (int(history.loc[history[column].idxmin(), "epoch"]),
            int(history["epoch"].max()))


def build_model(kind: str, shape: tuple[int, int], hp: dict, seed: int,
                side: dict | None, qos: str = "rt"):
    """A fold model: published settings, fixed epochs, no early stopping."""
    if kind == "wsrec_mf":
        # Regularisation and step size are per QoS type in WSRec, so the fold
        # model has to be told which matrix it is fitting.
        return WsrecMatrixFactorization(
            n_users=shape[0], n_services=shape[1], n_factors=hp["dimension"],
            lam=hp["lambda"][qos], eta_init=hp["eta_init"][qos],
            max_iter=hp["max_iter"], seed=seed, use_bias=hp["use_bias"],
            line_search_steps=hp["line_search_steps"])
    if kind == "ldcf":
        # `validation_fraction=0` keeps the fold model on the deployed model's
        # epoch instead of letting it choose one of its own.
        return LdcfPredictor(
            shape[0], shape[1], side["user_lc"], side["item_lc"],
            layers=tuple(hp["layers"]), learning_rate=hp["learning_rate"],
            batch_size=hp["batch_size"], epochs=hp["epochs"],
            validation_fraction=0.0, seed=seed)
    raise ValueError(f"알 수 없는 예측기 종류: {kind}")


def ldcf_side_features(dataset) -> dict[str, np.ndarray]:
    """The two location columns LDCF feeds into its shared embedding table."""
    from .baselines.run_ldcf import location_codes

    user_lc, item_lc = location_codes(dataset)
    return {"user_lc": user_lc, "item_lc": item_lc}


def main() -> int:
    if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=config.OUTPUT_DIR / "base_predictors")
    parser.add_argument("--split-dir", type=Path,
                        default=config.OUTPUT_DIR / "splits" / config.SPLIT_VERSION)
    parser.add_argument("--version", default=config.SPLIT_VERSION)
    parser.add_argument("--predictors", nargs="+",
                        default=["PMF", "BiasMF", "LDCF"])
    parser.add_argument("--device", default=None,
                        help="cuda / cpu for NFMF fold models; default picks cuda")
    parser.add_argument("--qos-types", nargs="+", default=config.QOS_TYPES)
    parser.add_argument("--seeds", nargs="+", type=int, default=config.SPLIT_SEEDS)
    # All of them. The old default was the lowest density alone, which once
    # produced a corrector with training data for a single density and no error
    # anywhere to say so.
    parser.add_argument("--densities", nargs="+", type=float,
                        default=sorted(config.DENSITIES))
    parser.add_argument("--folds", type=int, default=FOLDS)
    # The per-cell predictions live under seed-specific paths and never
    # collide, but the summary and the manifest are one file each. Shards that
    # run side by side would overwrite one another's, so each takes a suffix
    # and the pieces are merged afterwards.
    parser.add_argument("--summary-suffix", default="",
                        help="appended to train_oof_summary / _manifest names")
    parser.add_argument("--epoch-source", default="deployed",
                        choices=["deployed", "manifest"],
                        help="`deployed` reads the released model's best epoch "
                             "from its history; `manifest` uses the nominal budget")
    args = parser.parse_args()

    dataset = load_dataset()
    # Built only for the predictor that needs them. Importing `baselines.run_ldcf`
    # unconditionally once made a run that asked only for NFMF fail at startup,
    # before it had done any work.
    kinds = {SPECS[name]["kind"] for name in args.predictors}
    side = ldcf_side_features(dataset) if "ldcf" in kinds else None
    records = []

    def run_joint(name: str, spec: dict, hp: dict, budget: int) -> None:
        """Fit one shared fold model and retain both QoS prediction heads."""
        context = build_context(dataset.users, dataset.services)
        matrices = {qos: dataset.matrix(qos) for qos in args.qos_types}
        for seed in args.seeds:
            splits = {qos: np.load(args.split_dir / f"{qos}_seed{seed}_{args.version}.npz")
                      for qos in args.qos_types}
            for density in args.densities:
                tag = density_tag(density)
                entries = splits[args.qos_types[0]][f"{tag}_train_entries"]
                for qos in args.qos_types[1:]:
                    assert np.array_equal(entries, splits[qos][f"{tag}_train_entries"]),                         "QoS별 학습 엔트리가 다릅니다"
                values = {qos: values_at(matrices[qos], entries) for qos in args.qos_types}
                # Same partition as every other predictor: drawn from the split
                # seed, so the correctors stay comparable.
                folds = np.random.default_rng(seed).integers(0, args.folds, len(entries))
                oof = {qos: np.empty(len(entries), dtype=float) for qos in args.qos_types}
                chosen = best_epoch(args.root, spec["family"], spec["prefix"],
                                    args.version, args.qos_types[0], seed, density,
                                    hp.get("select_by", "mae"))
                fold_hp = dict(hp)
                if args.epoch_source == "deployed" and chosen:
                    fold_hp["epochs"] = chosen[0]
                started = time.perf_counter()
                for fold in range(args.folds):
                    held = folds == fold
                    if not held.any():
                        continue
                    model = NfmfPredictor(
                        context, epochs=fold_hp["epochs"],
                        learning_rate=fold_hp["learning_rate"],
                        weight_decay=fold_hp["weight_decay"],
                        batch_size=fold_hp["batch_size"],
                        bias_weight=fold_hp["bias_weight"],
                        target_scaling=fold_hp["target_scaling"],
                        device=args.device,
                        # No validation is passed, so the loop runs the full
                        # budget and keeps its final state.
                        validation_fraction=0.0, seed=seed)
                    model.fit(entries[~held], {qos: values[qos][~held]
                                               for qos in args.qos_types})
                    predicted = model.predict_entries(entries[held])
                    for qos in args.qos_types:
                        oof[qos][held] = predicted[qos]
                elapsed = time.perf_counter() - started
                for qos in args.qos_types:
                    combo = (args.root / spec["family"] / args.version / qos
                             / f"seed{seed}" / tag)
                    combo.mkdir(parents=True, exist_ok=True)
                    frame = prediction_frame(qos, seed, density, "train", entries,
                                             values[qos], oof[qos], spec["prefix"])
                    frame.to_csv(combo / f"{spec['prefix']}_train_predictions.csv",
                                 index=False)
                    mae = float(np.abs(values[qos] - oof[qos]).mean())
                    records.append({
                        "predictor": name, "qos_type": qos, "seed": seed,
                        "density": density, "n_entries": int(len(entries)),
                        "folds": args.folds, "oof_mae": mae,
                        "epochs_fixed": fold_hp["epochs"], "epochs_manifest": budget,
                        "epochs_best": chosen[0] if chosen else None,
                        "epochs_last_run": chosen[1] if chosen else None,
                        "seconds": elapsed})
                    log(f"  {name:6s} {qos} seed{seed} {tag}: "
                        f"OOF MAE {mae:.4f}  ({elapsed:.0f}초)")

    for name in args.predictors:
        spec = SPECS[name]
        hp = read_hyperparameters(args.root, spec["family"], args.version)
        # PMF and BiasMF have no epoch to inherit: WSRec runs a fixed 300
        # iterations with no early stopping, so a fold model matches the
        # deployed one by running the same 300.
        fixed = spec["kind"] == "wsrec_mf"
        budget = hp["max_iter"] if fixed else hp["epochs"]
        log(f"{name}: epoch 출처={'고정' if fixed else args.epoch_source} "
            f"(manifest {budget}), 조기종료 없음, {args.folds}-fold")
        if spec.get("joint"):
            run_joint(name, spec, hp, budget)
            continue
        for qos in args.qos_types:
            matrix = dataset.matrix(qos)
            for seed in args.seeds:
                split = np.load(args.split_dir / f"{qos}_seed{seed}_{args.version}.npz")
                for density in args.densities:
                    tag = density_tag(density)
                    fold_hp = dict(hp)
                    # None for PMF and BiasMF: there is no deployed epoch to
                    # inherit, because WSRec runs a fixed 300 iterations.
                    chosen = None
                    if not fixed:
                        chosen = best_epoch(args.root, spec["family"], spec["prefix"],
                                            args.version, qos, seed, density,
                                            hp.get("select_by", "mae"))
                        if args.epoch_source == "deployed" and chosen:
                            fold_hp["epochs"] = chosen[0]
                    entries = split[f"{tag}_train_entries"]
                    values = values_at(matrix, entries)
                    # Folds are drawn from the split seed, so the same partition
                    # serves every predictor and the correctors stay comparable.
                    folds = np.random.default_rng(seed).integers(0, args.folds, len(entries))
                    oof = np.empty(len(entries), dtype=float)
                    started = time.perf_counter()
                    for fold in range(args.folds):
                        held = folds == fold
                        if not held.any():
                            continue
                        extra = side if spec["kind"] == "ldcf" else None
                        model = build_model(spec["kind"], matrix.shape, fold_hp,
                                            seed, extra, qos)
                        # No validation is passed, so the loop runs the full
                        # budget and keeps its final state.
                        model.fit(entries[~held], values[~held])
                        oof[held] = model.predict_entries(entries[held])

                    combo = (args.root / spec["family"] / args.version / qos
                             / f"seed{seed}" / tag)
                    combo.mkdir(parents=True, exist_ok=True)
                    frame = prediction_frame(qos, seed, density, "train", entries,
                                             values, oof, spec["prefix"])
                    frame.to_csv(combo / f"{spec['prefix']}_train_predictions.csv", index=False)

                    mae = float(np.abs(values - oof).mean())
                    elapsed = time.perf_counter() - started
                    records.append({
                        "predictor": name, "qos_type": qos, "seed": seed,
                        "density": density, "n_entries": int(len(entries)),
                        "folds": args.folds, "oof_mae": mae,
                        # PMF and BiasMF count iterations, not epochs, and have
                        # no epoch to inherit -- the budget is fixed at 300.
                        "epochs_fixed": fold_hp["max_iter"] if fixed else fold_hp["epochs"],
                        "epochs_manifest": budget,
                        "epochs_best": chosen[0] if chosen else None,
                        "epochs_last_run": chosen[1] if chosen else None,
                        "seconds": elapsed})
                    log(f"  {name:6s} {qos} seed{seed} {tag}: "
                        f"OOF MAE {mae:.4f}  ({elapsed:.0f}초)")

    summary = pd.DataFrame(records)
    out = args.root / f"train_oof_manifest{args.summary_suffix}.json"
    summary.to_csv(args.root / f"train_oof_summary{args.summary_suffix}.csv",
                   index=False)
    out.write_text(json.dumps({
        "purpose": "out-of-fold training-split predictions for corrector training",
        "folds": args.folds,
        "early_stopping": "disabled during fold training",
        "epoch_source": args.epoch_source,
        "epoch_rule": "the baseline's best epoch, fixed before this experiment; "
                      "not the last epoch run, which overshoots it by the "
                      "early-stopping patience",
        "hyperparameters": "read from each baseline manifest, not re-selected",
        "cf_note": "UPCC/IPCC/UIPCC are excluded: the evidence probe's training "
                   "log is already leave-one-entry-out",
        "predictors": args.predictors, "seeds": args.seeds,
        "densities": args.densities, "qos_types": args.qos_types,
    }, indent=2), encoding="utf-8")
    log(f"outputs -> {args.root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
