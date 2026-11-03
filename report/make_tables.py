"""Regenerate the paper tables from saved predictions and versioned summaries.

This module performs aggregation only and never fits a model."""
from __future__ import annotations

import argparse
import itertools
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from corc import config
from corc.experiment_data import (PREDICTORS, density_tag, entry_predictions,
                        load_candidates)
from corc.evaluation import holm, interval, per_user_regret

PREDICTORS_USED = ["UIPCC", "PMF", "BiasMF", "LDCF", "NFMF"]
K = 10


# Table 3 and Table 4 vary one thing each: the mechanism at fixed inputs, and
# the inputs at a fixed mechanism. Both read correctors that have already run.
ARMS = {
    "corc": ("test", "final"),
    "core_direction": ("rq2_test", "core_direction"),
    "core_reliability": ("rq2_test", "core_reliability"),
    "core_only": ("rq2_test", "core_only"),
    "placebo": ("placebo", "final"),
    "linear": ("capacity/linear", "final"),
}


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def fmt(value: float, digits: int = 5) -> str:
    """Fixed width, so a figure quoted in the text and the table it came from
    cannot disagree about how many digits a number has."""
    return f"{value:.{digits}f}"


def pvalue(value: float) -> str:
    return "< 0.001" if value < 0.001 else f"{value:.3f}"


def test_mae(final: Path, out: Path) -> pd.DataFrame:
    """Compute held-out entry-level MAE from saved test predictions."""
    from corc.experiment_data import load_role

    evidence = config.OUTPUT_DIR / "evidence_probes" / config.SPLIT_VERSION / "full"
    root = config.OUTPUT_DIR / "base_predictors"
    key = ["user_id", "service_id"]
    rows = []
    for predictor, density, seed, qos in itertools.product(
            PREDICTORS_USED, sorted(config.DENSITIES), config.SPLIT_SEEDS,
            config.QOS_TYPES):
        tag = density_tag(density)
        base = load_role(root, evidence, config.SPLIT_VERSION, predictor, qos,
                         seed, density, "test")[key + ["y_true", "pred"]]
        row = {"predictor": predictor, "density": density, "seed": seed,
               "qos_type": qos,
               "vanilla": float((base["pred"] - base["y_true"]).abs().mean())}
        for label, (folder, condition) in ARMS.items():
            got = pd.read_csv(final / folder / f"{predictor}_{qos}_{condition}"
                                               f"_{tag}_seed{seed}_test.csv")
            got = got.drop_duplicates(key)
            merged = base.merge(got, on=key, how="inner", suffixes=("", "_c"))
            assert len(merged) == len(base), f"{label}: {len(merged)} vs {len(base)}"
            row[label] = float((merged["pred_c"] - merged["y_true"]).abs().mean())
        rows.append(row)
        log(f"MAE {predictor:7s} d={density:.0%} seed {seed} {qos}")
    table = pd.DataFrame(rows)
    table.to_csv(out, index=False)
    return table


KEY_LEVELS = ["predictor", "density", "seed", "arm"]


def save_regret(per: dict, constant_mae: pd.DataFrame, folder: Path) -> None:
    """Cache the sixty cells so Tables 3 and 4 need not reopen the predictions."""
    folder.mkdir(parents=True, exist_ok=True)
    frame = pd.concat(per, axis=1)
    frame.columns.names = KEY_LEVELS
    frame.to_csv(folder / "regret_per_user.csv")
    constant_mae.to_csv(folder / "constant_mae.csv", index=False)


def load_regret(folder: Path) -> tuple[dict, pd.DataFrame]:
    """Read that cache back into the shape `table_3` and `table_4` expect."""
    frame = pd.read_csv(folder / "regret_per_user.csv", header=[0, 1, 2, 3],
                        index_col=0)
    per = {}
    for predictor, density, seed, arm in frame.columns:
        per[(predictor, float(density), int(seed), arm)] = frame[
            (predictor, density, seed, arm)]
    return per, pd.read_csv(folder / "constant_mae.csv")


def regret_arms(final: Path, data: Path) -> tuple[dict, pd.DataFrame]:
    """Compute per-user regret for the correction arms in Tables 3 and 4."""
    per, constant_mae = {}, []
    root = config.OUTPUT_DIR / "base_predictors"
    for predictor, density, seed in itertools.product(
            PREDICTORS_USED, sorted(config.DENSITIES), config.SPLIT_SEEDS):
        tag = density_tag(density)
        frame = load_candidates("test", density, seed, K, data)
        ordered = frame.sort_values(["sample_id", "candidate_rank"])
        vanilla = {qos: ordered[f"pred_{predictor}_{qos}"].to_numpy(float)
                   for qos in config.QOS_TYPES}
        per[(predictor, density, seed, "vanilla")] = per_user_regret(frame, vanilla, K)

        family, prefix = PREDICTORS[predictor]
        cell = root / family / config.SPLIT_VERSION
        shift, row = {}, {"predictor": predictor, "density": density, "seed": seed}
        for qos in config.QOS_TYPES:
            folder = cell / qos / f"seed{seed}" / tag
            train = pd.read_csv(folder / f"{prefix}_train_predictions.csv",
                                usecols=["y_true", "pred"])
            shift[qos] = -float((train["pred"] - train["y_true"]).median())
            held = pd.read_csv(folder / f"{prefix}_test_predictions.csv",
                               usecols=["y_true", "pred"])
            row[qos] = float((held["pred"] + shift[qos] - held["y_true"]).abs().mean())
        constant_mae.append(row)
        per[(predictor, density, seed, "constant")] = per_user_regret(
            frame, {qos: vanilla[qos] + shift[qos] for qos in config.QOS_TYPES}, K)

        for key, (folder, condition) in ARMS.items():
            pred = entry_predictions(final / folder, predictor, condition, tag,
                                     seed, "test", ordered)
            per[(predictor, density, seed, key)] = per_user_regret(frame, pred, K)
        log(f"{predictor:7s} d={density:.0%} seed {seed}")
    return per, pd.DataFrame(constant_mae)


def table_1(args) -> None:
    """Server-selection regret for every predictor-density condition."""
    source = pd.read_csv(args.rq1_dir / "rq1_test.csv")
    deltas = pd.read_csv(args.rq1_dir / "rq1_test_user_deltas.csv",
                         header=[0, 1, 2], index_col=0)
    deltas.columns.names = ["predictor", "density", "seed"]
    # The density level is a string in the header, written by whatever
    # formatting produced it, so it is matched as a number rather than by text.
    columns = deltas.columns.to_frame(index=False)
    columns["density"] = columns.density.astype(float)

    order = {name: i for i, name in enumerate(PREDICTORS_USED)}
    source = (source.assign(_p=source.predictor.map(order))
              .sort_values(["_p", "density"]).drop(columns="_p"))
    rows = []
    for record in source.itertuples():
        take = ((columns.predictor == record.predictor)
                & np.isclose(columns.density, record.density)).to_numpy()
        assert take.sum() == len(config.SPLIT_SEEDS), f"{record.predictor} 열 수 이상"
        # base minus corrected, so a positive difference is an improvement,
        # and the seeds collapse before the interval is taken.
        paired = -deltas.loc[:, take].mean(axis=1).to_numpy()
        # Seeded per row rather than from one stream, so a row's interval does
        # not depend on how many rows were drawn before it -- reordering the
        # table, or building it alone, leaves the numbers alone.
        rng = np.random.default_rng(
            [args.seed, order[record.predictor], round(record.density * 100)])
        low, high, _, _ = interval(paired, args.draws, rng)
        rows.append({
            "Base predictor": record.predictor,
            "Density": f"{record.density:.0%}",
            "Base mean regret": fmt(record.vanilla),
            "CoRC mean regret": fmt(record.corrected),
            "Regret reduction (%)": f"{100 * record.drop:.1f}",
            "95% CI of regret difference": f"[{low:.5f}, {high:.5f}]",
            "Holm-adjusted p": pvalue(record.holm)})
    rows.append({
        "Base predictor": "Overall (macro-average)", "Density": "—",
        "Base mean regret": fmt(source.vanilla.mean()),
        "CoRC mean regret": fmt(source.corrected.mean()),
        "Regret reduction (%)":
            f"{100 * (1 - source.corrected.mean() / source.vanilla.mean()):.1f}",
        "95% CI of regret difference": "—",
        "Holm-adjusted p": "—"})

    table = pd.DataFrame(rows)
    table.to_csv(args.out_dir / "table_1_regret.csv", index=False)
    log("Table 1 완료")


def table_2(args, mae: pd.DataFrame) -> None:
    """QoS prediction accuracy, on the held-out entry basis."""
    rows = []
    for predictor, density in itertools.product(PREDICTORS_USED,
                                                sorted(config.DENSITIES)):
        block = mae[(mae.predictor == predictor) & np.isclose(mae.density, density)]
        row = {"Base predictor": predictor, "Density": f"{density:.0%}"}
        for qos, label, digits in (("rt", "RT", 4), ("tp", "TP", 3)):
            got = block[block.qos_type == qos]
            van, cor = float(got.vanilla.mean()), float(got.corc.mean())
            row[f"Base {label} MAE"] = fmt(van, digits)
            row[f"CoRC {label} MAE"] = fmt(cor, digits)
            row[f"{label} MAE reduction (%)"] = f"{100 * (1 - cor / van):.1f}"
        rows.append(row)
    table = pd.DataFrame(rows)
    table.to_csv(args.out_dir / "table_2_mae.csv", index=False)
    log("Table 2 완료")


def _significance(per: dict, keys: list[str], reference: str) -> dict:
    """How many predictors separate from the reference, and the worst p."""
    def for_predictor(key: str, predictor: str) -> pd.Series:
        stack = [per[(predictor, density, seed, key)]
                 for density, seed in itertools.product(sorted(config.DENSITIES),
                                                        config.SPLIT_SEEDS)]
        return pd.concat(stack, axis=1).mean(axis=1)

    raw = {}
    for predictor, key in itertools.product(PREDICTORS_USED, keys):
        gap = for_predictor(key, predictor) - for_predictor(reference, predictor)
        raw[(predictor, key)] = stats.wilcoxon(gap, np.zeros(len(gap))).pvalue
    adjusted = holm(raw)
    return {key: (f"{sum(1 for p in PREDICTORS_USED if adjusted[(p, key)] < 0.05)}"
                  f"/{len(PREDICTORS_USED)}",
                  pvalue(max(adjusted[(p, key)] for p in PREDICTORS_USED)))
            for key in keys}


def table_3(args, per: dict, constant_mae: pd.DataFrame, mae: pd.DataFrame) -> None:
    """What the correction mechanism buys, at fixed inputs."""
    pooled = lambda key: float(pd.concat(
        [per[k] for k in per if k[3] == key], axis=1).mean(axis=1).mean())
    error = lambda column, qos: float(mae[mae.qos_type == qos][column].mean())
    reference = pooled("corc")
    ladder = _significance(per, ["vanilla", "constant", "linear"], "corc")

    rows = []
    for key, label, mechanism, params in (
            ("vanilla", "None (Base)", "—", "0"),
            ("constant", "Constant correction", "one scalar per QoS", "1"),
            ("linear", "Linear correction", "linear map over 17 inputs", "18"),
            ("corc", "CoRC", "two-layer MLP over 17 inputs", "3,265")):
        regret = pooled(key)
        rt = constant_mae.rt.mean() if key == "constant" else error(key, "rt")
        tp = constant_mae.tp.mean() if key == "constant" else error(key, "tp")
        rows.append({
            "Correction": label, "Mechanism": mechanism, "Params per QoS": params,
            "RT MAE": fmt(rt, 4), "TP MAE": fmt(tp, 2), "Regret": fmt(regret),
            "ΔRegret vs CoRC (%)": "—" if key == "corc"
                else f"{100 * (regret / reference - 1):+.1f}",
            "Sig.": "—" if key == "corc" else ladder[key][0],
            "Weakest Holm p": "—" if key == "corc" else ladder[key][1]})
    table = pd.DataFrame(rows)
    table.to_csv(args.out_dir / "table_3_correction_ladder.csv", index=False)
    log("Table 3 완료")


def table_4(args, per: dict, mae: pd.DataFrame) -> None:
    """What the collaborative evidence buys, at a fixed mechanism."""
    pooled = lambda key: float(pd.concat(
        [per[k] for k in per if k[3] == key], axis=1).mean(axis=1).mean())
    error = lambda column, qos: float(mae[mae.qos_type == qos][column].mean())
    reference = pooled("corc")
    evidence = _significance(per, ["core_direction", "core_reliability",
                                   "placebo", "core_only"], "corc")

    rows = []
    for key, label, dim, ticks in (
            ("corc", "Core + Direction + Reliability (CoRC)", 17, ("✓", "✓", "✓")),
            ("core_direction", "Core + Direction", 5, ("✓", "", "")),
            ("core_reliability", "Core + Reliability", 13, ("", "✓", "✓")),
            ("placebo", "Core + shuffled evidence", 17, ("~", "~", "~")),
            ("core_only", "Core only", 1, ("", "", ""))):
        regret = pooled(key)
        rows.append({
            "Variant": label, "Dir.": ticks[0], "Sim.": ticks[1], "Disp.": ticks[2],
            "Dim.": dim, "RT MAE": fmt(error(key, "rt"), 4),
            "TP MAE": fmt(error(key, "tp"), 2), "Regret": fmt(regret),
            "ΔRegret (%)": "—" if key == "corc"
                else f"{100 * (regret / reference - 1):+.1f}",
            "Sig.": "—" if key == "corc" else evidence[key][0],
            "Weakest Holm p": "—" if key == "corc" else evidence[key][1]})
    table = pd.DataFrame(rows)
    table.to_csv(args.out_dir / "table_4_evidence_ladder.csv", index=False)
    log("Table 4 완료")


def table_6(args) -> None:
    """Build the validation-only structure-search table for Final-17."""
    root = args.structure_dir
    regret = pd.read_csv(root / "structure_regret_3seeds.csv")
    key = ["predictor", "seed"]
    base_regret = regret[regret.condition == "full"].set_index(key).regret

    panels = [
        ("A. Collaborative source", "ablation", [
            ("wo_general_cf", "General CF (UCF, SCF)", "Retain"),
            ("wo_geography", "Geography (Geo-UCF, Geo-SCF)", "Retain"),
            ("wo_service_side", "Service-side probes", "Retain"),
            ("wo_user_side", "User-side probes", "Retain"),
            ("wo_as", "AS (AS-UCF, AS-SCF)", "Remove")]),
        ("B. Reliability evidence", "ablation_kinds", [
            ("wo_similarity", "Similarity (sum, maximum)", "Retain"),
            ("wo_dispersion", "Dispersion (neighbor QoS variance)", "Retain"),
            ("wo_support", "Support (count, availability)", "Remove")]),
        ("C. Combined removal", "combine", [
            ("wo_as_support", "AS + Support", "Final-17")]),
    ]
    sources = {name: pd.read_csv(root / f"{name}.csv")
               for name in ("ablation", "ablation_kinds", "combine")}
    for frame in sources.values():
        assert frame.seed.nunique() == len(config.SPLIT_SEEDS), "seed 부족"
    reference = sources["ablation"]
    full = reference[reference.condition == "full"]
    rows = [{"Panel": "—", "Removed component": "Full",
             "#Feat.": int(full.n_features.iloc[0]),
             "R_MAE": fmt(float(full.ratio.mean()), 4),
             "ΔR_MAE": "—", "Worse cells": "—",
             "Val. regret": fmt(float(base_regret.mean())),
             "Decision": "Reference"}]

    for panel, source, spec in panels:
        block = sources[source]
        paired = (block.groupby(["condition"] + key).ratio.mean()
                  .rename("R").reset_index())
        base = paired[paired.condition == "full"].set_index(key).R
        for condition, label, decision in spec:
            cell = block[block.condition == condition]
            assert len(cell), f"{source}: {condition} 없음"
            gap = (paired[paired.condition == condition].set_index(key).R
                   - base).dropna()
            got = (regret[regret.condition == condition]
                   .set_index(key).regret).dropna()
            rows.append({
                "Panel": panel, "Removed component": label,
                "#Feat.": int(cell.n_features.iloc[0]),
                "R_MAE": fmt(float(cell.ratio.mean()), 4),
                "ΔR_MAE": f"{gap.mean():+.4f}",
                "Worse cells": f"{int((gap > 0).sum())}/{len(gap)}",
                "Val. regret": fmt(float(got.mean())),
                "Decision": decision})

    table = pd.DataFrame(rows)
    table.to_csv(args.out_dir / "table_6_structure_search.csv", index=False)
    log("Table 6 완료")


def main() -> int:
    if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
        sys.stdout.reconfigure(encoding="utf-8")
    final = config.OUTPUT_DIR / "final_method" / config.SPLIT_VERSION
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tables", nargs="+", default=["1", "2", "3", "4", "6"],
                        choices=["1", "2", "3", "4", "6"])
    parser.add_argument("--final-dir", type=Path, default=final)
    parser.add_argument("--rq1-dir", type=Path,
                        default=config.RESULTS_DIR / "table_1")
    parser.add_argument("--structure-dir", type=Path,
                        default=config.RESULTS_DIR / "table_6")
    parser.add_argument("--dataset-dir", type=Path,
                        default=config.OUTPUT_DIR / "selection_dataset" / config.SPLIT_VERSION)
    parser.add_argument("--out-dir", type=Path,
                        default=config.RESULTS_DIR / "generated")
    parser.add_argument("--n-candidates", type=int, default=K)
    parser.add_argument("--refresh-mae", action="store_true",
                        help="recompute held-out entry MAE instead of reusing the cache")
    parser.add_argument("--refresh-regret", action="store_true",
                        help="recompute the sixty cells of per-user regret")
    parser.add_argument("--draws", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20261010)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    want = set(args.tables)

    mae = None
    if want & {"2", "3", "4"}:
        cached = config.RESULTS_DIR / "table_2_3_4" / "mae_test_entries.csv"
        # Sixty cells times two QoS types of joining, so it is cached rather
        # than recomputed for every table that needs it.
        mae = (pd.read_csv(cached) if cached.exists() and not args.refresh_mae
               else test_mae(args.final_dir, cached))
    per = constant_mae = None
    if want & {"3", "4"}:
        # Sixty cells of per-user regret, shared by both ladders. Reducing them
        # from the predictions takes minutes and needs `outputs/`, so the result
        # is cached under `results/` and read from there by default.
        cache = config.RESULTS_DIR / "table_2_3_4"
        if (cache / "regret_per_user.csv").exists() and not args.refresh_regret:
            per, constant_mae = load_regret(cache)
        else:
            per, constant_mae = regret_arms(args.final_dir, args.dataset_dir)
            save_regret(per, constant_mae, cache)

    if "1" in want:
        table_1(args)
    if "2" in want:
        table_2(args, mae)
    if "3" in want:
        table_3(args, per, constant_mae, mae)
    if "4" in want:
        table_4(args, per, mae)
    if "6" in want:
        table_6(args)
    log(f"-> {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
