"""Measure validation server-selection regret for the already-trained structure
conditions. No model is refitted."""
from __future__ import annotations

import argparse
import itertools
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from corc import config
from corc.experiment_data import density_tag
from corc.evaluation import kendall, pair_error
from corc.server_selection import WEIGHT_COLUMNS, score

KEY = ["user_id", "service_id"]


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def measure(frame: pd.DataFrame, predicted: dict, k: int) -> dict:
    """Compute prediction, ranking, and regret metrics for one candidate set."""
    n = len(frame) // k
    grid = lambda col: torch.as_tensor(
        frame[col].to_numpy(dtype=np.float64).reshape(n, k), dtype=torch.float32)
    weights = {key: grid(col) for key, col in WEIGHT_COLUMNS.items()}
    ld, fr = grid("LD"), grid("FR")
    truth = score(grid("true_rt"), grid("true_tp"), ld, fr, weights)
    guess = score(torch.as_tensor(predicted["rt"].reshape(n, k), dtype=torch.float32),
                  torch.as_tensor(predicted["tp"].reshape(n, k), dtype=torch.float32),
                  ld, fr, weights)
    row = torch.arange(n)
    best = truth.argmax(1)
    pick = guess.argmax(1)
    per_request = (truth[row, best] - truth[row, pick]).numpy()
    masked = truth.clone()
    masked[row, best] = -float("inf")
    second = masked.argmax(1)
    best_guess = guess[row, best]

    users = frame.groupby("sample_id", sort=True)["user_id"].first().to_numpy()
    counts = pd.Series(users).map(pd.Series(users).value_counts()).to_numpy()
    weight = 1.0 / counts
    out = {}
    for qos in config.QOS_TYPES:
        true_grid = frame[f"true_{qos}"].to_numpy(float).reshape(n, k)
        guess_grid = predicted[qos].reshape(n, k)
        out[f"absolute_{qos}"] = float(np.abs(guess_grid - true_grid).mean())
        out[f"relative_{qos}"] = pair_error(guess_grid, true_grid)
    truth_np, guess_np = truth.numpy(), guess.numpy()
    margin_true = truth[row, best] - truth[row, second]
    margin_guess = guess[row, best] - guess[row, second]
    out.update({
        "score_absolute": float(np.abs(guess_np - truth_np).mean()),
        "score_relative": pair_error(guess_np, truth_np),
        "kendall": kendall(guess_np, truth_np),
        "critical_margin_error": float((margin_guess - margin_true).abs().mean()),
    })
    return {**out,
            "regret": float((per_request * weight).sum() / weight.sum()),
            "regret_instance": float(per_request.mean()),
            "top1": float((pick == best).float().mean()),
            "cpa": float((best_guess > guess[row, second]).float().mean()),
            "opa": float(((best_guess[:, None] > guess).sum(1) / (k - 1)).float().mean())}


def main() -> int:
    if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--structure-dir", type=Path,
                        default=config.OUTPUT_DIR / "structure_ad" / config.SPLIT_VERSION)
    parser.add_argument("--dataset-dir", type=Path,
                        default=config.OUTPUT_DIR / "selection_dataset" / config.SPLIT_VERSION)
    parser.add_argument("--out-dir", type=Path,
                        default=config.OUTPUT_DIR / "structure_ad" / config.SPLIT_VERSION)
    parser.add_argument("--predictors", nargs="+",
                        default=["UIPCC", "PMF", "BiasMF", "LDCF"])
    parser.add_argument("--density", type=float, default=0.10)
    # Several seeds are averaged the way the main table averages them: each
    # seed is a repetition of the same condition, not an extra observation.
    parser.add_argument("--seeds", nargs="+", type=int, default=[42])
    parser.add_argument("--n-candidates", type=int, default=10)
    # The structure search ran on validation and stays there. Pointing this at
    # test is for the four conditions the paper reports as its ablation, whose
    # predictions were written once on the held-out split.
    parser.add_argument("--role", default="validation",
                        choices=["validation", "test"])
    parser.add_argument("--out-name", default="structure_regret.csv")
    args = parser.parse_args()

    tag = density_tag(args.density)
    k, role = args.n_candidates, args.role
    rows = []
    for seed in args.seeds:
        pattern = re.compile(rf"^(?P<predictor>[A-Za-z]+)_(?P<qos>rt|tp)_(?P<condition>.+)"
                             rf"_{tag}_seed{seed}_{role}\.csv$")
        found: dict[tuple[str, str], dict[str, Path]] = defaultdict(dict)
        for path in sorted(args.structure_dir.glob(f"*_{tag}_seed{seed}_{role}.csv")):
            match = pattern.match(path.name)
            if match and match["predictor"] in args.predictors:
                found[(match["predictor"], match["condition"])][match["qos"]] = path
        complete = {key: value for key, value in found.items()
                    if set(value) == set(config.QOS_TYPES)}
        log(f"seed {seed}: 조건 {len(complete)}개 "
            f"({len(set(c for _, c in complete))}가지 구조)")

        for predictor in args.predictors:
            # The test candidate file carries no role in its name -- it is the
            # original one -- so the suffix follows the same rule as RQ1.
            suffix = "" if role == "test" else f"_{role}"
            frame = pd.read_csv(args.dataset_dir
                                / f"candidates_k{k}_{tag}{suffix}_seed{seed}.csv")
            frame = frame.sort_values(["sample_id", "candidate_rank"]).reset_index(drop=True)
            vanilla = {qos: frame[f"pred_{predictor}_{qos}"].to_numpy(float)
                       for qos in config.QOS_TYPES}
            base = measure(frame, vanilla, k)
            base_mae = {qos: float(np.abs(vanilla[qos] - frame[f"true_{qos}"]).mean())
                        for qos in config.QOS_TYPES}
            rows.append({"predictor": predictor, "seed": seed,
                         "condition": "vanilla", "R_MAE": 1.0, **base})
            for (name, condition), paths in sorted(complete.items()):
                if name != predictor:
                    continue
                predicted, mae = {}, {}
                for qos, path in paths.items():
                    lookup = pd.read_csv(path).drop_duplicates(KEY).set_index(KEY)["pred"]
                    values = frame.set_index(KEY).index.map(lookup).to_numpy(float)
                    assert not np.isnan(values).any(), f"{path.name}: 후보쌍 누락"
                    predicted[qos] = values
                    mae[qos] = float(np.abs(values - frame[f"true_{qos}"]).mean())
                rows.append({"predictor": predictor, "seed": seed,
                             "condition": condition,
                             "R_MAE": 0.5 * sum(mae[q] / base_mae[q] for q in config.QOS_TYPES),
                             **measure(frame, predicted, k)})
            log(f"  seed {seed} {predictor} 완료")

    cells = pd.DataFrame(rows)
    cells.to_csv(args.out_dir / args.out_name, index=False)
    # The seeds collapse before anything is compared, so a condition and its
    # base reference are always averaged over the same three fits.
    table = (cells.groupby(["predictor", "condition"], as_index=False).mean()
             .drop(columns="seed"))

    pd.set_option("display.width", 220)
    columns = ["R_MAE", "score_absolute", "score_relative", "critical_margin_error",
               "kendall", "opa", "cpa", "top1", "regret"]
    summary = table.groupby("condition")[columns].mean()
    base = summary.loc["vanilla"]
    summary["regret_drop"] = 1 - summary.regret / base.regret
    order = ["vanilla", "core_only", "core_direction", "core_reliability",
             "A_base", "B_direction", "C_reliability", "D_full",
             "wo_user_side", "wo_service_side", "wo_general_cf", "wo_geography",
             "wo_as", "wo_support", "wo_similarity", "wo_dispersion",
             "wo_as_support", "final"]
    print()
    print(f"구조 조건별 — {args.role}, 밀도 {args.density:.0%}, "
          f"seed {args.seeds}, 예측기 {len(args.predictors)}개 평균")
    print(summary.reindex([c for c in order if c in summary.index])
          .to_string(float_format=lambda x: f"{x:.4f}"))
    # Paired with the base cell by cell, so a measure that merely ranks the
    # predictors cannot score well by accident.
    base = table[table.condition == "vanilla"].set_index("predictor")
    moved = table[table.condition != "vanilla"].copy()
    for column in columns:
        reference = moved.predictor.map(base[column])
        moved[column] = (moved[column] / reference - 1 if column in
                         ("R_MAE", "score_absolute", "score_relative",
                          "critical_margin_error", "regret")
                         else moved[column] - reference)
    print()
    print(f"Correlation between change against the base and change in regret "
          f"(조건 {moved.condition.nunique()}개 x 예측기 {moved.predictor.nunique()}개 "
          f"= {len(moved)}쌍)")
    for column in columns[:-1]:
        rho = moved[[column, "regret"]].corr(method="spearman").iloc[0, 1]
        print(f"  {column:22s} Spearman {rho:+.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
