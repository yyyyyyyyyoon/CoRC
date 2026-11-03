"""Reduce the versioned paper summaries to the CSV rows plotted in Figures 3 and
4. The plotting itself is performed outside this repository."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from corc import config

# The main setting has no sweep directory of its own: it is the run the rest of
# the paper uses, so its value is read from that run. Both references are
# chosen to cover exactly the cells their sweep covers -- the neighbourhood
# sweep is density 10% over five predictors, the capacity sweep is all four
# densities -- because a curve whose reference averages a different set of
# cells bends for reasons that have nothing to do with the swept parameter.
NEIGHBOURS = [5, 10, 20, 50, 100, 200]
MAIN_N = 20
CAPACITY = [("linear", "Linear", 18), ("16", "(16)", 305),
            ("32x16", "(32, 16)", 1_137), ("64x32", "(64, 32)", 3_265),
            ("128x64", "(128, 64)", 10_689)]
MAIN_CAPACITY = "(64, 32)"
# The main run's summary. It sits in `test/` because that run predicted the
# test role, but the error it summarises is the validation one, which is
# what this figure wants -- every row carries `mae_role` saying so. The
# held-out entry error for the same run is `mae_test_entries.csv`, and that
# is what the tables report.
CONDITIONS = [
    ("global_shift", "Train-derived global shift"),
    ("oracle_shift", "Oracle candidate-wise shift"),
    ("corc_common", "CoRC common component"),
    ("corc_relative", "CoRC relative component"),
    ("corc", "Full CoRC"),
]


def r_mae(frame: pd.DataFrame) -> float:
    """Average per-cell validation MAE ratios against the base predictor."""
    if "mae_role" in frame.columns:
        roles = set(frame.mae_role.unique())
        assert roles == {"validation"}, f"validation 이 아닌 기준: {roles}"
    return float(frame.groupby(["predictor", "seed", "qos_type"]).ratio.mean().mean())


def figure_3(rq3_dir: Path, out_dir: Path) -> None:
    source = pd.read_csv(rq3_dir / "rq3_controls.csv")
    base = source[source.condition == "vanilla"]
    key = ["predictor", "density", "seed"]
    rows = []
    for condition, label in CONDITIONS:
        got = source[source.condition == condition].set_index(key)
        ref = base.set_index(key)
        # Ratio of means over the sixty cells, matching every other relative
        # change the paper reports.
        change = lambda col: 100 * (got[col].mean() / ref[col].mean() - 1)
        rows.append({
            "condition": label,
            "d_mae_rt_pct": round(change("absolute_rt"), 2),
            "d_mae_tp_pct": round(change("absolute_tp"), 2),
            "d_relative_rt_pct": round(change("relative_rt"), 2),
            "d_relative_tp_pct": round(change("relative_tp"), 2),
            "d_regret_pct": round(change("regret"), 2),
            "top1": round(float(got.top1.mean()), 4),
            "cpa": round(float(got.cpa.mean()), 4),
            "opa": round(float(got.opa.mean()), 4),
            "cells": len(got)})
    frame = pd.DataFrame(rows)
    frame.to_csv(out_dir / "figure_3_decomposition.csv", index=False)
    print(frame.to_string(index=False))


def figure_4a(source: Path, out_dir: Path) -> None:
    rows = []
    for n in NEIGHBOURS:
        frame = pd.read_csv(source / f"neighbour_n{n}.csv")
        # The n = 20 file is the main run, which swept structures as well, so
        # only its final-structure rows belong on this curve.
        if "condition" in frame.columns:
            frame = frame[frame.condition == "final"]
        rows.append({"n": n, "r_mae": round(r_mae(frame), 4),
                     "is_main": n == MAIN_N, "cells": len(frame)})
    frame = pd.DataFrame(rows)
    assert frame.cells.nunique() == 1, f"셀 수 불일치: {frame.cells.tolist()}"
    frame.to_csv(out_dir / "figure_4a_neighbours.csv", index=False)
    print(frame.to_string(index=False))


def figure_4b(source: Path, out_dir: Path) -> None:
    rows = []
    for folder, label, params in CAPACITY:
        frame = pd.read_csv(source / f"capacity_{folder}.csv")
        rows.append({"hidden": label, "params": params,
                     "r_mae": round(r_mae(frame), 4),
                     "is_main": label == MAIN_CAPACITY, "cells": len(frame)})
    frame = pd.DataFrame(rows)
    assert frame.cells.nunique() == 1, f"셀 수 불일치: {frame.cells.tolist()}"
    frame.to_csv(out_dir / "figure_4b_capacity.csv", index=False)
    print(frame.to_string(index=False))


def main() -> int:
    if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--figures", nargs="+", default=["3", "4a", "4b"],
                        choices=["3", "4a", "4b"])
    parser.add_argument("--rq3-dir", type=Path,
                        default=config.RESULTS_DIR / "figure_3")
    parser.add_argument("--sweep-dir", type=Path,
                        default=config.RESULTS_DIR / "figure_4")
    parser.add_argument("--out-dir", type=Path,
                        default=config.RESULTS_DIR / "generated")
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    if "3" in args.figures:
        print("=== Figure 3 ===")
        figure_3(args.rq3_dir, args.out_dir)
    if "4a" in args.figures:
        print("\n=== Figure 4(a) neighbourhood size ===")
        figure_4a(args.sweep_dir, args.out_dir)
    if "4b" in args.figures:
        print("\n=== Figure 4(b) corrector capacity ===")
        figure_4b(args.sweep_dir, args.out_dir)
    print(f"\n-> {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
