"""Bootstrap the percentage reduction in server-selection regret by resampling
paired per-user base and corrected results."""
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
from corc.experiment_data import density_tag, entry_predictions, load_candidates
from corc.evaluation import per_user_regret


def main() -> int:
    if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
        sys.stdout.reconfigure(encoding="utf-8")
    final = config.OUTPUT_DIR / "final_method" / config.SPLIT_VERSION
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictors", nargs="+",
                        default=["UIPCC", "PMF", "BiasMF", "LDCF", "NFMF"])
    parser.add_argument("--predictions", type=Path, default=final / "test")
    parser.add_argument("--dataset-dir", type=Path,
                        default=config.OUTPUT_DIR / "selection_dataset" / config.SPLIT_VERSION)
    parser.add_argument("--out", type=Path, default=final / "k10" / "rq1_test_reduction_ci.csv")
    parser.add_argument("--n-candidates", type=int, default=10)
    parser.add_argument("--draws", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20261013)
    args = parser.parse_args()

    k = args.n_candidates
    arms = {}
    for predictor, density, seed in itertools.product(
            args.predictors, sorted(config.DENSITIES), config.SPLIT_SEEDS):
        frame = load_candidates("test", density, seed, k, args.dataset_dir)
        ordered = frame.sort_values(["sample_id", "candidate_rank"])
        vanilla = {qos: ordered[f"pred_{predictor}_{qos}"].to_numpy(float)
                   for qos in config.QOS_TYPES}
        corrected = entry_predictions(args.predictions, predictor, "final",
                                      density_tag(density), seed, "test", ordered)
        arms[(predictor, density, seed, "vanilla")] = per_user_regret(frame, vanilla, k)
        arms[(predictor, density, seed, "corrected")] = per_user_regret(frame, corrected, k)
        print(f"[{time.strftime('%H:%M:%S')}] {predictor:7s} d={density:.0%} seed {seed}",
              flush=True)

    rng = np.random.default_rng(args.seed)
    rows = []
    for predictor, density in itertools.product(args.predictors, sorted(config.DENSITIES)):
        # The three fits are repetitions of one condition, so a user's regret is
        # averaged over them before the resampling begins.
        base = pd.concat([arms[(predictor, density, s, "vanilla")]
                          for s in config.SPLIT_SEEDS], axis=1).mean(axis=1).to_numpy()
        got = pd.concat([arms[(predictor, density, s, "corrected")]
                         for s in config.SPLIT_SEEDS], axis=1).mean(axis=1).to_numpy()
        n = len(base)
        index = rng.integers(0, n, size=(args.draws, n))
        # Both arms follow the same index: the pair is the unit, not the arm.
        ratio = 1 - got[index].mean(axis=1) / base[index].mean(axis=1)
        low, high = np.quantile(ratio, [0.025, 0.975])
        observed = 1 - got.mean() / base.mean()
        test = stats.wilcoxon(got - base, np.zeros(n))
        rows.append({"predictor": predictor, "density": density, "n_users": n,
                     "vanilla": base.mean(), "corrected": got.mean(),
                     "reduction": 100 * observed,
                     "low": 100 * float(low), "high": 100 * float(high),
                     "excludes_zero": bool(low > 0), "p": test.pvalue})

    # The twenty conditions share one set of users, so the average row can be
    # resampled the same way: a user's regret is pooled over the conditions
    # first and the ratio is formed inside each resample.
    base = pd.concat([arms[k] for k in arms if k[3] == "vanilla"],
                     axis=1).mean(axis=1).to_numpy()
    got = pd.concat([arms[k] for k in arms if k[3] == "corrected"],
                    axis=1).mean(axis=1).to_numpy()
    index = rng.integers(0, len(base), size=(args.draws, len(base)))
    ratio = 1 - got[index].mean(axis=1) / base[index].mean(axis=1)
    low, high = np.quantile(ratio, [0.025, 0.975])
    test = stats.wilcoxon(got - base, np.zeros(len(base)))
    rows.append({"predictor": "Average", "density": np.nan, "n_users": len(base),
                 "vanilla": base.mean(), "corrected": got.mean(),
                 "reduction": 100 * (1 - got.mean() / base.mean()),
                 "low": 100 * float(low), "high": 100 * float(high),
                 "excludes_zero": bool(low > 0), "p": test.pvalue})
    print()
    print(f"전체 평균 행: {base.mean():.4f} -> {got.mean():.4f}  "
          f"{100 * (1 - got.mean() / base.mean()):.1f}% "
          f"[{100 * low:.1f}, {100 * high:.1f}]  p={test.pvalue:.3e}")

    table = pd.DataFrame(rows)
    table.to_csv(args.out, index=False)
    pd.set_option("display.width", 200)
    print()
    print(f"감소율의 95% CI — 사용자 단위 paired bootstrap {args.draws:,}회")
    print(f"{'예측기':9s}{'밀도':>6s}{'Base':>9s}{'CoRC':>9s}{'감소율':>8s}{'95% CI':>18s}")
    for row in table.itertuples():
        print(f"{row.predictor:9s}{row.density:>6.0%}{row.vanilla:>9.4f}"
              f"{row.corrected:>9.4f}{row.reduction:>8.1f}"
              f"  [{row.low:>5.1f}, {row.high:>5.1f}]")
    print()
    print(f"하한이 0을 넘는 조건 {int(table.excludes_zero.sum())}/{len(table)}")
    overall_base = table.vanilla.mean()
    overall_got = table.corrected.mean()
    print(f"전체 평균 {overall_base:.4f} -> {overall_got:.4f}  "
          f"({100 * (1 - overall_got / overall_base):.1f}%)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
