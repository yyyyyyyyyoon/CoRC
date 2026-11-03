"""Run RQ3 controls that separate entry-level error changes from changes in
server-selection decisions using common shifts and error reassignment."""
from __future__ import annotations

import argparse
import itertools
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from corc import config
from corc.experiment_data import PREDICTORS, density_tag, load_candidates
from corc.evaluation import measure
from corc.server_selection import WEIGHT_COLUMNS, score

KEY = ["user_id", "service_id"]


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def grids(frame: pd.DataFrame, k: int) -> dict:
    """The candidate frame as (instance, candidate) arrays, sorted once."""
    frame = frame.sort_values(["sample_id", "candidate_rank"])
    n = frame["sample_id"].nunique()
    take = lambda col: frame[col].to_numpy(dtype=np.float64).reshape(n, k)
    return {"n": n,
            "users": frame.groupby("sample_id", sort=True)["user_id"].first().to_numpy(),
            "weights": {key: take(col) for key, col in WEIGHT_COLUMNS.items()},
            "ld": take("LD"), "fr": take("FR"),
            "true": {qos: take(f"true_{qos}") for qos in config.QOS_TYPES}}


def scores(pred: dict, g: dict) -> np.ndarray:
    as_t = lambda x: torch.as_tensor(x, dtype=torch.float32)
    return score(as_t(pred["rt"]), as_t(pred["tp"]), as_t(g["ld"]), as_t(g["fr"]),
                 {key: as_t(value) for key, value in g["weights"].items()}).numpy()


def decide(pred: dict, g: dict, truth: np.ndarray) -> dict:
    """Compute decision metrics, averaging regret per user before aggregation."""
    guess = scores(pred, g)
    row = np.arange(g["n"])
    best, pick = truth.argmax(1), guess.argmax(1)
    gap = truth[row, best] - truth[row, pick]
    per_user = pd.Series(gap, index=g["users"]).groupby(level=0).mean()
    return {"regret": float(per_user.mean()),
            "regret_instance": float(gap.mean()),
            "top1": float((pick == best).mean()),
            "per_user": per_user}


def decomposition(pred: dict, g: dict, k: int) -> dict:
    """The exact squared-error split into a common and a relative part."""
    out = {}
    for qos in config.QOS_TYPES:
        e = pred[qos] - g["true"][qos]
        mean = e.mean(axis=1, keepdims=True)
        total = float((e ** 2).sum())
        common = float((k * mean ** 2).sum())
        out[f"mse_{qos}"] = total / (g["n"] * k)
        out[f"common_share_{qos}"] = common / total
    return out


def main() -> int:
    if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path,
                        default=config.OUTPUT_DIR / "selection_dataset" / config.SPLIT_VERSION)
    parser.add_argument("--corrected-dir", type=Path,
                        default=config.OUTPUT_DIR / "final_method" / config.SPLIT_VERSION / "test")
    parser.add_argument("--out-dir", type=Path,
                        default=config.OUTPUT_DIR / "final_method" / config.SPLIT_VERSION / "rq3")
    parser.add_argument("--predictors", nargs="+",
                        default=["UIPCC", "PMF", "BiasMF", "LDCF", "NFMF"])
    parser.add_argument("--densities", nargs="+", type=float,
                        default=sorted(config.DENSITIES))
    parser.add_argument("--seeds", nargs="+", type=int, default=config.SPLIT_SEEDS)
    parser.add_argument("--n-candidates", type=int, default=10)
    parser.add_argument("--predictor-root", type=Path,
                        default=config.OUTPUT_DIR / "base_predictors")
    parser.add_argument("--version", default=config.SPLIT_VERSION)
    parser.add_argument("--condition", default="final")
    parser.add_argument("--role", default="test", choices=["validation", "test"])
    parser.add_argument("--permutations", type=int, default=200)
    parser.add_argument("--permutation-seed", type=int, default=20261010)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    k = args.n_candidates
    rows, draws = [], []
    observed_users, permuted_users = [], []
    for predictor, density, seed in itertools.product(args.predictors, args.densities,
                                                      args.seeds):
        tag = density_tag(density)
        frame = load_candidates(args.role, density, seed, k, args.dataset_dir)
        g = grids(frame, k)
        truth = scores(g["true"], g)
        ordered = frame.sort_values(["sample_id", "candidate_rank"])

        vanilla = {qos: ordered[f"pred_{predictor}_{qos}"].to_numpy(float).reshape(g["n"], k)
                   for qos in config.QOS_TYPES}
        corrected = {}
        for qos in config.QOS_TYPES:
            path = (args.corrected_dir / f"{predictor}_{qos}_{args.condition}"
                                         f"_{tag}_seed{seed}_{args.role}.csv")
            lookup = pd.read_csv(path).drop_duplicates(KEY).set_index(KEY)["pred"]
            values = ordered.set_index(KEY).index.map(lookup).to_numpy(float)
            assert not np.isnan(values).any(), f"{path.name}: 후보쌍 누락"
            corrected[qos] = values.reshape(g["n"], k)

        # The global shift is read off the predictor's own training-split
        # residuals -- the same out-of-fold predictions the corrector is fitted
        # on -- so nothing about it depends on the split it is applied to. Not
        # the training candidate rows: NFMF was added to that file after it
        # was built and its columns there are entirely missing.
        shift = {}
        for qos in config.QOS_TYPES:
            family, prefix = PREDICTORS[predictor]
            path = (args.predictor_root / family / args.version / qos / f"seed{seed}"
                    / tag / f"{prefix}_train_predictions.csv")
            train = pd.read_csv(path, usecols=["y_true", "pred"])
            residual = train["pred"] - train["y_true"]
            assert residual.notna().all(), f"{path}: 잔차 결측"
            shift[qos] = -float(residual.median())
        global_shift = {qos: vanilla[qos] + shift[qos] for qos in config.QOS_TYPES}
        # The per-set shift that minimises absolute error is the negative
        # median of that set's signed residuals.
        oracle = {qos: vanilla[qos] - np.median(vanilla[qos] - g["true"][qos],
                                                axis=1, keepdims=True)
                  for qos in config.QOS_TYPES}

        # The correction CoRC actually applied, split the way the score sees
        # it. Within one candidate set the common part moves every candidate
        # alike and the min-max cancels it; only the part that differs between
        # candidates can reach the ordering. Nothing is clipped between the
        # correction and the normalisation, so the split is exact.
        delta = {qos: corrected[qos] - vanilla[qos] for qos in config.QOS_TYPES}
        common = {qos: vanilla[qos] + delta[qos].mean(axis=1, keepdims=True)
                  for qos in config.QOS_TYPES}
        relative = {qos: vanilla[qos] + (delta[qos]
                                         - delta[qos].mean(axis=1, keepdims=True))
                    for qos in config.QOS_TYPES}
        conditions = {"vanilla": vanilla,
                      "global_shift": global_shift, "oracle_shift": oracle,
                      "corc_common": common, "corc_relative": relative,
                      "corc": corrected}
        for name, pred in conditions.items():
            got, _ = measure(frame, pred["rt"].ravel(), pred["tp"].ravel(), k)
            got["regret"] = decide(pred, g, truth)["regret"]
            rows.append({"predictor": predictor, "density": density, "seed": seed,
                         "condition": name, **got, **decomposition(pred, g, k),
                         "shift_rt": shift["rt"] if name == "global_shift" else np.nan,
                         "shift_tp": shift["tp"] if name == "global_shift" else np.nan})

        # Error reassignment. One permutation per instance per replicate, the
        # same one for RT and TP so a candidate keeps a coherent pair.
        rng = np.random.default_rng(args.permutation_seed + seed)
        base = decide(vanilla, g, truth)
        observed_users.append(base["per_user"].rename((predictor, density, seed)))
        replicate_users = []
        error = {qos: vanilla[qos] - g["true"][qos] for qos in config.QOS_TYPES}
        rowsel = np.arange(g["n"])[:, None]
        regrets = np.empty(args.permutations)
        for b in range(args.permutations):
            order = rng.random((g["n"], k)).argsort(axis=1)
            moved = {qos: g["true"][qos] + error[qos][rowsel, order]
                     for qos in config.QOS_TYPES}
            if b == 0:
                # The invariant is the stronger one: each candidate set keeps
                # its own multiset of signed errors, bit for bit. Comparing the
                # computed MAE instead would fail at the last bit, because the
                # same ten numbers summed in a different order do not give the
                # same float -- about 6e-16 relative, which is round-off and
                # not a property of the construction.
                for qos in config.QOS_TYPES:
                    assert np.array_equal(np.sort(error[qos], axis=1),
                                          np.sort(error[qos][rowsel, order], axis=1)),                         "후보집합 내 오차 다중집합이 보존되지 않음"
                assert np.array_equal(np.sort(order, axis=1),
                                      np.tile(np.arange(k), (g["n"], 1))),                     "순열이 후보집합을 벗어남"
            got = decide(moved, g, truth)
            regrets[b] = got["regret"]
            replicate_users.append(got["per_user"].rename(b))
        block = pd.concat(replicate_users, axis=1)
        block.columns = pd.MultiIndex.from_product(
            [[predictor], [density], [seed], block.columns],
            names=["predictor", "density", "seed", "replicate"])
        permuted_users.append(block)
        draws.append({"predictor": predictor, "density": density, "seed": seed,
                      "permutations": args.permutations,
                      "permutation_seed": args.permutation_seed + seed,
                      "observed": base["regret"], "top1": base["top1"],
                      "shuffled_mean": float(regrets.mean()),
                      "shuffled_low": float(np.quantile(regrets, 0.025)),
                      "shuffled_high": float(np.quantile(regrets, 0.975)),
                      "shuffled_min": float(regrets.min()),
                      "shuffled_max": float(regrets.max()),
                      "spread": float(regrets.max() - regrets.min())})
        log(f"  {predictor:7s} d={density:.0%} seed {seed}")

    table = pd.DataFrame(rows)
    table.to_csv(args.out_dir / "rq3_controls.csv", index=False)
    shuffled = pd.DataFrame(draws)
    shuffled.to_csv(args.out_dir / "rq3_shuffle.csv", index=False)
    # One row per user: their regret averaged over the cells, observed and
    # under each replicate, so the permutation can be tested once rather than
    # sixty times.
    pd.concat(observed_users, axis=1).to_csv(
        args.out_dir / "rq3_shuffle_observed_users.csv")
    pd.concat(permuted_users, axis=1).to_csv(
        args.out_dir / "rq3_shuffle_permuted_users.csv")

    pd.set_option("display.width", 240)
    cells = len(table) // len(conditions)
    print()
    print(f"{cells}개 셀 평균 — {args.role}, K={k}")
    columns = ["absolute_rt", "absolute_tp", "score_absolute", "score_relative",
               "opa", "cpa", "top1", "regret_instance"]
    order = ["vanilla", "global_shift", "oracle_shift",
             "corc_common", "corc_relative", "corc"]
    summary = table.groupby("condition")[columns].mean().reindex(order)
    print(summary.to_string(float_format=lambda x: f"{x:.4f}"))
    print()
    print("Against the base predictions")
    base = summary.loc["vanilla"]
    for name in order[1:]:
        r = summary.loc[name]
        print(f"  {name:13s} RT MAE {r.absolute_rt / base.absolute_rt - 1:+7.2%}"
              f"   TP MAE {r.absolute_tp / base.absolute_tp - 1:+7.2%}"
              f"   top1 {r.top1 - base.top1:+.6f}"
              f"   regret {r.regret_instance / base.regret_instance - 1:+7.2%}")
    print()
    # Stated as a maximum over cells rather than an average: the claim is that
    # it holds everywhere, so the worst cell is the one that has to be shown.
    for name, against in (("global_shift", "vanilla"),
                          ("oracle_shift", "vanilla"),
                          ("corc_common", "vanilla"),
                          ("corc_relative", "corc")):
        pair = table[table.condition.isin([against, name])].pivot_table(
            index=["predictor", "density", "seed"], columns="condition",
            values=["regret_instance", "top1", "cpa", "opa"])
        gap = {field: float(np.abs(pair[(field, name)]
                                   - pair[(field, against)]).max())
               for field in ("regret_instance", "top1", "cpa", "opa")}
        print(f"{name:14s} vs {against:8s} 셀별 최대 차이 — "
              + "  ".join(f"{field} {value:.3e}" for field, value in gap.items()))
    print()
    print("공통/상대 제곱오차 분해 — 후보 집합 내 공통 성분 비중")
    share = table.groupby("condition")[["common_share_rt", "common_share_tp"]].mean()
    print(share.reindex(order).to_string(float_format=lambda x: f"{x:.3f}"))
    print()
    print(f"오차 재배치 — 후보 집합 내 순열 {args.permutations}회, signed error 다중집합 보존")
    merged = shuffled.copy()
    merged["relative_spread"] = merged.spread / merged.observed
    merged["absolute_increase"] = merged.shuffled_mean - merged.observed
    merged["relative_increase"] = merged.absolute_increase / merged.observed
    print(merged.groupby("predictor")[["observed", "shuffled_mean",
                                       "absolute_increase", "relative_increase"]]
          .mean().sort_values("relative_increase", ascending=False)
          .to_string(float_format=lambda x: f"{x:.5f}"))
    print("  relative_increase는 baseline regret이 작을수록 커지는 분모 효과를 "
          "포함하므로 absolute_increase와 함께 읽어야 한다")
    print()
    # The permutation distribution is the null "these error magnitudes, placed
    # at random". Where the observed regret falls against it says whether the
    # predictor's errors sit on the candidates in a way the choice benefits
    # from -- a question the error magnitude alone cannot answer.
    below = shuffled.observed < shuffled.shuffled_low
    above = shuffled.observed > shuffled.shuffled_high
    merged["gap"] = merged.shuffled_mean / merged.observed - 1
    print(f"관측 regret이 순열 95% 구간보다 낮은 셀 {int(below.sum())}/{len(shuffled)}, "
          f"높은 셀 {int(above.sum())}/{len(shuffled)}")
    print(f"순열 평균이 관측보다 {merged.gap.mean():+.1%} "
          f"(셀별 범위 {merged.gap.min():+.1%} ~ {merged.gap.max():+.1%})")
    print(f"순열 분포 자체의 폭(최대-최소) 셀 평균 {merged.spread.mean():.5f}, "
          f"관측 regret 대비 {merged.relative_spread.mean():.1%}")
    print("각 후보 집합은 자신의 signed error 다중집합을 그대로 유지하므로 "
          "(MAE는 합산 순서 때문에 상대 6e-16까지만 일치), 위 변동은 "
          "모두 오차가 어느 후보에 놓이는가에서 온다")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
