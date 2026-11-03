"""Select learning rate and weight decay for the final 17-column corrector using
validation MAE only. Settings are selected per predictor and density."""
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
from corc.experiment_data import density_tag
from corc.corrector_inputs import FINAL, FULL
from .train_corrector import fit, prepare

RATES = (1e-4, 3e-4, 1e-3, 3e-3, 1e-2)
DECAYS = (0.0, 1e-4, 1e-3)


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main() -> int:
    if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictors", nargs="+",
                        default=["UIPCC", "PMF", "BiasMF", "LDCF", "NFMF"])
    parser.add_argument("--densities", nargs="+", type=float,
                        default=sorted(config.DENSITIES))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--rates", nargs="+", type=float, default=list(RATES))
    parser.add_argument("--decays", nargs="+", type=float, default=list(DECAYS))
    parser.add_argument("--root", type=Path,
                        default=config.OUTPUT_DIR / "base_predictors")
    parser.add_argument("--evidence-dir", type=Path,
                        default=config.OUTPUT_DIR / "evidence_probes" / config.SPLIT_VERSION / "full")
    parser.add_argument("--split-dir", type=Path,
                        default=config.OUTPUT_DIR / "splits" / config.SPLIT_VERSION)
    parser.add_argument("--version", default=config.SPLIT_VERSION)
    parser.add_argument("--out-dir", type=Path,
                        default=config.OUTPUT_DIR / "final_method" / config.SPLIT_VERSION)
    # The sweep is long enough to outlive a session, and every setting is
    # written as it finishes, so a restart picks up where it stopped.
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    keep = torch.as_tensor([column in set(FINAL) for column in FULL],
                           dtype=torch.float32)
    total = (len(args.predictors) * len(args.densities)
             * len(args.rates) * len(args.decays) * len(config.QOS_TYPES))
    log(f"Final-17, entry-only, seed {args.seed}  ->  {total}회")

    rows, done = [], set()
    previous = args.out_dir / "final_lr_search.csv"
    if args.resume and previous.exists():
        rows = pd.read_csv(previous).to_dict(orient="records")
        done = {(r["predictor"], round(r["density"], 4)) for r in rows
                if sum(1 for q in rows if q["predictor"] == r["predictor"]
                       and round(q["density"], 4) == round(r["density"], 4))
                >= len(args.rates) * len(args.decays)}
        log(f"재시작: 끝난 셀 {len(done)}개 건너뜀")

    for predictor, density in itertools.product(args.predictors, args.densities):
        if (predictor, round(density, 4)) in done:
            continue
        data = {qos: prepare(predictor, qos, args.seed, density, args)
                for qos in config.QOS_TYPES}
        vanilla = {qos: float(torch.abs(data[qos]["r"]["validation"]).mean())
                   for qos in config.QOS_TYPES}
        for rate, decay in itertools.product(args.rates, args.decays):
            started = time.perf_counter()
            ratios = {}
            for qos in config.QOS_TYPES:
                cell = data[qos]
                model, epoch = fit(cell["x"]["train"] * keep, cell["r"]["train"],
                                   cell["x"]["validation"] * keep,
                                   cell["r"]["validation"], cell["scale"],
                                   args.seed, rate, decay)
                with torch.no_grad():
                    residual = (model(cell["x"]["validation"] * keep)
                                * cell["scale"])
                ratios[qos] = (float(torch.abs(residual - cell["r"]["validation"]).mean())
                               / vanilla[qos])
            rows.append({"predictor": predictor, "density": density,
                         "seed": args.seed, "learning_rate": rate,
                         "weight_decay": decay,
                         **{f"ratio_{qos}": ratios[qos] for qos in config.QOS_TYPES},
                         "R_MAE": float(np.mean(list(ratios.values()))),
                         "seconds": round(time.perf_counter() - started, 1)})
        best = min((r for r in rows if r["predictor"] == predictor
                    and np.isclose(r["density"], density)), key=lambda r: r["R_MAE"])
        log(f"  {predictor:7s} d={density:.0%}: lr={best['learning_rate']:.0e} "
            f"wd={best['weight_decay']:.0e}  R_MAE {best['R_MAE']:.4f}")
        pd.DataFrame(rows).to_csv(args.out_dir / "final_lr_search.csv", index=False)

    table = pd.DataFrame(rows)
    chosen = (table.loc[table.groupby(["predictor", "density"]).R_MAE.idxmin()]
              [["predictor", "density", "learning_rate", "weight_decay", "R_MAE"]]
              .reset_index(drop=True))
    chosen[["predictor", "density", "learning_rate", "weight_decay"]].to_csv(
        args.out_dir / "corrector_optimizers.csv", index=False)

    pd.set_option("display.width", 200)
    print()
    print("선택된 설정 (예측기 x 밀도, RT·TP 공통)")
    print(chosen.pivot_table(index="predictor", columns="density",
                             values="learning_rate").to_string())
    print()
    print("weight decay")
    print(chosen.pivot_table(index="predictor", columns="density",
                             values="weight_decay").to_string())
    print()
    print("격자 경계 점검 — 선택된 값의 분포")
    print(chosen.groupby(["learning_rate", "weight_decay"]).size().to_string())
    print()
    print("학습률별 전체 평균 R_MAE")
    print(table.groupby("learning_rate").R_MAE.mean().to_string(
        float_format=lambda x: f"{x:.4f}"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
