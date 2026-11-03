"""Train and apply the residual corrector for the requested feature condition and
network capacity. Training statistics come from the training role, model
selection uses validation, and test is used only for final prediction."""
from __future__ import annotations

import argparse
import itertools
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from corc import config
from corc.experiment_data import density_tag, load_role
from corc.corrector_inputs import (CORE, FINAL, FULL,
                                        CONTEXT_OVERLAP_PROBES,
                                        FINAL_REMOVED_KINDS,
                                        FINAL_REMOVED_PROBES, RELIABILITY_KINDS,
                                        ablation_columns, attach,
                                        final_conditions, load_probe_frame,
                                        matrix, without, without_kind)

HIDDEN = (64, 32)
EPOCHS = 100
PATIENCE = 5
BATCH = 2048


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


class Corrector(nn.Module):
    """Predict one standardised residual from the selected CoRC inputs."""

    def __init__(self, n_features: int, hidden: tuple[int, ...] = HIDDEN):
        super().__init__()
        layers, width = [], n_features
        for size in hidden:
            layers += [nn.Linear(width, size), nn.ReLU()]
            width = size
        head = nn.Linear(width, 1)
        nn.init.zeros_(head.weight)
        nn.init.zeros_(head.bias)
        self.stack = nn.Sequential(*layers, head)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.stack(x).squeeze(1)


def fit(train_x, train_r, valid_x, valid_r, scale: float, seed: int,
        learning_rate: float, weight_decay: float,
        hidden: tuple[int, ...] = HIDDEN) -> tuple[Corrector, int]:
    """Absolute error on the residual, early stopped on the validation split."""
    torch.manual_seed(seed)
    model = Corrector(train_x.shape[1], tuple(hidden))
    optimiser = torch.optim.Adam(model.parameters(), lr=learning_rate,
                                 weight_decay=weight_decay)
    generator = torch.Generator().manual_seed(seed)
    best, best_state, stale = np.inf, None, 0
    for epoch in range(1, EPOCHS + 1):
        model.train()
        order = torch.randperm(len(train_x), generator=generator)
        for start in range(0, len(order), BATCH):
            batch = order[start:start + BATCH]
            optimiser.zero_grad()
            loss = torch.abs(model(train_x[batch]) * scale - train_r[batch]).mean()
            loss.backward()
            optimiser.step()
        model.eval()
        with torch.no_grad():
            score = float(torch.abs(model(valid_x) * scale - valid_r).mean())
        if score + 1e-9 < best:
            best, stale = score, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            best_epoch = epoch
        else:
            stale += 1
            if stale >= PATIENCE:
                break
    model.load_state_dict(best_state)
    return model, best_epoch


def prepare(predictor: str, qos: str, seed: int, density: float, args,
            extra_roles: tuple[str, ...] = ()):
    """Prepare split-role inputs using training-only scaling statistics.

    Extra roles are prediction targets and never participate in fitting.
    """
    tag = density_tag(density)
    split = np.load(args.split_dir / f"{qos}_seed{seed}_{args.version}.npz")
    train_entries = split[f"{tag}_train_entries"]
    frames = {}
    for role in ("train", "validation", *extra_roles):
        probe = load_probe_frame(args.evidence_dir, qos, seed, tag, role,
                                 train_entries)
        base = load_role(args.root, args.evidence_dir, args.version, predictor,
                         qos, seed, density, role)
        merged = probe.merge(base[["user_id", "service_id", "pred", "y_true"]],
                             on=["user_id", "service_id"])
        frames[role] = attach(merged)
    train_x, mean, std = matrix(frames["train"], FULL)
    inputs = {"train": train_x}
    for role in ("validation", *extra_roles):
        inputs[role], _, _ = matrix(frames[role], FULL, mean, std)
    residual = {role: (frame["y_true"] - frame["pred"]).to_numpy(float)
                for role, frame in frames.items()}
    scale = float(np.std(residual["train"])) or 1.0
    return {"x": {role: torch.as_tensor(value, dtype=torch.float32)
                  for role, value in inputs.items()},
            "r": {role: torch.as_tensor(value, dtype=torch.float32)
                  for role, value in residual.items()},
            "scale": scale, "mean": mean, "std": std, "frames": frames}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode",
                        choices=["final", "ablation", "kinds", "final4",
                                 "overlap", "combine"],
                        default="final")
    parser.add_argument("--predictors", nargs="+", default=None)
    parser.add_argument("--densities", nargs="+", type=float, default=[0.10])
    parser.add_argument("--seeds", nargs="+", type=int, default=[42])
    parser.add_argument("--roles", nargs="+", default=["validation"],
                        choices=["validation", "test"])
    parser.add_argument("--hidden", nargs="*", type=int, default=list(HIDDEN))
    parser.add_argument("--shuffle-evidence", type=int, metavar="SEED")
    parser.add_argument("--evidence-dir", type=Path)
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument("--resume", action="store_true")
    return parser


def main() -> int:
    if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
        sys.stdout.reconfigure(encoding="utf-8")
    args = build_parser().parse_args()
    args.version = config.SPLIT_VERSION
    args.root = config.OUTPUT_DIR / "base_predictors"
    args.split_dir = config.OUTPUT_DIR / "splits" / args.version
    args.settings = config.ROOT / "configs" / "corrector_optimizers.csv"
    args.evidence_dir = (args.evidence_dir or config.OUTPUT_DIR
                         / "evidence_probes" / args.version / "full")
    args.out_dir = (args.out_dir or config.OUTPUT_DIR
                    / ("final_method" if args.mode == "final" else "structure_ad")
                    / args.version)
    if args.predictors is None:
        args.predictors = (["UIPCC", "PMF", "BiasMF", "LDCF", "NFMF"]
                           if args.mode == "final"
                           else ["UIPCC", "PMF", "BiasMF", "LDCF"])
    args.out_dir.mkdir(parents=True, exist_ok=True)

    settings = pd.read_csv(args.settings)
    if args.mode == "final4":
        conditions = final_conditions()
    elif args.mode == "final":
        conditions = {"final": FINAL}
    elif args.mode == "overlap":
        conditions = {
            "final": list(FINAL),
            "wo_context_overlap": without(
                FINAL_REMOVED_PROBES + CONTEXT_OVERLAP_PROBES,
                FINAL_REMOVED_KINDS),
        }
    elif args.mode == "combine":
        conditions = {
            "full": list(FULL),
            "wo_as": without(probes=("as_ucf", "as_scf")),
            "wo_as_support": without(probes=("as_ucf", "as_scf"),
                                     kinds=("support",)),
        }
    elif args.mode == "kinds":
        conditions = {"full": list(FULL)}
        conditions.update({f"wo_{kind}": without_kind(kind)
                           for kind in RELIABILITY_KINDS})
    elif args.mode == "ablation":
        conditions = {name: ablation_columns(name) for name in
                      ("full", "wo_user_side", "wo_service_side",
                       "wo_general_cf", "wo_geography", "wo_as")}
    target = {"kinds": "ablation_kinds"}.get(args.mode, args.mode)
    log(f"조건 {len(conditions)}개 x 예측기 {len(args.predictors)}개 x QoS 2개 "
        f"= {2 * len(conditions) * len(args.predictors)}회")

    rows, done = [], set()
    previous = args.out_dir / (target + ".csv")
    if args.resume and previous.exists():
        rows = pd.read_csv(previous).to_dict(orient="records")
        done = {(r["predictor"], round(r["density"], 4), r["seed"],
                 r["qos_type"], r["condition"]) for r in rows}
        log(f"재시작: 끝난 조합 {len(done)}개 건너뜀")

    for predictor, density, seed in itertools.product(args.predictors,
                                                      args.densities, args.seeds):
        cell = settings[(settings.predictor == predictor)
                        & (np.isclose(settings.density, density))]
        # The rate and decay were chosen for the eight-column input. Reused
        # unchanged so the structures differ in inputs and nothing else; that
        # they were not re-chosen for 45 columns is recorded as a limitation.
        cell = (cell if not cell.empty
                else settings[np.isclose(settings.density, density)]).iloc[0]
        rate, decay = float(cell.learning_rate), float(cell.weight_decay)
        if all((predictor, round(density, 4), seed, qos, name) in done
               for qos in config.QOS_TYPES for name in conditions):
            continue
        for qos in config.QOS_TYPES:
            extra = tuple(r for r in args.roles if r != "validation")
            data = prepare(predictor, qos, seed, density, args, extra)
            vanilla = float(torch.abs(data["r"]["validation"]).mean())
            for name, keep in conditions.items():
                if (predictor, round(density, 4), seed, qos, name) in done:
                    continue
                started = time.perf_counter()
                keep_mask = torch.as_tensor(
                    [column in set(keep) for column in FULL], dtype=torch.float32)
                inputs = data["x"]
                if args.shuffle_evidence is not None:
                    # Each role is permuted on its own, with the core column
                    # left alone: the base prediction is not evidence and
                    # moving it would change a different thing.
                    # One permutation for every evidence column, not one per
                    # column: the probes move together in reality -- a high
                    # similarity sum goes with a small deviation -- and
                    # shuffling them apart would test a different null. What is
                    # broken here is only that this evidence vector belongs to
                    # this entry.
                    evidence = [i for i, c in enumerate(FULL) if c not in CORE]
                    inputs = {}
                    for role, value in data["x"].items():
                        g = torch.Generator().manual_seed(
                            args.shuffle_evidence + seed
                            + sum(ord(c) for c in role))
                        order = torch.randperm(value.shape[0], generator=g)
                        moved = value.clone()
                        moved[:, evidence] = value[order][:, evidence]
                        inputs[role] = moved
                model, epoch = fit(inputs["train"] * keep_mask, data["r"]["train"],
                                   inputs["validation"] * keep_mask,
                                   data["r"]["validation"], data["scale"],
                                   seed, rate, decay, tuple(args.hidden))
                with torch.no_grad():
                    delta = (model(inputs["validation"] * keep_mask)
                             * data["scale"]).numpy()
                # The summary is always the validation error, whatever roles
                # `--roles` asked to be predicted: validation is the criterion
                # the conditions here are compared on, and test must not enter
                # a comparison that chooses between them. The role is written
                # into the table so a reader of the file cannot mistake the
                # basis -- an earlier run was saved as `test_final.csv` and its
                # validation error was read as a held-out one. The held-out
                # entry error is produced separately, by `report.make_tables`.
                corrected = data["frames"]["validation"]["pred"].to_numpy(float) + delta
                truth = data["frames"]["validation"]["y_true"].to_numpy(float)
                rows.append({
                    "predictor": predictor, "qos_type": qos, "condition": name,
                    "n_features": len(keep), "density": density,
                    "seed": seed, "epoch": epoch,
                    "mae_role": "validation",
                    "vanilla_mae": vanilla,
                    "mae": float(np.abs(corrected - truth).mean()),
                    "ratio": float(np.abs(corrected - truth).mean()) / vanilla,
                    "seconds": round(time.perf_counter() - started, 1)})
                r = rows[-1]
                log(f"  {predictor:7s} {qos} {name:16s} ({r['n_features']:2d}개) "
                    f"MAE {r['mae']:.4f}  비율 {r['ratio']:.4f}  "
                    f"epoch {epoch:3d}  {r['seconds']:.0f}초")
                # Corrected predictions are kept so the decision measures can
                # be computed afterwards without refitting; they take no part
                # in the choice made here.
                for role in args.roles:
                    with torch.no_grad():
                        shift = (model(inputs[role] * keep_mask)
                                 * data["scale"]).numpy()
                    out = data["frames"][role][["user_id", "service_id"]].copy()
                    out["pred"] = (data["frames"][role]["pred"].to_numpy(float)
                                   + shift)
                    out.to_csv(args.out_dir / f"{predictor}_{qos}_{name}"
                                              f"_{density_tag(density)}"
                                              f"_seed{seed}_{role}.csv", index=False)
        pd.DataFrame(rows).to_csv(args.out_dir / (target + ".csv"), index=False)

    table = pd.DataFrame(rows)
    objective = table.groupby("condition").ratio.mean().sort_values()
    pd.set_option("display.width", 220)
    if args.mode == "final":
        print()
        print("Fixed structure applied; MAE relative to the base predictor")
        print(table.pivot_table(index=["predictor", "qos_type"], columns="density",
                                values="ratio")
              .to_string(float_format=lambda x: f"{x:.4f}"))
        print()
        print(f"전체 평균 R_MAE = {table.ratio.mean():.4f}  (셀 {len(table)}개)")
        print("시드별 평균")
        print(table.groupby("seed").ratio.mean().to_string(
            float_format=lambda x: f"{x:.4f}"))
        return 0
    print()
    print("R_MAE = mean MAE ratio against the base predictor; lower is better")
    wide = table.pivot_table(index="condition", columns=["predictor", "qos_type"],
                             values="ratio")
    pd.set_option("display.width", 220)
    print(wide.reindex(objective.index).to_string(float_format=lambda x: f"{x:.4f}"))
    print()
    for name, value in objective.items():
        print(f"  {name:16s} R_MAE = {value:.4f}")
    print(f"\n선택: {objective.index[0]}  (validation MAE 비율 기준, regret 미사용)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
