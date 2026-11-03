"""Build MEC candidate sets by joining held-out QoS predictions, collaborative
evidence, server locations, and server status. This stage requires the
non-redistributable MEC inputs described in the data README."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from corc import config
from corc.experiment_data import KEY, PREDICTORS, STRUCTURE3, density_tag


N_CANDIDATES = 5
EARTH_KM = 6371.0
# The baselines whose columns every candidate row must carry. Anything listed
# here has to exist for every cell, or the build fails rather than writing a
# partial instance. NFMF is merged in afterwards, which is why it is absent.
STANDARD = ["UIPCC", "PMF", "BiasMF", "LDCF"]


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def haversine(lat1: np.ndarray, lon1: np.ndarray,
              lat2: np.ndarray, lon2: np.ndarray) -> np.ndarray:
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dp, dl = p2 - p1, np.radians(lon2 - lon1)
    a = np.sin(dp / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * EARTH_KM * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def require_mec() -> None:
    """Validate the required non-redistributable MEC input files."""
    missing = [p for p in (config.MEC_SAMPLES, config.MEC_LOCATIONS,
                           config.MEC_STATUS) if not p.exists()]
    if not missing:
        return
    raise SystemExit(
        "MEC 측 데이터를 찾지 못했습니다. 다음 파일이 필요합니다:\n  "
        + "\n  ".join(str(p) for p in missing)
        + f"\n\n현재 찾는 위치: {config.MEC_DIR}\n"
        "다른 곳에 있다면 환경변수로 알려주세요:\n"
        "  CORC_MEC_DIR=/path/to/mec python -m pipeline.build_selection_dataset ...\n"
        "구성과 생성 절차는 README 의 'MEC dataset' 절을 참고하세요.")


def load_requests() -> pd.DataFrame:
    """One row per request: who, when, where, which service, and its QoS weights."""
    cols = ["sample_id", "matrix_user_id", "predicted_lat", "predicted_lon",
            "server_time_slot", "service_id", "service_name",
            "weight_rt", "weight_tp", "weight_ld", "weight_fr"]
    frame = pd.read_csv(config.MEC_SAMPLES, usecols=cols).drop_duplicates("sample_id")
    frame["user_id"] = frame["matrix_user_id"].str.extract(r"(\d+)").astype(int)
    return frame.drop(columns=["matrix_user_id"]).reset_index(drop=True)


def load_servers() -> pd.DataFrame:
    """Edge servers with their synthetic location and the WS-DREAM column they carry."""
    frame = pd.read_csv(config.MEC_LOCATIONS)
    # `matrix_service_id` is one-based; the matrices this project uses are not.
    frame["service_id"] = frame["matrix_service_id"].str.extract(r"(\d+)").astype(int) - 1
    return frame[["server_id", "latitude", "longitude", "service_id"]]


def cold_services(split_dir: Path, version: str, qos: str, seeds: list[int],
                  shape: tuple[int, int], density: float) -> np.ndarray:
    """Return services unobserved in training for any seed.

    Pooling across seeds keeps the eligible candidate set comparable.
    """
    cold = set()
    for seed in seeds:
        entries = np.load(split_dir / f"{qos}_seed{seed}_{version}.npz")[
            f"{density_tag(density)}_train_entries"]
        observed = np.zeros(shape, dtype=bool)
        observed[entries[:, 0], entries[:, 1]] = True
        cold |= set(np.flatnonzero(~observed.any(axis=0)).tolist())
    return np.array(sorted(cold), dtype=np.int64)


def held_out_pairs(split_dir: Path, version: str, qos: str, seed: int,
                   shape: tuple[int, int], role: str,
                   density: float | None = None) -> pd.DataFrame:
    """Return measured pairs held out from the requested split role."""
    # The held-out roles are one array each; the training entries are a nested
    # series, one per density, so that role needs the density to name its key.
    key = (f"{density_tag(density)}_train_entries" if role == "train"
           else f"{role}_entries")
    entries = np.load(split_dir / f"{qos}_seed{seed}_{version}.npz")[key]
    if entries.ndim == 1:
        users, services = np.unravel_index(entries.astype(np.int64), shape)
    else:
        users, services = entries[:, 0], entries[:, 1]
    return pd.DataFrame({"user_id": users, "service_id": services})


def pick_candidates(requests: pd.DataFrame, servers: pd.DataFrame,
                    eligible: pd.DataFrame, n: int) -> pd.DataFrame:
    """The `n` nearest eligible servers to each request's predicted location."""
    pool = eligible.merge(servers, on="service_id", how="inner")
    picked = []
    for user_id, block in requests.groupby("user_id", sort=False):
        options = pool[pool.user_id == user_id]
        if len(options) < n:
            continue
        lat = options["latitude"].to_numpy()
        lon = options["longitude"].to_numpy()
        for row in block.itertuples(index=False):
            distance = haversine(np.full(len(lat), row.predicted_lat),
                                 np.full(len(lon), row.predicted_lon), lat, lon)
            near = np.argpartition(distance, n - 1)[:n]
            near = near[np.argsort(distance[near])]
            chosen = options.iloc[near].copy()
            chosen["sample_id"] = row.sample_id
            chosen["server_distance_km"] = distance[near]
            chosen["candidate_rank"] = np.arange(1, n + 1)
            picked.append(chosen)
    return pd.concat(picked, ignore_index=True)


def attach_predictions(frame: pd.DataFrame, root: Path, version: str, qos: str,
                       seed: int, density: float, role: str = "test") -> pd.DataFrame:
    """Every standard predictor's output for the candidate entries, plus the truth."""
    out = frame
    for name in STANDARD:
        family, prefix = PREDICTORS[name]
        path = (root / family / version / qos / f"seed{seed}" / density_tag(density)
                / f"{prefix}_{role}_predictions.csv")
        block = pd.read_csv(path, usecols=KEY + ["y_true", "pred"])
        # Both QoS types are attached, so the predictor name alone would collide.
        block = block.rename(columns={"pred": f"pred_{name}_{qos}",
                                      "y_true": f"true_{qos}"})
        keep = KEY + [f"pred_{name}_{qos}"] + ([f"true_{qos}"] if f"true_{qos}" not in out else [])
        out = out.merge(block[keep], on=KEY, how="left")
    return out


def main() -> int:
    if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence-dir", type=Path,
                        default=config.OUTPUT_DIR / "evidence_probes" / config.SPLIT_VERSION / "full")
    parser.add_argument("--root", type=Path, default=config.OUTPUT_DIR / "base_predictors")
    parser.add_argument("--split-dir", type=Path,
                        default=config.OUTPUT_DIR / "splits" / config.SPLIT_VERSION)
    parser.add_argument("--version", default=config.SPLIT_VERSION)
    parser.add_argument("--out-dir", type=Path,
                        default=config.OUTPUT_DIR / "selection_dataset" / config.SPLIT_VERSION)
    parser.add_argument("--seeds", nargs="+", type=int, default=config.SPLIT_SEEDS)
    parser.add_argument("--density", type=float, default=min(config.DENSITIES))
    parser.add_argument("--n-candidates", type=int, default=N_CANDIDATES)
    parser.add_argument("--role", default="test",
                        choices=["test", "validation", "train"],
                        help="which split role supplies the candidate pairs. "
                             "`train` builds the instances a decision-focused "
                             "corrector is fitted on, from that density's "
                             "training entries and their out-of-fold predictions")
    args = parser.parse_args()
    require_mec()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    shape = (339, 5825)
    requests = load_requests()
    servers = load_servers()
    status = pd.read_csv(config.MEC_STATUS).rename(columns={"load": "LD", "fail_rate": "FR"})
    log(f"요청 {len(requests):,}  엣지 서버 {len(servers):,}  상태 행 {len(status):,}")

    manifest = {"n_candidates": args.n_candidates, "density": args.density,
                "source": str(config.MEC_SAMPLES), "seeds": args.seeds,
                "candidate_rule": "nearest eligible servers to the predicted location; "
                                  f"eligible = (user, service) held out in the {args.role} split",
                "role": args.role}

    # Taken at the lowest density so every density excludes the same services,
    # which keeps one instance set comparable across all four.
    cold = cold_services(args.split_dir, args.version, "rt", args.seeds, shape,
                         min(config.DENSITIES))
    manifest["cold_services_excluded"] = {
        "count": int(len(cold)), "of": shape[1],
        "rule": "no training observation at the lowest density, pooled over seeds"}
    log(f"적격에서 제외할 cold-start 서비스 {len(cold)}개 / {shape[1]} "
        f"({len(cold) / shape[1]:.2%})")

    for seed in args.seeds:
        # RT and TP share one partition in this split version, so either gives
        # the same eligible set; RT is read for concreteness.
        eligible = held_out_pairs(args.split_dir, args.version, "rt", seed, shape,
                                  args.role, args.density)
        eligible = eligible[~eligible.service_id.isin(cold)]
        candidates = pick_candidates(requests, servers, eligible, args.n_candidates)
        frame = candidates.merge(requests, on="sample_id", how="inner",
                                 suffixes=("", "_req"))
        frame = frame.merge(status[["server_id", "time_slot", "LD", "FR"]],
                            left_on=["server_id", "server_time_slot"],
                            right_on=["server_id", "time_slot"], how="left").drop(columns="time_slot")
        for qos in config.QOS_TYPES:
            frame = attach_predictions(frame, args.root, args.version, qos, seed,
                                       args.density, args.role)
            # Neighbour dispersion is measured in the QoS's own units, so the RT
            # and TP probes give different values for the same column name --
            # `ucf_neighbor_qos_variance` averages about 2 for RT and 3,655 for
            # TP. Reading one probe for both would feed each risk model and
            # corrector features from the wrong matrix.
            # The probe has to be the one for this role. Reading the test
            # probe for a validation or training instance leaves every
            # structural column empty, and a corrector handed empty evidence
            # still fits -- it just stops being the method, with nothing in
            # the output to say so.
            probe = pd.read_csv(
                args.evidence_dir / qos / f"seed{seed}" / density_tag(args.density)
                / f"evidence_{args.role}_predictions.csv",
                usecols=KEY + STRUCTURE3[qos])
            frame = frame.merge(probe.rename(
                columns={c: f"{c}__{qos}" for c in STRUCTURE3[qos]}), on=KEY, how="left")

        complete = frame.groupby("sample_id")["true_rt"].transform(
            lambda s: s.notna().all() and len(s) == args.n_candidates)
        frame = frame[complete].reset_index(drop=True)
        # Predictions differ by density, so the density has to be in the name;
        # otherwise a sweep silently overwrites the previous density's file.
        suffix = "" if args.role == "test" else f"_{args.role}"
        path = (args.out_dir / f"candidates_k{args.n_candidates}"
                f"_{density_tag(args.density)}{suffix}_seed{seed}.csv")
        frame.to_csv(path, index=False)
        log(f"  seed {seed}: 요청 {frame.sample_id.nunique():,}  행 {len(frame):,}  -> {path.name}")
        manifest[f"k{args.n_candidates}_{density_tag(args.density)}_seed{seed}"] = {"requests": int(frame.sample_id.nunique()),
                                   "rows": int(len(frame))}

    name = (f"manifest_k{args.n_candidates}_{density_tag(args.density)}"
            f"{'' if args.role == 'test' else '_' + args.role}.json")
    (args.out_dir / name).write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    log(f"outputs -> {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
