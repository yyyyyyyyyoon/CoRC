"""Create and persist the fixed nested WS-DREAM splits shared by all experiment
stages."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from corc import config
from corc.dataset import (QosDataset, fixed_research_split,
                               load_dataset)


def sha256_file(path: Path) -> str:
    h = hashcorc.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def density_tag(density: float) -> str:
    return f"density{int(round(density * 100)):02d}"


def metadata_summary(dataset: QosDataset) -> dict[str, Any]:
    users = dataset.users
    services = dataset.services
    return {
        "users": {
            "rows": int(len(users)),
            "country_missing": int(users["country"].isna().sum()),
            "as_no_missing": int(users["as_no"].isna().sum()),
            "lat_missing": int(users["lat"].isna().sum()),
            "lon_missing": int(users["lon"].isna().sum()),
            "unique_country": int(users["country"].nunique(dropna=True)),
            "unique_as_no": int(users["as_no"].nunique(dropna=True)),
        },
        "services": {
            "rows": int(len(services)),
            "country_missing": int(services["country"].isna().sum()),
            "as_no_missing": int(services["as_no"].isna().sum()),
            "lat_missing": int(services["lat"].isna().sum()),
            "lon_missing": int(services["lon"].isna().sum()),
            "unique_country": int(services["country"].nunique(dropna=True)),
            "unique_as_no": int(services["as_no"].nunique(dropna=True)),
        },
    }


def matrix_summary(name: str, matrix: np.ndarray) -> dict[str, Any]:
    observed = int(np.sum(~np.isnan(matrix)))
    total = int(matrix.size)
    return {
        "qos_type": name,
        "shape": [int(matrix.shape[0]), int(matrix.shape[1])],
        "total_entries": total,
        "observed_entries": observed,
        "observed_fraction": observed / total,
        "missing_entries": total - observed,
    }


def save_bundle_npz(bundle, out_path: Path) -> None:
    arrays: dict[str, np.ndarray] = {
        "observed_entries": bundle.observed_entries,
        "model_pool_entries": bundle.model_pool_entries,
        "validation_entries": bundle.validation_entries,
        "test_entries": bundle.test_entries,
    }
    for density, entries in bundle.density_train_entries.items():
        tag = density_tag(density)
        arrays[f"{tag}_train_entries"] = entries
        for fold_id, fold_entries in enumerate(bundle.density_meta_folds[density]):
            arrays[f"{tag}_meta_fold{fold_id}_entries"] = fold_entries
    np.savez_compressed(out_path, **arrays)


def build_splits(out_dir: Path, seeds: list[int]) -> tuple[pd.DataFrame, dict[str, Any]]:
    dataset = load_dataset()
    out_dir.mkdir(parents=True, exist_ok=True)

    # RT and TP are measured in the same invocation, so their observation masks
    # nearly coincide: intersecting them costs 2.45% of RT entries and 0.18% of
    # TP entries. Paying that gives both QoS types an identical partition, which
    # a joint RT/TP utility needs downstream and which lets one user-service pair
    # be followed across both.
    shared_mask = None
    if config.SHARED_OBSERVATION_MASK:
        shared_mask = np.ones(dataset.rt.shape, dtype=bool)
        for qos_type in config.QOS_TYPES:
            shared_mask &= ~np.isnan(dataset.matrix(qos_type))

    rows: list[dict[str, Any]] = []
    for qos_type in config.QOS_TYPES:
        matrix = dataset.matrix(qos_type)
        n_observed = int(shared_mask.sum() if shared_mask is not None else np.sum(~np.isnan(matrix)))
        for seed in seeds:
            bundle = fixed_research_split(
                matrix,
                qos_type=qos_type,
                seed=seed,
                densities=config.DENSITIES,
                validation_fraction=config.VALIDATION_FRACTION,
                test_fraction=config.TEST_FRACTION,
                meta_folds=config.META_FOLDS,
                observed_mask=shared_mask,
            )
            npz_name = f"{qos_type}_seed{seed}_{config.SPLIT_VERSION}.npz"
            save_bundle_npz(bundle, out_dir / npz_name)

            rows.append({
                "qos_type": qos_type,
                "seed": seed,
                "density": "",
                "split_file": npz_name,
                "observed_entries": n_observed,
                "model_pool_entries": len(bundle.model_pool_entries),
                "validation_entries": len(bundle.validation_entries),
                "test_entries": len(bundle.test_entries),
                "train_entries": "",
                "meta_folds": "",
                "train_fraction_of_observed": "",
            })
            for density in config.DENSITIES:
                train_entries = bundle.density_train_entries[float(density)]
                fold_sizes = [len(f) for f in bundle.density_meta_folds[float(density)]]
                rows.append({
                    "qos_type": qos_type,
                    "seed": seed,
                    "density": density,
                    "split_file": npz_name,
                    "observed_entries": n_observed,
                    "model_pool_entries": len(bundle.model_pool_entries),
                    "validation_entries": len(bundle.validation_entries),
                    "test_entries": len(bundle.test_entries),
                    "train_entries": len(train_entries),
                    "meta_folds": "|".join(str(x) for x in fold_sizes),
                    "train_fraction_of_observed": len(train_entries) / n_observed,
                })

    summary = pd.DataFrame(rows)
    manifest = {
        "split_version": config.SPLIT_VERSION,
        "created_by": Path(__file__).name,
        "policy": {
            "qos_types": config.QOS_TYPES,
            "densities": config.DENSITIES,
            "seeds": seeds,
            "validation_fraction": config.VALIDATION_FRACTION,
            "test_fraction": config.TEST_FRACTION,
            "meta_folds": config.META_FOLDS,
            "density_denominator": "all observed entries for the same qos_type",
            "density_train_policy": "nested prefix samples from model_pool after validation/test holdout",
            "validation_test_policy": "fixed before density sampling and common across densities",
        },
        "dataset": {
            "paths": {
                "rt": str(config.RT_PATH),
                "tp": str(config.TP_PATH),
                "userlist": str(config.USERLIST_PATH),
                "wslist": str(config.WSLIST_PATH),
            },
            "sha256": {
                "rt": sha256_file(config.RT_PATH),
                "tp": sha256_file(config.TP_PATH),
                "userlist": sha256_file(config.USERLIST_PATH),
                "wslist": sha256_file(config.WSLIST_PATH),
            },
            "matrices": [
                matrix_summary("rt", dataset.rt),
                matrix_summary("tp", dataset.tp),
            ],
            "metadata": metadata_summary(dataset),
        },
    }
    return summary, manifest


def write_readme(out_dir: Path, summary: pd.DataFrame, manifest: dict[str, Any]) -> None:
    compact = summary[summary["density"] != ""].copy()
    compact["density"] = compact["density"].map(lambda x: f"{float(x):.0%}")
    lines = [
        "# CERP-QoS fixed splits",
        "",
        f"Split version: `{manifest['split_version']}`",
        "",
        "Contract:",
        "",
        "- Validation and final test entries are selected before density masks.",
        "- Validation and final test entries are shared by all densities for the same QoS type and seed.",
        "- Density train entries are nested prefix samples: 5% is contained in 10%, 10% in 15%, and 15% in 20%.",
        "- Meta folds partition the density-specific train entries for later OOF profiler/corrector training.",
        "- Actual QoS values remain in the source matrix; split files store only `(user_id, service_id)` entry arrays.",
        "",
        "Summary:",
        "",
        compact.to_markdown(index=False),
        "",
    ]
    (out_dir / "README.md").write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=config.OUTPUT_DIR / "splits" / config.SPLIT_VERSION,
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=config.SPLIT_SEEDS)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary, manifest = build_splits(args.out_dir, args.seeds)
    summary.to_csv(args.out_dir / "split_summary.csv", index=False)
    (args.out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    write_readme(args.out_dir, summary, manifest)
    print(f"Wrote fixed splits to {args.out_dir}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
