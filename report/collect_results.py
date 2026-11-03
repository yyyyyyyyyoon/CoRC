"""Copy compact paper-result summaries from ignored experiment outputs into the
versioned `results/paper` directory."""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

from corc import config


def plan(outputs: Path) -> dict[str, list[tuple[Path, str]]]:
    """Where each file goes. A key starting with `_` is written outside
    `results/` -- the corrector's settings are an input to the pipeline, not a
    record of it, so they live in the top-level `configs/`."""
    final = outputs / "final_method" / config.SPLIT_VERSION
    structure = outputs / "structure_ad" / config.SPLIT_VERSION
    flat = lambda *paths: [(p, p.name) for p in paths]
    return {
        # Read back by `pipeline.train_corrector`: an input, not a result,
        # so it is written to the top-level `configs/` rather than here.
        "_configs": flat(final / "corrector_optimizers.csv"),
        # The sweep those were chosen from, kept so what was searched is visible.
        "tuning": flat(final / "final_lr_search.csv"),
        "table_1": flat(final / "k10" / "rq1_test.csv",
                        final / "k10" / "rq1_test_user_deltas.csv",
                        final / "k10" / "rq1_test_reduction_ci.csv"),
        "table_2_3_4": flat(final / "mae_test_entries.csv"),  # regret 캐시도 여기
        # One sentence of the paper: "With K = 5 and K = 20 candidates, CoRC
        # reduced the average regret by ...".
        "k_sensitivity": [(final / "k5" / "rq1_test.csv", "k5_rq1_test.csv"),
                          (final / "k5" / "rq1_test_user_deltas.csv", "k5_user_deltas.csv"),
                          (final / "k20" / "rq1_test.csv", "k20_rq1_test.csv"),
                          (final / "k20" / "rq1_test_user_deltas.csv", "k20_user_deltas.csv")],
        "table_6": flat(structure / "structure_regret_3seeds.csv",
                        structure / "ablation.csv",
                        structure / "ablation_kinds.csv",
                        structure / "combine.csv"),
        # `rq3_shuffle` is the paper's error *permutation*; the word "shuffled"
        # is used there for Table 4's evidence placebo, which is a different
        # experiment, so the file is renamed on the way across.
        "figure_3": [(final / "rq3" / "rq3_controls.csv", "rq3_controls.csv"),
                     (final / "rq3" / "rq3_shuffle.csv", "rq3_permutation.csv")],
        "figure_4": (
            [(final / "neighbour_k" / f"n{n}" / "summary.csv", f"neighbour_n{n}.csv")
             for n in (5, 10, 50, 100, 200)]
            + [(final / "rq2" / "rq2_overlap.csv", "neighbour_n20.csv"),
               (final / "test" / "validation_summary.csv", "capacity_64x32.csv")]
            + [(final / "capacity" / d / "summary.csv", f"capacity_{d}.csv")
               for d in ("linear", "16", "32x16", "128x64")]),
    }


def main() -> int:
    if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outputs", type=Path, default=config.OUTPUT_DIR)
    parser.add_argument("--results", type=Path, default=config.RESULTS_DIR)
    args = parser.parse_args()

    copied, missing = 0, []
    for folder, items in plan(args.outputs).items():
        target = (config.ROOT / folder.lstrip("_") if folder.startswith("_")
                  else args.results / folder)
        target.mkdir(parents=True, exist_ok=True)
        for source, name in items:
            if not source.exists():
                missing.append(source)
                continue
            shutil.copy2(source, target / name)
            copied += 1

    size = sum(p.stat().st_size for p in args.results.rglob("*") if p.is_file())
    print(f"{copied}개 복사, {args.results} 합계 {size / 1024:.1f} KB")
    for p in missing:
        print(f"  없음: {p}")
    # Not an error: a partial pipeline run legitimately has no capacity sweep.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
