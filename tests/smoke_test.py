"""Check dependencies, command entry points, dataset shape, and reproducibility of
the versioned table and figure data. Temporary outputs are isolated from the
experiment directory."""
from __future__ import annotations

import subprocess
import sys
import tempfile
import traceback
from pathlib import Path

import pandas as pd

from corc import config

ENTRY_POINTS = [
    "pipeline.prepare_fixed_splits", "pipeline.base_predictors.run_uipcc",
    "pipeline.base_predictors.run_mf", "pipeline.base_predictors.run_ldcf",
    "pipeline.base_predictors.run_nfmf", "pipeline.build_train_oof",
    "pipeline.build_evidence", "pipeline.build_selection_dataset",
    "pipeline.select_final_lr", "pipeline.train_corrector",
    "report.make_tables", "report.make_figure_data", "report.collect_results",
    "report.bootstrap_reduction", "report.analyze_rq3_controls",
    "report.ablation_regret",
]


def check(label: str):
    """Run one check, report pass or fail, and keep going either way."""
    def wrap(fn):
        try:
            detail = fn()
            print(f"  OK    {label}" + (f" — {detail}" if detail else ""))
            return True
        except Exception as exc:                       # noqa: BLE001
            print(f"  FAIL  {label}")
            for line in traceback.format_exception_only(type(exc), exc):
                print(f"          {line.rstrip()}")
            return False
    return wrap


def main() -> int:
    if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
        sys.stdout.reconfigure(encoding="utf-8")
    results: list[bool] = []

    print("환경")

    @check("패키지")
    def _packages():
        import numpy, scipy, sklearn, tabulate, torch       # noqa: F401
        return f"torch {torch.__version__}, pandas {pd.__version__}"
    results.append(_packages)

    @check("진입점 16개")
    def _entry_points():
        broken = [m for m in ENTRY_POINTS
                  if subprocess.run([sys.executable, "-m", m, "--help"],
                                    capture_output=True).returncode != 0]
        if broken:
            raise RuntimeError(f"--help 실패: {broken}")
        return f"{len(ENTRY_POINTS)}개 모두 기동"
    results.append(_entry_points)

    print("\n데이터")

    @check("WS-DREAM")
    def _wsdream():
        from corc.dataset import load_dataset
        data = load_dataset()
        rt = data.matrix("rt")
        if rt.shape != (339, 5825):
            raise RuntimeError(f"행렬 모양 {rt.shape}, 기대 (339, 5825)")
        observed = int((rt > 0).sum())
        if not 1_800_000 < observed < 1_900_000:
            raise RuntimeError(f"관측 {observed:,}개가 예상 범위 밖")
        return f"339 x 5,825, 관측 {observed:,}"
    results.append(_wsdream)

    @check("Shanghai Telecom")
    def _mec():
        missing = [p.name for p in (config.MEC_SAMPLES, config.MEC_LOCATIONS,
                                    config.MEC_STATUS) if not p.exists()]
        if missing:
            # Not a failure: the data cannot be redistributed, and every stage
            # except the candidate-set build runs without it.
            return f"없음 (선택 데이터셋 생성 단계만 사용) — {', '.join(missing)}"
        stations = len(pd.read_csv(config.MEC_LOCATIONS))
        if stations != 2977:
            raise RuntimeError(f"기지국 {stations}개, 기대 2,977")
        return f"기지국 {stations:,}"
    results.append(_mec)

    print("\n결과 재현  (results/paper/ 만 읽음)")

    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp)

        def rebuild(module: str, label: str) -> bool:
            @check(label)
            def _run():
                run = subprocess.run(
                    [sys.executable, "-m", module, "--out-dir", str(out)],
                    capture_output=True, text=True, errors="replace")
                if run.returncode != 0:
                    tail = (run.stderr or "").strip().splitlines()[-1:]
                    raise RuntimeError(" ".join(tail))
                checked = 0
                for built in sorted(out.glob("*.csv")):
                    stored = config.RESULTS_DIR / "generated" / built.name
                    if not stored.exists():
                        continue
                    a = pd.read_csv(built).astype(str)
                    b = pd.read_csv(stored).astype(str)
                    if not a.equals(b):
                        raise RuntimeError(f"{built.name} 이 체크인된 것과 다름")
                    checked += 1
                if not checked:
                    raise RuntimeError("대조할 파일이 없음")
                return f"{checked}개 파일 일치"
            return _run

        results.append(rebuild("report.make_tables", "표 1-6"))
        results.append(rebuild("report.make_figure_data", "그림 3-4"))

    failed = results.count(False)
    print()
    print(f"{len(results) - failed}/{len(results)} 통과"
          + ("" if not failed else f", {failed}건 실패"))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
