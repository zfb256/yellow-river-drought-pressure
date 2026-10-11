"""Run the pipeline from raw data to the result tables and figures; usage: python scripts/run_all.py [--from STAGE]."""
import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
STAGES = [
    ("aggregate", ["aggregate_products.py cmfd", "aggregate_products.py gleam", "aggregate_basins.py", "aggregate_grun.py"]),
    ("data_checks", ["drought_indices.py", "check_homogeneity.py", "pet_check.py"]),
    ("indices", ["drought_indices_v2.py"]),
    ("validation", ["validation.py", "nested_basins.py"]),
    ("mechanisms", ["mechanism_analysis.py", "mechanism_robustness.py", "mechanism_undammed.py", "counterfactual.py",
                    "counterfactual.py --snow-shift=-1", "counterfactual.py --snow-shift=1", "demand_checks.py", "pet_drivers.py"]),
    ("figures", ["study_figures.py", "make_mechanism_figures.py"]),
    ("checks", ["check_mechanisms.py"]),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--from", dest="start", default="aggregate")
    ap.add_argument("--only")
    args = ap.parse_args()
    names = [s for s, _ in STAGES]
    todo = [args.only] if args.only else names[names.index(args.start):]
    (ROOT / "logs").mkdir(exist_ok=True)
    for stage, steps in STAGES:
        if stage not in todo:
            continue
        for step in steps:
            cmd = [sys.executable, f"scripts/{step.split()[0]}", *step.split()[1:]]
            log = ROOT / "logs" / f"{stage}__{step.replace('.py', '').replace(' ', '_')}.log"
            print(f"[{stage}] {step}", flush=True)
            with open(log, "w", encoding="utf-8") as fh:
                if subprocess.run(cmd, cwd=ROOT, stdout=fh, stderr=subprocess.STDOUT).returncode:
                    sys.exit(f"failed: {step}, see {log}")


if __name__ == "__main__":
    main()
