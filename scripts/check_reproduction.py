"""Evaluate the released checkpoints and compare with the paper's ablation table (Table 3).

For each released configuration (Baseline, Baseline + BC-MSRR, Baseline + LCJS,
HiBlastNet) and seed (100, 101, 102), this script runs scripts/evaluate.py unless
a result already exists, then reports mean +/- sample SD over the three seeds next
to the published values.

    python scripts/check_reproduction.py            # evaluate what is missing, then compare
    python scripts/check_reproduction.py --no-run   # compare existing results only

The paper's Table 3 comes from the original inference run; a later re-inference of
the same checkpoints changed a few values in the last reported digit, so differences
of about 0.01-0.02 are expected.
"""
import argparse
import json
from pathlib import Path
import statistics
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
SEEDS = (100, 101, 102)

# Table 3: mean and sample SD over seeds 100/101/102 (test set, unfiltered predictions).
PAPER_TABLE3 = {
    "gcnet": {"label": "Baseline", "dice": (80.86, 0.10), "small_survival": (55.18, 0.92),
              "matched_recall": (83.32, 0.17), "fp_per_leaf": (10.80, 0.11), "boundary_f1": (71.16, 0.53)},
    "gcnet_bc": {"label": "Baseline + BC-MSRR", "dice": (81.68, 0.09), "small_survival": (60.30, 0.33),
                 "matched_recall": (85.39, 0.06), "fp_per_leaf": (13.01, 0.15), "boundary_f1": (72.46, 0.40)},
    "gcnet_lcjs": {"label": "Baseline + LCJS", "dice": (81.08, 0.04), "small_survival": (57.93, 2.14),
                   "matched_recall": (84.33, 0.36), "fp_per_leaf": (12.26, 0.68), "boundary_f1": (71.85, 0.73)},
    "hiblastnet": {"label": "HiBlastNet", "dice": (81.93, 0.17), "small_survival": (65.43, 2.65),
                   "matched_recall": (87.22, 0.18), "fp_per_leaf": (15.27, 0.60), "boundary_f1": (72.66, 0.14)},
}
METRICS = ("dice", "small_survival", "matched_recall", "fp_per_leaf", "boundary_f1")


def read_metrics(method, seed):
    runs = sorted((ROOT / "outputs" / method / f"seed{seed}").glob("test_*_full_leaf_*/metrics.json"))
    if not runs:
        return None
    m = json.loads(runs[-1].read_text(encoding="utf-8"))["metrics"]
    combined = m["component"]["combined"]
    return {
        "dice": 100 * m["lesion_mean_dice"],
        "small_survival": 100 * combined["small_component_survival_recall"],
        "matched_recall": 100 * combined["matched_component_recall"],
        "fp_per_leaf": combined["false_positive_components_per_leaf"],
        "boundary_f1": 100 * m["boundary"]["combined_lesion_boundary_f1"],
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--no-run", action="store_true", help="Only compare results that already exist")
    parser.add_argument("--tolerance", type=float, default=0.05,
                        help="Largest acceptable |reproduced mean - paper mean| (default 0.05)")
    args = parser.parse_args(argv)

    for method in PAPER_TABLE3:
        for seed in SEEDS:
            if read_metrics(method, seed) is None and not args.no_run:
                print(f"Evaluating {method} seed {seed} ...", flush=True)
                code = subprocess.run(
                    [sys.executable, str(ROOT / "scripts" / "evaluate.py"), "--method", method, "--seed", str(seed)],
                    cwd=ROOT,
                ).returncode
                if code:
                    print(f"  evaluation failed with exit code {code}")

    all_ok = True
    header = f"{'Configuration':22s} {'Metric':15s} {'Reproduced':>17s} {'Paper':>17s} {'Diff':>7s}"
    print("\n" + header + "\n" + "-" * len(header))
    for method, paper in PAPER_TABLE3.items():
        results = [read_metrics(method, seed) for seed in SEEDS]
        if any(r is None for r in results):
            missing = [str(s) for s, r in zip(SEEDS, results) if r is None]
            print(f"{paper['label']:22s} missing results for seeds {', '.join(missing)}")
            all_ok = False
            continue
        for metric in METRICS:
            values = [r[metric] for r in results]
            mean, sd = statistics.mean(values), statistics.stdev(values)
            diff = mean - paper[metric][0]
            ok = abs(diff) <= args.tolerance
            all_ok &= ok
            print(f"{paper['label']:22s} {metric:15s} {mean:8.2f} +/- {sd:5.2f} "
                  f"{paper[metric][0]:8.2f} +/- {paper[metric][1]:5.2f} {diff:+7.2f}{'' if ok else '  <-- check'}")
    print("\nAll values within tolerance." if all_ok else "\nSome values differ from the paper or are missing.")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
