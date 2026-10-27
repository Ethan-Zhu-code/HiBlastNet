"""Evaluate one method from configs/EXPERIMENTS.csv with its evaluation configuration.

Released checkpoints (HiBlastNet, the refinement control, the Baseline and
Baseline + LCJS; seeds 100, 101, 102) are found under weights/ by --seed.
For any other checkpoint, pass --model-path, and --deploy false if it comes
from train.py.
"""
import argparse
import csv
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def released_weight(row, seed):
    pattern = row.get("released_weights", "")
    if not pattern:
        return None
    return ROOT / pattern.replace("{100,101,102}", str(seed))


def build_command(args, row):
    command = [sys.executable, "-B", str(ROOT / "test.py"),
               "--config", str(ROOT / row["evaluation_config"]),
               "--full-leaf-split", args.split]
    if args.model_path is not None:
        # User paths follow the caller's working directory, even though the
        # evaluator itself runs from the repository root.
        command += ["--model-path", str(Path(args.model_path).resolve())]
        output_dir = ROOT / "outputs" / row["id"] / "custom"
    else:
        weight = released_weight(row, args.seed)
        if weight is None:
            raise SystemExit(
                f"No released weights for '{row['id']}'. Train it first and pass "
                "--model-path outputs/training/.../checkpoint_best.pth --deploy false"
            )
        command += ["--model-path", str(weight)]
        output_dir = ROOT / "outputs" / row["id"] / f"seed{args.seed}"
    command += ["--output-dir", str(output_dir)]
    if args.deploy is not None:
        command += ["--deploy", args.deploy]
    return command


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--method", "--run", dest="method", default="hiblastnet",
                        help="Method from --list (default: hiblastnet)")
    parser.add_argument("--list", action="store_true", help="List available methods")
    parser.add_argument("--split", choices=["val", "test"], default="test")
    parser.add_argument("--seed", type=int, choices=[100, 101, 102], default=100,
                        help="Which released checkpoint to evaluate (default: 100)")
    parser.add_argument("--model-path", help="Evaluate this checkpoint instead of a released one")
    parser.add_argument("--deploy", choices=["true", "false"], default=None,
                        help="false for checkpoints from train.py; released GCNet weights need no flag")
    parser.add_argument("--dry-run", action="store_true", help="Print the command without evaluating")
    args = parser.parse_args(argv)
    with (ROOT / "configs/EXPERIMENTS.csv").open(encoding="utf-8", newline="") as handle:
        entries = list(csv.DictReader(handle))
    if args.list:
        for row in entries:
            weights = "released weights" if row["released_weights"] else "train first"
            print(f"{row['id']:24s} {row['role']:11s} {weights:17s} {row['description']}")
        return 0
    index = {row["id"]: row for row in entries}
    if args.method not in index:
        parser.error(f"Unknown method: {args.method}. Use --list to see available methods.")
    command = build_command(args, index[args.method])
    if args.dry_run:
        print(json.dumps(dict(method=args.method, argv=command, cwd=str(ROOT)), indent=2))
        return 0
    return subprocess.run(command, cwd=ROOT).returncode


if __name__ == "__main__":
    sys.exit(main())
