"""Command-line entry point for the experimental workflow."""

import argparse
from pathlib import Path
import sys


def make_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("preflight", help="Check generators, damage and decoder")
    experiment = commands.add_parser("experiment", help="Run the decoding experiment")
    experiment.add_argument("--config", type=Path)
    experiment.add_argument("--trials", type=int)
    experiment.add_argument("--seed-base", type=int)
    experiment.add_argument("--out", type=Path, default=Path("results/final"))
    experiment.add_argument("--overwrite", action="store_true")
    analyze = commands.add_parser("analyze", help="Fit polynomial models and generate figures")
    analyze.add_argument("--input", type=Path, required=True)
    analyze.add_argument("--out", type=Path, required=True)
    validate = commands.add_parser("validate", help="Run the T01–T19 validation suite")
    validate.add_argument("--input", type=Path, required=True)
    validate.add_argument("--smoke-input", type=Path)
    validate.add_argument("--manuscript", type=Path, default=Path("reports/보고서_3쪽_원고.md"))
    validate.add_argument("--pdf", type=Path, default=Path("reports/보고서_3쪽.pdf"))
    validate.add_argument("--student-id")
    validate.add_argument("--student-name")
    validate.add_argument("--qa-out", type=Path)
    report = commands.add_parser("export-report", help="Render A4 PDF and Markdown manuscript")
    report.add_argument("--input", type=Path, required=True)
    report.add_argument("--out", type=Path, required=True)
    report.add_argument("--manuscript", type=Path)
    report.add_argument("--student-id")
    report.add_argument("--student-name")
    report.add_argument("--preview", action="store_true", help="Use blank identity fields")
    return parser


def main(argv=None):
    parser = make_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "preflight":
            from .checks import preflight

            results = preflight()
            for key, (passed, detail) in results.items():
                print(f"{key} {'PASS' if passed else 'FAIL'} {detail}")
            return int(not all(ok for ok, _ in results.values()))

        if args.command == "experiment":
            from .config import load_config
            from .experiment import run_experiment

            config = load_config(args.config, trials=args.trials, seed_base=args.seed_base)
            files = run_experiment({**config, "out": args.out, "overwrite": args.overwrite})
            print(f"{4 * 18 * config['trials']:,} attempts · 72 conditions")
            print(*(str(path) for path in files), sep="\n")

        if args.command == "analyze":
            from .analysis import analyze_run

            analyze_run(args.input, args.out)
            print(args.out / "analysis_summary.csv")

        if args.command == "validate":
            from .validation import validate_run

            results = validate_run(
                args.input, smoke_input=args.smoke_input, manuscript=args.manuscript, pdf=args.pdf,
                student_id=args.student_id, student_name=args.student_name, qa_out=args.qa_out,
            )
            return int(any(entry["status"] == "FAIL" for entry in results.values()))

        if args.command == "export-report":
            from .report import export_report

            manuscript, pages = export_report(
                args.input, args.out, args.manuscript,
                args.student_id, args.student_name, args.preview,
            )
            print(f"{args.out} · A4 {pages}쪽")
            print(manuscript)
    except (OSError, ValueError, RuntimeError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1
    return 0
