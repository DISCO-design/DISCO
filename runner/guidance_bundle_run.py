"""Create a reproducibility bundle for one DISCO guidance run."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from disco.model.cogen_inference.guidance import (
    RunBundleConfig,
    validate_run_bundle,
    write_run_bundle,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--out-json", type=Path, required=True)
    parser.add_argument("--dump-dir", type=Path)
    parser.add_argument("--summary-csv", type=Path)
    parser.add_argument("--planned-run-manifest", type=Path)
    parser.add_argument("--analysis-manifest", type=Path)
    parser.add_argument("--diagnostics-dir", type=Path)
    parser.add_argument("--repo-dir", type=Path)
    parser.add_argument("--issues-csv", type=Path)
    parser.add_argument("--system", default="DISCO")
    parser.add_argument("--mode", default="")
    parser.add_argument("--variant", default="")
    parser.add_argument("--replicate", default="")
    parser.add_argument("--seed", default="")
    parser.add_argument("--sha256-max-bytes", type=int, default=10_000_000)
    parser.add_argument("--no-git", action="store_true")
    parser.add_argument("--require-summary", action="store_true")
    parser.add_argument("--require-diagnostics", action="store_true")
    parser.add_argument("--require-plan", action="store_true")
    return parser.parse_args()


def write_issues_csv(path: Path, issues) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["severity", "check", "path", "message"]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows([issue.as_dict() for issue in issues])


def main() -> None:
    args = parse_args()
    config = RunBundleConfig(
        run_dir=args.run_dir.resolve(),
        out_json=args.out_json.resolve(),
        system=args.system,
        mode=args.mode,
        variant=args.variant,
        replicate=args.replicate,
        seed=args.seed,
        dump_dir=args.dump_dir.resolve() if args.dump_dir is not None else None,
        summary_csv=args.summary_csv.resolve()
        if args.summary_csv is not None
        else None,
        planned_run_manifest=args.planned_run_manifest.resolve()
        if args.planned_run_manifest is not None
        else None,
        analysis_manifest=args.analysis_manifest.resolve()
        if args.analysis_manifest is not None
        else None,
        diagnostics_dir=args.diagnostics_dir.resolve()
        if args.diagnostics_dir is not None
        else None,
        repo_dir=args.repo_dir.resolve() if args.repo_dir is not None else Path.cwd(),
        sha256_max_bytes=args.sha256_max_bytes,
        include_git=not args.no_git,
    )
    bundle = write_run_bundle(config)
    issues = validate_run_bundle(
        bundle,
        require_summary=args.require_summary,
        require_diagnostics=args.require_diagnostics,
        require_plan=args.require_plan,
    )
    if args.issues_csv is not None:
        write_issues_csv(args.issues_csv.resolve(), issues)

    num_errors = sum(1 for issue in issues if issue.severity == "error")
    print(
        "Run bundle "
        + ("pass" if num_errors == 0 else "fail")
        + f": {num_errors} error(s), {len(issues) - num_errors} warning(s)"
    )
    print(f"Wrote bundle to {config.out_json}")
    print(
        json.dumps(
            {
                "summary_rows": bundle["summary_csv"]["num_rows"],
                "diagnostics_metadata_records": bundle["diagnostics"][
                    "num_metadata_records"
                ],
            },
            sort_keys=True,
        )
    )
    if num_errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
