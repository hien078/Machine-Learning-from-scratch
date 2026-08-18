"""Run every quality gate, in the order CI runs them.

One command for the whole contract: lint, format, notebook format, types, unit
tests with the coverage floor, and project tests. The CI `test` job invokes this
script, so the local gate and the CI gate cannot drift apart.

All gates run even after one fails, so a single pass reports every problem.
"""

from __future__ import annotations

import subprocess
import sys

CHECKS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("lint", ("ruff", "check", "src", "tests", "scripts", "projects")),
    ("format", ("ruff", "format", "--check", "src", "tests", "scripts", "projects")),
    ("notebook format", (sys.executable, "scripts/normalize_notebooks.py", "--check")),
    ("types", ("mypy",)),
    (
        "unit tests",
        (
            sys.executable,
            "-m",
            "pytest",
            "--cov=ml_first_principles",
            "--cov-report=term",
            "--cov-fail-under=85",
        ),
    ),
    (
        "project tests",
        (sys.executable, "-m", "pytest", "projects", "-q", "-p", "no:cacheprovider"),
    ),
)


def main() -> None:
    failed: list[str] = []
    for name, command in CHECKS:
        print(f"\n{'=' * 60}\n== {name}: {' '.join(command)}\n{'=' * 60}", flush=True)
        if subprocess.run(command).returncode != 0:
            failed.append(name)

    print(f"\n{'=' * 60}")
    if failed:
        print(f"FAILED ({len(failed)}/{len(CHECKS)}): {', '.join(failed)}")
        raise SystemExit(1)
    print(f"All {len(CHECKS)} gates passed.")


if __name__ == "__main__":
    main()
