"""Validate that notebooks execute cleanly without touching the source files.

Runs every notebook (or the paths given on the command line) top-to-bottom on a
fresh kernel against an in-memory copy, per NOTEBOOK_STANDARDS.md §11: source
notebooks are never overwritten with generated outputs.
"""

from __future__ import annotations

import glob
import os
import sys
import time
from pathlib import Path

import nbformat
from nbclient import NotebookClient
from nbclient.exceptions import CellExecutionError


def find_notebooks() -> list[str]:
    patterns = [
        "topics/**/*.ipynb",
        "_template_first_principles.ipynb",
    ]
    notebooks = []
    for pattern in patterns:
        for path in glob.glob(pattern, recursive=True):
            if os.path.isfile(path) and not path.startswith("."):
                notebooks.append(path)
    return sorted(set(notebooks))


def run_notebook(path: str) -> tuple[bool, str]:
    print(f"Executing: {path}...", end="", flush=True)
    start_time = time.time()
    try:
        with open(path, encoding="utf-8") as f:
            nb = nbformat.read(f, as_version=4)

        # Environment with PYTHONPATH pointing to src
        src_path = str(Path("src").resolve())
        env = os.environ.copy()
        env["PYTHONPATH"] = src_path + ":" + env.get("PYTHONPATH", "")

        # Execute the in-memory copy only; errors must fail the run.
        client = NotebookClient(
            nb,
            timeout=180,
            allow_errors=False,
            kernel_name="python3",
        )
        client.execute(env=env)

        elapsed = time.time() - start_time
        print(f" OK ({elapsed:.2f}s)")
        return True, ""
    except CellExecutionError as e:
        elapsed = time.time() - start_time
        print(f" FAILED ({elapsed:.2f}s)")
        return False, f"Cell execution error: {e.ename}: {e.evalue}"
    except Exception as e:
        elapsed = time.time() - start_time
        print(f" ERROR ({elapsed:.2f}s)")
        return False, f"Error: {e}"


def main() -> None:
    notebooks = sys.argv[1:] or find_notebooks()
    print(f"Found {len(notebooks)} notebooks to validate.")

    failures = []
    for path in notebooks:
        ok, err = run_notebook(path)
        if not ok:
            failures.append((path, err))

    print("\n" + "=" * 60)
    print(f"VALIDATION SUMMARY: {len(notebooks) - len(failures)}/{len(notebooks)} Succeeded")
    print("=" * 60)

    if failures:
        print("\nFailures:")
        for path, err in failures:
            print(f"  - {path}: {err}")
        sys.exit(1)
    print("\nAll notebooks executed cleanly; source files left untouched.")


if __name__ == "__main__":
    main()
