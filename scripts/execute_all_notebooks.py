"""Validate that notebooks execute cleanly, optionally persisting fresh outputs.

Runs every notebook (or the paths given via positionals / --only) top-to-bottom
on a fresh kernel against an in-memory copy. By default source notebooks are
never overwritten (NOTEBOOK_STANDARDS.md §11). With --write, the freshly
executed notebook is written back to disk: this script is the only legitimate
producer of committed outputs (NOTEBOOK_STANDARDS.md §8), using the inline
matplotlib backend so figures are captured headlessly and deterministically.
"""

from __future__ import annotations

import argparse
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


def run_notebook(path: str, write: bool) -> tuple[bool, str]:
    print(f"Executing: {path}...", end="", flush=True)
    start_time = time.time()
    try:
        with open(path, encoding="utf-8") as f:
            nb = nbformat.read(f, as_version=4)

        # Environment with PYTHONPATH pointing to src
        src_path = str(Path("src").resolve())
        env = os.environ.copy()
        env["PYTHONPATH"] = src_path + ":" + env.get("PYTHONPATH", "")
        if write:
            env["MPLBACKEND"] = "module://matplotlib_inline.backend_inline"

        # Execute the in-memory copy; errors must fail the run.
        client = NotebookClient(
            nb,
            timeout=180,
            allow_errors=False,
            kernel_name="python3",
        )
        client.execute(env=env)

        if write:
            with open(path, "w", encoding="utf-8") as f:
                nbformat.write(nb, f)

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


def select_notebooks(selections: list[str]) -> list[str]:
    """Expand paths, directories, or globs into a sorted notebook list."""
    notebooks: set[str] = set()
    for item in selections:
        if os.path.isdir(item):
            notebooks.update(glob.glob(os.path.join(item, "**", "*.ipynb"), recursive=True))
        else:
            notebooks.update(glob.glob(item, recursive=True) or [item])
    return sorted(notebooks)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "paths",
        nargs="*",
        help="Optional notebook files. Defaults to topics/ plus the template.",
    )
    parser.add_argument(
        "--only",
        action="append",
        default=[],
        metavar="PATH_OR_GLOB",
        help="Restrict to notebooks under a path, directory, or glob (repeatable).",
    )
    parser.add_argument(
        "--write",
        action="store_true",
        help="Persist the freshly executed notebooks (outputs included) back to disk.",
    )
    args = parser.parse_args()

    selections = list(args.paths) + list(args.only)
    notebooks = select_notebooks(selections) if selections else find_notebooks()
    print(f"Found {len(notebooks)} notebooks to validate.")

    failures = []
    for path in notebooks:
        ok, err = run_notebook(path, args.write)
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
    if args.write:
        print("\nAll notebooks executed cleanly; fresh outputs written back to disk.")
    else:
        print("\nAll notebooks executed cleanly; source files left untouched.")


if __name__ == "__main__":
    main()
