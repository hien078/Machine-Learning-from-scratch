from __future__ import annotations

import argparse
import copy
import json
import uuid
import warnings
from collections import Counter
from pathlib import Path
from typing import Any

import nbformat

CANONICAL_KERNELSPEC = {
    "display_name": "Python 3",
    "language": "python",
    "name": "python3",
}
EXCLUDED_DIRECTORIES = {".git", ".ipynb_checkpoints", ".venv"}


def _stable_cell_id(relative_path: Path, cell_index: int) -> str:
    key = f"ml-workspace/{relative_path.as_posix()}#cell-{cell_index}"
    return uuid.uuid5(uuid.NAMESPACE_URL, key).hex[:12]


def _protected_payload(notebook: dict[str, Any]) -> dict[str, Any]:
    """Return fields that normalization is not allowed to change."""
    protected = copy.deepcopy(notebook)
    protected.pop("nbformat_minor", None)
    protected.get("metadata", {}).pop("kernelspec", None)
    for cell in protected.get("cells", []):
        cell.pop("id", None)
        if cell.get("cell_type") == "code":
            cell.pop("execution_count", None)
            cell.pop("outputs", None)
    return protected


def _notebook_paths(root: Path) -> list[Path]:
    return sorted(
        path
        for path in root.rglob("*.ipynb")
        if not any(part in EXCLUDED_DIRECTORIES for part in path.parts)
    )


def normalize_notebook(path: Path, root: Path, write: bool, clear_outputs: bool) -> Counter[str]:
    """Normalize one notebook while preserving all academic content.

    Args:
        path: Notebook path.
        root: Repository root used to derive deterministic cell IDs.
        write: Whether to persist the normalized bytes.
        clear_outputs: Whether stored outputs/execution counts are cleared.
            By default they are kept: committed outputs are legitimate as long
            as they come from ``execute_all_notebooks.py --write`` (fresh
            kernel), per NOTEBOOK_STANDARDS.md §8.

    Returns:
        Counts of defects found and changes proposed for this notebook.
    """
    raw = path.read_bytes()
    notebook = json.loads(raw.decode("utf-8-sig"))
    before = _protected_payload(notebook)
    relative_path = path.relative_to(root)
    counts: Counter[str] = Counter()

    if raw.startswith(b"\xef\xbb\xbf"):
        counts["bom"] += 1
    if b"\r\n" in raw:
        counts["crlf"] += 1

    cells = notebook.get("cells", [])
    for index, cell in enumerate(cells):
        if not cell.get("id"):
            cell["id"] = _stable_cell_id(relative_path, index)
            counts["cell_ids_added"] += 1
        if cell.get("cell_type") == "code":
            if "outputs" not in cell:
                cell["outputs"] = []
                counts["missing_outputs_added"] += 1
            if "execution_count" not in cell:
                cell["execution_count"] = None
                counts["missing_execution_counts_added"] += 1
            if clear_outputs:
                outputs = cell.get("outputs", [])
                if outputs:
                    counts["outputs_cleared"] += len(outputs)
                    cell["outputs"] = []
                if cell.get("execution_count") is not None:
                    counts["execution_counts_cleared"] += 1
                    cell["execution_count"] = None

    if notebook.get("nbformat_minor") != 5:
        notebook["nbformat_minor"] = 5
        counts["nbformat_minor_updated"] += 1

    metadata = notebook.setdefault("metadata", {})
    if metadata.get("kernelspec") != CANONICAL_KERNELSPEC:
        metadata["kernelspec"] = CANONICAL_KERNELSPEC.copy()
        counts["kernelspec_normalized"] += 1

    if _protected_payload(notebook) != before:
        raise RuntimeError(f"Protected notebook content changed: {relative_path}")

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        nbformat.validate(notebook)

    object_changed = any(
        counts[key]
        for key in (
            "cell_ids_added",
            "outputs_cleared",
            "execution_counts_cleared",
            "missing_outputs_added",
            "missing_execution_counts_added",
            "nbformat_minor_updated",
            "kernelspec_normalized",
        )
    )
    if object_changed:
        normalized = (json.dumps(notebook, ensure_ascii=False, indent=1) + "\n").encode()
    else:
        normalized = raw.removeprefix(b"\xef\xbb\xbf").replace(b"\r\n", b"\n")

    if normalized != raw:
        counts["files_changed"] += 1
        if write:
            path.write_bytes(normalized)

    return counts


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Normalize Jupyter notebooks without changing cell sources."
    )
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--write",
        action="store_true",
        help="Persist changes. Without this flag the command is a dry run.",
    )
    parser.add_argument(
        "--clear-outputs",
        action="store_true",
        help=(
            "Also clear stored outputs and execution counts. By default outputs "
            "are kept; execute_all_notebooks.py --write is their canonical producer."
        ),
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Exit with status 1 if any notebook would change (CI gate).",
    )
    parser.add_argument(
        "paths",
        nargs="*",
        type=Path,
        help="Optional notebook files or directories. Defaults to the repository root.",
    )
    args = parser.parse_args()
    root = args.root.resolve()
    totals: Counter[str] = Counter()
    changed_paths: list[Path] = []

    if args.paths:
        paths = []
        for requested in args.paths:
            candidate = requested.resolve()
            if candidate.is_dir():
                paths.extend(_notebook_paths(candidate))
            else:
                paths.append(candidate)
        paths = sorted(set(paths))
    else:
        paths = _notebook_paths(root)
    for path in paths:
        counts = normalize_notebook(path, root, args.write, args.clear_outputs)
        totals.update(counts)
        if counts["files_changed"]:
            changed_paths.append(path.relative_to(root))

    mode = "WRITE" if args.write else "DRY RUN"
    print(f"mode={mode}")
    print(f"notebooks_checked={len(paths)}")
    for key in sorted(totals):
        print(f"{key}={totals[key]}")
    print("changed_paths:")
    for path in changed_paths:
        print(path)
    if args.check and changed_paths:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
