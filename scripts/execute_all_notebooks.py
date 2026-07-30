import glob
import os
import sys
import time
from pathlib import Path

import nbformat
from nbclient import NotebookClient
from nbclient.exceptions import CellExecutionError

def find_notebooks():
    patterns = [
        "foundations/**/*.ipynb",
        "topics/**/*.ipynb",
        "_template_first_principles.ipynb"
    ]
    notebooks = []
    for pattern in patterns:
        for path in glob.glob(pattern, recursive=True):
            if os.path.isfile(path) and not path.startswith("."):
                notebooks.append(path)
    return sorted(list(set(notebooks)))

def run_notebook(path: str, python_path: str) -> tuple[bool, str]:
    print(f"Executing: {path}...", end="", flush=True)
    start_time = time.time()
    try:
        with open(path, "r", encoding="utf-8") as f:
            nb = nbformat.read(f, as_version=4)

        # Environment with PYTHONPATH pointing to src
        src_path = str(Path("src").resolve())
        env = os.environ.copy()
        env["PYTHONPATH"] = src_path + ":" + env.get("PYTHONPATH", "")

        client = NotebookClient(
            nb,
            timeout=180,
            allow_errors=True,
            kernel_name="python3"
        )

        client.execute(env=env)

        with open(path, "w", encoding="utf-8") as f:
            nbformat.write(nb, f)

        elapsed = time.time() - start_time
        print(f" DONE ({elapsed:.2f}s)")
        return True, ""
    except CellExecutionError as e:
        elapsed = time.time() - start_time
        print(f" FAILED ({elapsed:.2f}s)")
        return False, f"Cell execution error: {e.ename}: {e.evalue}"
    except Exception as e:
        elapsed = time.time() - start_time
        print(f" ERROR ({elapsed:.2f}s)")
        return False, f"Error: {str(e)}"

def main():
    python_path = str(Path(".venv/bin/python").resolve())
    notebooks = find_notebooks()
    print(f"Found {len(notebooks)} notebooks to execute.")

    successes = []
    failures = []

    for path in notebooks:
        ok, err = run_notebook(path, python_path)
        if ok:
            successes.append(path)
        else:
            failures.append((path, err))

    print("\n" + "=" * 60)
    print(f"EXECUTION SUMMARY: {len(successes)}/{len(notebooks)} Succeeded")
    print("=" * 60)

    if failures:
        print("\nFailures:")
        for path, err in failures:
            print(f"  - {path}: {err}")
        sys.exit(1)
    else:
        print("\nAll notebooks executed and saved successfully!")

if __name__ == "__main__":
    main()
