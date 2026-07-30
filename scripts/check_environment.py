from __future__ import annotations

import importlib
import sys
from importlib import metadata

REQUIRED_IMPORTS = (
    ("numpy", "numpy"),
    ("pandas", "pandas"),
    ("scipy", "scipy"),
    ("matplotlib", "matplotlib"),
    ("sklearn", "scikit-learn"),
    ("jupyterlab", "jupyterlab"),
    ("ipykernel", "ipykernel"),
    ("ipywidgets", "ipywidgets"),
    ("nbformat", "nbformat"),
    ("nbclient", "nbclient"),
)
OPTIONAL_IMPORTS = (("torch", "torch"),)


def _check_imports(imports: tuple[tuple[str, str], ...]) -> list[str]:
    failures: list[str] = []
    for module_name, distribution_name in imports:
        try:
            importlib.import_module(module_name)
            version = metadata.version(distribution_name)
        except (ImportError, metadata.PackageNotFoundError, OSError) as exc:
            failures.append(f"{module_name}: {type(exc).__name__}: {exc}")
        else:
            print(f"{module_name}=={version}")
    return failures


def main() -> None:
    print(f"python={sys.version.split()[0]}")
    print(f"executable={sys.executable}")
    print("required:")
    required_failures = _check_imports(REQUIRED_IMPORTS)
    print("optional:")
    optional_failures = _check_imports(OPTIONAL_IMPORTS)

    for failure in optional_failures:
        print(f"OPTIONAL MISSING: {failure}")
    for failure in required_failures:
        print(f"REQUIRED FAILURE: {failure}")

    if required_failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
