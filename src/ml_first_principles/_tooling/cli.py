"""``mlfp`` — one entry point for the repository's quality tooling.

Subcommands:
    mlfp check     run every quality gate in CI order
    mlfp nb-exec   execute notebooks on a fresh kernel (``--write`` persists outputs)
    mlfp nb-fmt    check or normalize notebook format
"""

from __future__ import annotations

import sys

from ml_first_principles._tooling import check, execute_notebooks, normalize_notebooks

_COMMANDS = {
    "check": check.main,
    "nb-exec": execute_notebooks.main,
    "nb-fmt": normalize_notebooks.main,
}


def main() -> None:
    if len(sys.argv) < 2 or sys.argv[1] in {"-h", "--help"} or sys.argv[1] not in _COMMANDS:
        names = ", ".join(_COMMANDS)
        print(f"usage: mlfp {{{names}}} [options]\n\n{__doc__}")
        raise SystemExit(0 if len(sys.argv) >= 2 and sys.argv[1] in {"-h", "--help"} else 2)
    command = sys.argv.pop(1)
    sys.argv[0] = f"mlfp {command}"
    _COMMANDS[command]()


if __name__ == "__main__":
    main()
