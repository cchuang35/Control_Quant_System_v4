"""Run the current main v4 validation script.

The current main candidate is v4.2-candidate-C.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    script = root / "scripts" / "run_v42_candidate_c_validation.py"
    raise SystemExit(subprocess.call([sys.executable, str(script)], cwd=root))


if __name__ == "__main__":
    main()
