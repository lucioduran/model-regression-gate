"""File-based promotion decisions. Exit 0 pass, 1 fail, 2 review."""

import argparse
import json
from pathlib import Path

from .gate import evaluate


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ["tasks", "reference", "candidate", "policy"]:
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    try:
        report = evaluate(
            *(
                json.loads(getattr(args, name).read_text())
                for name in ["tasks", "reference", "candidate", "policy"]
            )
        )
    except (OSError, ValueError) as exc:
        report = {"decision": "FAIL", "contract_errors": [str(exc)]}
    print(json.dumps(report, indent=2, allow_nan=False))
    raise SystemExit({"PASS": 0, "FAIL": 1, "REVIEW_REQUIRED": 2}[report["decision"]])


if __name__ == "__main__":
    main()
