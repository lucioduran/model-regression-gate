"""Offline consistency verification without training or model inference."""

import hashlib
import json
from pathlib import Path

from model_regression_gate.gate import evaluate

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    manifest = json.loads((ROOT / "evidence/manifest.json").read_text())
    for name, expected in manifest.items():
        path = (ROOT / name).resolve()
        if not path.is_relative_to(ROOT) or not path.is_file():
            raise SystemExit(f"Invalid manifest path: {name}")
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise SystemExit(f"Hash mismatch: {name}")
    policy = json.loads((ROOT / "data/policy.json").read_text())
    count = 0
    for partition in ["development", "evaluation"]:
        base = ROOT / "evidence" / partition
        tasks = json.loads((base / "tasks.json").read_text())
        reference = json.loads((base / "predictions/baseline.json").read_text())
        for path in sorted((base / "predictions").glob("*.json")):
            candidate = json.loads(path.read_text())
            actual = evaluate(tasks, reference, candidate, policy)
            expected = json.loads((base / "reports" / path.name).read_text())
            if actual != expected:
                raise SystemExit(f"Report mismatch: {partition}/{path.name}")
            count += 1
    for path in sorted((ROOT / "evidence/cases").glob("*.json")):
        if path.stem.endswith("-report"):
            continue
        case = json.loads(path.read_text())
        actual = evaluate(*(case[key] for key in ["tasks", "reference", "candidate", "policy"]))
        expected = json.loads(path.with_name(f"{path.stem}-report.json").read_text())
        if actual != expected:
            raise SystemExit(f"Case mismatch: {path.name}")
        count += 1
    print(f"Verified {len(manifest)} file hashes and replayed {count} decisions; no inference run.")


if __name__ == "__main__":
    main()
