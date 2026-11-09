"""Paired slice evaluation with explicit fail-closed contracts."""

from __future__ import annotations

import hashlib
import json
import math
from statistics import NormalDist
from typing import Any

Json = dict[str, Any]


def digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def wilson(successes: int, n: int, alpha: float) -> tuple[float, float]:
    if n <= 0 or not 0 <= successes <= n or not 0 < alpha < 1:
        raise ValueError("Invalid interval inputs")
    z = NormalDist().inv_cdf(1 - alpha / 2)
    p = successes / n
    denominator = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denominator
    radius = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denominator
    return max(0.0, center - radius), min(1.0, center + radius)


def _number(value: Any) -> bool:
    return type(value) in (int, float) and math.isfinite(value)


def validate(tasks: Any, reference: Any, candidate: Any, policy: Any) -> None:
    if not isinstance(policy, dict) or policy.get("schema") != 1:
        raise ValueError("Policy schema must be 1")
    labels = policy.get("labels")
    if (
        not isinstance(labels, list)
        or len(labels) < 2
        or any(not isinstance(x, str) or not x for x in labels)
        or len(set(labels)) != len(labels)
    ):
        raise ValueError("Policy labels must be distinct nonempty strings")
    if not isinstance(policy.get("task_version"), str) or not policy["task_version"]:
        raise ValueError("Missing task version")
    if not _number(policy.get("alpha")) or not 0.000001 <= policy["alpha"] <= 0.5:
        raise ValueError("Alpha must be finite and between 0.000001 and 0.5")
    if type(policy.get("min_samples")) is not int or policy["min_samples"] < 1:
        raise ValueError("Minimum samples must be a positive integer")
    if type(policy.get("human_review_required")) is not bool:
        raise ValueError("Human review flag must be boolean")
    slices = policy.get("slices")
    if (
        not isinstance(slices, dict)
        or not slices
        or len(slices) > 10000
        or any(
            not isinstance(k, str) or not k or not _number(v) or not 0 <= v <= 1
            for k, v in slices.items()
        )
    ):
        raise ValueError("Slices require finite tolerances in [0, 1]")
    if not isinstance(tasks, dict) or tasks.get("task_version") != policy["task_version"]:
        raise ValueError("Task version mismatch")
    rows = tasks.get("rows")
    if not isinstance(rows, list) or not rows:
        raise ValueError("Tasks must contain nonempty rows")
    ids = set()
    for row in rows:
        if (
            not isinstance(row, dict)
            or not isinstance(row.get("id"), str)
            or not row["id"]
            or row["id"] in ids
            or row.get("label") not in labels
            or not isinstance(row.get("slice"), str)
            or row["slice"] not in slices
            or not isinstance(row.get("text"), str)
            or not row["text"].strip()
        ):
            raise ValueError("Invalid, duplicate or unknown task row")
        ids.add(row["id"])
    for name, batch in [("reference", reference), ("candidate", candidate)]:
        if not isinstance(batch, dict) or batch.get("task_version") != policy["task_version"]:
            raise ValueError(f"{name} task version mismatch")
        if batch.get("tasks_hash") != digest(tasks):
            raise ValueError(f"{name} task content hash mismatch")
        model_hash = batch.get("model_hash")
        if (
            not isinstance(model_hash, str)
            or len(model_hash) != 64
            or any(c not in "0123456789abcdef" for c in model_hash)
        ):
            raise ValueError(f"{name} requires a SHA-256 model identity")
        predictions = batch.get("predictions")
        if not isinstance(predictions, dict) or set(predictions) != ids:
            raise ValueError(f"{name} prediction IDs must exactly match tasks")
        if any(v not in labels for v in predictions.values()):
            raise ValueError(f"{name} contains invalid labels")


def evaluate(tasks: Any, reference: Any, candidate: Any, policy: Any) -> Json:
    try:
        validate(tasks, reference, candidate, policy)
        hashes = {
            name: digest(value)
            for name, value in [
                ("tasks", tasks),
                ("reference", reference),
                ("candidate", candidate),
                ("policy", policy),
            ]
        }
    except (ValueError, TypeError, OverflowError) as exc:
        return {"schema": 1, "decision": "FAIL", "contract_errors": [str(exc)], "slices": {}}
    reports: Json = {}
    for name, tolerance in policy["slices"].items():
        rows = [r for r in tasks["rows"] if r["slice"] == name]
        n = len(rows)
        gains = losses = ref_correct = cand_correct = 0
        for row in rows:
            r = reference["predictions"][row["id"]] == row["label"]
            c = candidate["predictions"][row["id"]] == row["label"]
            ref_correct += r
            cand_correct += c
            gains += c and not r
            losses += r and not c
        interval = None
        state = "REVIEW_REQUIRED"
        reason = "insufficient_samples"
        if n:
            alpha = policy["alpha"] / (2 * len(policy["slices"]))
            gain = wilson(gains, n, alpha)
            loss = wilson(losses, n, alpha)
            interval = [gain[0] - loss[1], gain[1] - loss[0]]
            if n >= policy["min_samples"]:
                state = (
                    "FAIL"
                    if interval[1] < -tolerance
                    else "PASS"
                    if interval[0] >= -tolerance
                    else "REVIEW_REQUIRED"
                )
                reason = {
                    "FAIL": "supported_regression",
                    "PASS": "within_tolerance",
                    "REVIEW_REQUIRED": "uncertain_difference",
                }[state]
        reports[name] = {
            "n": n,
            "gains": gains,
            "losses": losses,
            "reference_correct": ref_correct,
            "candidate_correct": cand_correct,
            "accuracy_delta": (cand_correct - ref_correct) / n if n else None,
            "difference_interval": interval,
            "tolerance": tolerance,
            "decision": state,
            "reason": reason,
        }
    states = [x["decision"] for x in reports.values()]
    decision = (
        "FAIL"
        if "FAIL" in states
        else "REVIEW_REQUIRED"
        if "REVIEW_REQUIRED" in states or policy["human_review_required"]
        else "PASS"
    )
    return {
        "schema": 1,
        "decision": decision,
        "contract_errors": [],
        "hashes": hashes,
        "human_review_required": policy["human_review_required"],
        "slices": reports,
    }
