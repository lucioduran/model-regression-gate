"""Run local training and inference. Invoke through the shared benchmark lock."""

from __future__ import annotations

import copy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import random
import resource
import subprocess
import time
from typing import Any

from model_regression_gate.gate import Json, digest, evaluate, wilson
from model_regression_gate.model import infer, train

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "evidence"


def save(name: str, value: Any) -> None:
    path = OUT / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def load(name: str) -> Any:
    return json.loads((ROOT / "data" / name).read_text())


def main() -> None:
    if subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip():
        raise SystemExit("Commit the implementation and freeze inputs before measuring")
    started = datetime.now(timezone.utc).isoformat()
    wall = time.perf_counter()
    cpu = time.process_time()
    policy = load("policy.json")
    rows = load("train.json")
    baseline_rows = [r for r in rows if r["slice"].endswith("/standard")]
    labels = policy["labels"]
    rotated = copy.deepcopy(rows)
    for row in rotated:
        row["label"] = labels[(labels.index(row["label"]) + 1) % len(labels)]
    models = {
        "baseline": train(baseline_rows),
        "improved": train(rows),
        "regressed": train(rotated),
    }
    for seed in range(10):
        shuffled = list(baseline_rows)
        random.Random(seed).shuffle(shuffled)
        models[f"control-{seed}"] = train(shuffled)
    for name, model in models.items():
        save(f"models/{name}.json", model)
    cases: Json = {}
    for partition in ["development", "evaluation"]:
        tasks = {"task_version": policy["task_version"], "rows": load(f"{partition}.json")}
        save(f"{partition}/tasks.json", tasks)
        outputs = {name: infer(model, tasks) for name, model in models.items()}
        for name, predictions in outputs.items():
            save(f"{partition}/predictions/{name}.json", predictions)
        reports = {
            name: evaluate(tasks, outputs["baseline"], predictions, policy)
            for name, predictions in outputs.items()
        }
        for name, report in reports.items():
            save(f"{partition}/reports/{name}.json", report)
        cases[partition] = {name: report["decision"] for name, report in reports.items()}
        if partition == "evaluation":
            human_policy = {**policy, "human_review_required": True}
            small_tasks = {**tasks, "rows": tasks["rows"][:5]}
            missing = copy.deepcopy(outputs["improved"])
            del missing["predictions"][tasks["rows"][0]["id"]]
            invalid = copy.deepcopy(outputs["improved"])
            invalid["predictions"][tasks["rows"][0]["id"]] = "unknown"
            uncertain_policy = {**policy, "slices": {k: 0.0 for k in policy["slices"]}}
            variants = {
                "human-review": (tasks, outputs["baseline"], outputs["improved"], human_policy),
                "insufficient": (
                    small_tasks,
                    infer(models["baseline"], small_tasks),
                    infer(models["improved"], small_tasks),
                    policy,
                ),
                "missing": (tasks, outputs["baseline"], missing, policy),
                "invalid": (tasks, outputs["baseline"], invalid, policy),
                "uncertain": (tasks, outputs["baseline"], outputs["baseline"], uncertain_policy),
            }
            for name, args in variants.items():
                save(
                    f"cases/{name}.json",
                    dict(zip(["tasks", "reference", "candidate", "policy"], args)),
                )
                report = evaluate(*args)
                save(f"cases/{name}-report.json", report)
                cases[name] = report["decision"]
    rejections = sum(cases["evaluation"][f"control-{s}"] == "FAIL" for s in range(10))
    nonpasses = sum(cases["evaluation"][f"control-{s}"] != "PASS" for s in range(10))
    metadata = {
        "started_utc": started,
        "finished_utc": datetime.now(timezone.utc).isoformat(),
        "code_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "logical_cpus": os.cpu_count(),
        "hardware": subprocess.check_output(
            ["sysctl", "-n", "machdep.cpu.brand_string"], text=True
        ).strip(),
        "memory_bytes": int(subprocess.check_output(["sysctl", "-n", "hw.memsize"], text=True)),
        "thread_limits": {
            k: os.environ.get(k)
            for k in ["OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"]
        },
        "execution": "single-process, sequential, Python standard library only",
        "wall_seconds": time.perf_counter() - wall,
        "process_cpu_seconds": time.process_time() - cpu,
        "peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "rss_units": "bytes on macOS; includes interpreter and accumulated experiment state",
        "seeds": list(range(10)),
        "policy_hash": digest(policy),
        "data_hashes": {
            name: digest(load(f"{name}.json")) for name in ["train", "development", "evaluation"]
        },
        "model_hashes": {name: digest(model) for name, model in models.items()},
        "decisions": cases,
        "control_false_rejections": rejections,
        "control_nonpasses": nonpasses,
        "control_count": 10,
        "illustrative_iid_95pct_wilson": wilson(rejections, 10, 0.05),
        "control_limit": "Repeated order-invariance controls are dependent and share one task set. "
        "The IID interval is illustrative and not a population error bound.",
        "cost_limit": "Local CPU only; monetary cost and energy were not measured.",
    }
    save("run.json", metadata)
    manifest = {
        str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
        for folder in ["evidence", "data", "src", "scripts"]
        for p in sorted((ROOT / folder).rglob("*"))
        if p.is_file() and "__pycache__" not in str(p) and p.name != "manifest.json"
    }
    save("manifest.json", manifest)
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
