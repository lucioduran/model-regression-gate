"""Small multinomial naive Bayes intent classifier, with JSON artifacts."""

from collections import Counter
import math
import re
from typing import Any

from .gate import Json, digest


def tokens(text: str) -> list[str]:
    return re.findall(r"[a-z]+", text.lower())


def train(rows: list[Json]) -> Json:
    if not rows:
        raise ValueError("Training data is empty")
    labels = sorted({r["label"] for r in rows})
    counts = {label: Counter[str]() for label in labels}
    documents = Counter[str]()
    for row in rows:
        documents[row["label"]] += 1
        counts[row["label"]].update(tokens(row["text"]))
    vocabulary = sorted({token for counter in counts.values() for token in counter})
    return {
        "schema": 1,
        "algorithm": "multinomial-nb-laplace-v1",
        "labels": labels,
        "vocabulary": vocabulary,
        "documents": dict(documents),
        "counts": {k: dict(v) for k, v in counts.items()},
        "training_hash": digest(sorted(rows, key=lambda r: r["id"])),
    }


def predict(model: Json, text: str) -> str:
    vocabulary = set(model["vocabulary"])
    words = [w for w in tokens(text) if w in vocabulary]
    total_documents = sum(model["documents"].values())
    scores = {}
    for label in model["labels"]:
        counts = model["counts"][label]
        denominator = sum(counts.values()) + len(vocabulary)
        score = math.log(model["documents"][label] / total_documents)
        for word in words:
            score += math.log((counts.get(word, 0) + 1) / denominator)
        scores[label] = score
    return str(max(sorted(scores), key=lambda label: scores[label]))


def infer(model: Json, tasks: Json) -> dict[str, Any]:
    return {
        "task_version": tasks["task_version"],
        "model_hash": digest(model),
        "tasks_hash": digest(tasks),
        "predictions": {r["id"]: predict(model, r["text"]) for r in tasks["rows"]},
    }
