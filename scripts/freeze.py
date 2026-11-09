"""Create owned, illustrative English fixtures before any measurements."""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WORDS = {
    "billing": ["invoice", "payment", "charge", "refund"],
    "access": ["password", "login", "account", "authentication"],
    "delivery": ["parcel", "shipment", "tracking", "courier"],
}


def main() -> None:
    for partition, count, prefix in [
        ("train", 20, "Please handle"),
        ("development", 50, "I need assistance with"),
        ("evaluation", 100, "Help me resolve"),
    ]:
        rows = []
        for label, words in WORDS.items():
            for style in ["standard", "expanded"]:
                for i in range(count):
                    word = words[i % 2 + (2 if style == "expanded" else 0)]
                    rows.append(
                        {
                            "id": f"{partition}-{label}-{style}-{i:03}",
                            "text": f"{prefix} {word} for request {i:03}",
                            "label": label,
                            "slice": f"{label}/{style}",
                        }
                    )
        (ROOT / "data" / f"{partition}.json").write_text(json.dumps(rows, indent=2) + "\n")
    policy = {
        "schema": 1,
        "task_version": "support-intent-v1",
        "labels": list(WORDS),
        "alpha": 0.05,
        "min_samples": 50,
        "slices": {
            f"{label}/{style}": 0.10 for label in WORDS for style in ["standard", "expanded"]
        },
        "human_review_required": False,
    }
    (ROOT / "data/policy.json").write_text(json.dumps(policy, indent=2) + "\n")


if __name__ == "__main__":
    main()
