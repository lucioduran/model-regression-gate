import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from model_regression_gate.gate import digest, evaluate, wilson
from model_regression_gate.model import infer, predict, train


def fixture(n=200):
    tasks = {
        "task_version": "v1",
        "rows": [{"id": str(i), "label": "a", "slice": "main", "text": "test"} for i in range(n)],
    }
    batch = {
        "task_version": "v1",
        "model_hash": "a" * 64,
        "tasks_hash": digest(tasks),
        "predictions": {str(i): "a" for i in range(n)},
    }
    policy = {
        "schema": 1,
        "task_version": "v1",
        "labels": ["a", "b"],
        "alpha": 0.05,
        "min_samples": 50,
        "slices": {"main": 0.1},
        "human_review_required": False,
    }
    return tasks, batch, copy.deepcopy(batch), policy


class GateTests(unittest.TestCase):
    def test_no_change_pass(self):
        args = fixture()
        report = evaluate(*args)
        self.assertEqual(report["decision"], "PASS")
        self.assertEqual(report["hashes"]["tasks"], digest(args[0]))

    def test_regression_fail(self):
        args = fixture()
        args[2]["predictions"] = {str(i): "b" for i in range(200)}
        self.assertEqual(evaluate(*args)["decision"], "FAIL")

    def test_improvement_pass(self):
        args = fixture()
        args[1]["predictions"] = {str(i): "b" for i in range(200)}
        report = evaluate(*args)
        self.assertEqual(report["decision"], "PASS")
        self.assertEqual(report["slices"]["main"]["gains"], 200)

    def test_uncertain_boundary(self):
        args = fixture()
        args[2]["predictions"].update({str(i): "b" for i in range(20)})
        self.assertEqual(evaluate(*args)["decision"], "REVIEW_REQUIRED")

    def test_small_sample_and_empty_slice(self):
        args = fixture(5)
        self.assertEqual(evaluate(*args)["decision"], "REVIEW_REQUIRED")
        args = fixture()
        args[3]["slices"]["unobserved"] = 0.1
        self.assertEqual(evaluate(*args)["decision"], "REVIEW_REQUIRED")

    def test_human_review_and_failure_precedence(self):
        args = fixture()
        args[3]["human_review_required"] = True
        self.assertEqual(evaluate(*args)["decision"], "REVIEW_REQUIRED")
        del args[2]["predictions"]["0"]
        self.assertEqual(evaluate(*args)["decision"], "FAIL")

    def test_contract_mutations(self):
        mutations = [
            lambda a: a[2]["predictions"].pop("0"),
            lambda a: a[2]["predictions"].update(extra="a"),
            lambda a: a[2]["predictions"].update({"0": "unknown"}),
            lambda a: a[2].update(task_version="v2"),
            lambda a: a[2].update(model_hash="invalid"),
            lambda a: a[2].update(tasks_hash="0" * 64),
            lambda a: a[0]["rows"].append(a[0]["rows"][0]),
            lambda a: a[0]["rows"][0].update(slice="unknown"),
            lambda a: a[0]["rows"][0].update(text=""),
            lambda a: a[3].update(alpha=float("nan")),
            lambda a: a[3].update(alpha=0),
            lambda a: a[3].update(min_samples=True),
            lambda a: a[3].update(human_review_required="yes"),
            lambda a: a[3].update(labels=["a", "a"]),
            lambda a: a[3].update(slices={"main": -0.1}),
            lambda a: a[3].update(slices={"main": float("inf")}),
            lambda a: a[3].update(schema=9),
        ]
        for mutation in mutations:
            with self.subTest(mutation=mutation):
                args = fixture()
                mutation(args)
                report = evaluate(*args)
                self.assertEqual(report["decision"], "FAIL")
                self.assertTrue(report["contract_errors"])

    def test_wrong_shapes_fail_closed(self):
        for index in range(4):
            for bad in [None, [], 5, "bad"]:
                args = list(fixture())
                args[index] = bad
                self.assertEqual(evaluate(*args)["decision"], "FAIL")

    def test_wilson_known_value_and_symmetry(self):
        lower, upper = wilson(0, 100, 0.05)
        self.assertAlmostEqual(lower, 0)
        self.assertAlmostEqual(upper, 0.0369934982)
        self.assertAlmostEqual(wilson(100, 100, 0.05)[0], 1 - upper)
        for args in [(1, 0, 0.05), (2, 1, 0.05), (0, 1, 0)]:
            with self.assertRaises(ValueError):
                wilson(*args)

    def test_cli_exit_codes_and_missing_file(self):
        with tempfile.TemporaryDirectory() as temp:
            paths = [
                Path(temp) / f"{name}.json"
                for name in ["tasks", "reference", "candidate", "policy"]
            ]
            args = fixture()
            command = [sys.executable, "-m", "model_regression_gate.cli"]
            for name, path, value in zip(
                ["tasks", "reference", "candidate", "policy"], paths, args
            ):
                path.write_text(json.dumps(value))
                command.extend([f"--{name}", str(path)])
            self.assertEqual(subprocess.run(command, capture_output=True).returncode, 0)
            args[3]["human_review_required"] = True
            paths[3].write_text(json.dumps(args[3]))
            self.assertEqual(subprocess.run(command, capture_output=True).returncode, 2)
            paths[0].unlink()
            result = subprocess.run(command, capture_output=True)
            self.assertEqual(result.returncode, 1)
            self.assertEqual(json.loads(result.stdout)["decision"], "FAIL")


class ModelTests(unittest.TestCase):
    def test_train_infer_and_roundtrip(self):
        rows = [
            {"id": "1", "text": "refund charge", "label": "billing"},
            {"id": "2", "text": "password login", "label": "access"},
        ]
        model = train(rows)
        self.assertEqual(predict(model, "refund"), "billing")
        self.assertEqual(predict(model, "login"), "access")
        self.assertEqual(predict(model, "unseen"), "access")
        self.assertEqual(model, train(list(reversed(rows))))
        tasks = {"task_version": "v1", "rows": rows}
        self.assertEqual(infer(model, tasks), infer(json.loads(json.dumps(model)), tasks))
        with self.assertRaises(ValueError):
            train([])

    def test_frozen_partitions_disjoint(self):
        data = Path(__file__).resolve().parents[1] / "data"
        partitions = [
            json.loads((data / f"{name}.json").read_text())
            for name in ["train", "development", "evaluation"]
        ]
        for i, left in enumerate(partitions):
            for right in partitions[i + 1 :]:
                for field in ["id", "text"]:
                    self.assertFalse({r[field] for r in left} & {r[field] for r in right})


if __name__ == "__main__":
    unittest.main()
