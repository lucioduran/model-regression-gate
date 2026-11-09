import json
from pathlib import Path
import subprocess
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]


class EvidenceTests(unittest.TestCase):
    def test_saved_decisions_and_negative_development(self):
        run = json.loads((ROOT / "evidence/run.json").read_text())
        self.assertEqual(run["decisions"]["evaluation"]["baseline"], "PASS")
        self.assertEqual(run["decisions"]["evaluation"]["improved"], "PASS")
        self.assertEqual(run["decisions"]["evaluation"]["regressed"], "FAIL")
        self.assertEqual(run["decisions"]["development"]["baseline"], "REVIEW_REQUIRED")
        self.assertEqual(run["decisions"]["development"]["improved"], "REVIEW_REQUIRED")
        for case in ["human-review", "insufficient", "uncertain"]:
            self.assertEqual(run["decisions"][case], "REVIEW_REQUIRED")
        for case in ["missing", "invalid"]:
            self.assertEqual(run["decisions"][case], "FAIL")
        self.assertEqual(run["control_false_rejections"], 0)
        self.assertEqual(run["control_nonpasses"], 0)
        self.assertEqual(len(set(run["model_hashes"][f"control-{s}"] for s in range(10))), 1)

    def test_replay(self):
        result = subprocess.run(
            [sys.executable, str(ROOT / "scripts/replay.py")], capture_output=True, text=True
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("replayed 31 decisions", result.stdout)


if __name__ == "__main__":
    unittest.main()
