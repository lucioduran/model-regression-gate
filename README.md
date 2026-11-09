# Model Regression Gate

A reproducible, fail-closed promotion gate with paired accuracy comparisons,
slice tolerances, uncertainty, and a separate human-review-required state.

## Problem

An aggregate score can hide a regression in a small customer segment. A model
release also needs to distinguish broken output contracts from uncertain
measurements and decisions that still require a person.

## Scope

This Python package compares versioned reference and candidate predictions.
It includes real local training and inference for a small multinomial naive
Bayes support-ticket classifier. It is not an LLM benchmark. All fixtures are
owned synthetic English examples, licensed under MIT along with the code.
No hosted inference, paid API, external model download, or GPU is required.

## Quick start

Use Python 3.15.0 for the recorded environment. The runtime has zero dependencies.
From a cloned checkout, the following verification is entirely offline:

```sh
PYTHONPATH=src python3 scripts/replay.py
PYTHONPATH=src python3 -m unittest discover -s tests -v
```

Replay checks 90 SHA-256 file hashes and recomputes 31 decisions from saved
predictions. It does not retrain models or establish measurement authenticity.

For an installed CLI, build and install with the pinned tooling:

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements-dev.txt
.venv/bin/python -m pip wheel --no-build-isolation --no-deps --wheel-dir dist .
.venv/bin/python -m pip install --no-deps dist/model_regression_gate-0.1.0-py3-none-any.whl
```

## Usage

```sh
.venv/bin/model-regression-gate \
  --tasks evidence/evaluation/tasks.json \
  --reference evidence/evaluation/predictions/baseline.json \
  --candidate evidence/evaluation/predictions/improved.json \
  --policy data/policy.json
```

The command emits a JSON report. Exit codes are 0 for PASS, 1 for FAIL, and 2
for REVIEW_REQUIRED. Candidate files named `regressed.json` demonstrate FAIL.
Use `evidence/cases/` for complete human-review, uncertain, insufficient,
missing-prediction and invalid-label inputs with their saved reports.

A task bundle contains `task_version` and rows with unique `id`, nonempty `text`,
`label`, and `slice`. Prediction bundles contain the same version, a canonical
`tasks_hash`, a 64-character SHA-256 `model_hash`, and an exact ID-to-label map.
The policy declares labels, slice tolerances, alpha, minimum samples and an
explicit human review flag. `model_regression_gate.gate.digest` computes
canonical JSON identities. A caller-supplied model hash is an identity claim;
the gate cannot authenticate the caller or prove inference occurred.

## Architecture

- `src/model_regression_gate/gate.py`: validation, paired intervals and decisions.
- `src/model_regression_gate/model.py`: deterministic count-based training and inference.
- `scripts/experiment.py`: sequential local experiment and evidence capture.
- `scripts/replay.py`: offline hash and decision verification without inference.
- `data/`: frozen train, development, evaluation and policy files.
- `evidence/`: model artifacts, raw predictions, reports and execution metadata.

Each slice measures gains and losses on the same tasks. Bonferroni-adjusted
Wilson intervals give an approximate difference interval. PASS requires every
slice's lower bound to meet its tolerance; a supported regression or a contract
violation yields FAIL. Insufficient samples, overlapping bounds or an explicit
human-review flag yield REVIEW_REQUIRED. FAIL takes precedence over review.
No human approval is recorded. See [the design](docs/design.md) for the formula.

## Reproduction

Inputs and policy were frozen at `c2fe489`; model execution used source commit
`2937f6b7efb3ec07f12ab2253d56a5610e3573c8`. The final release adds evidence,
documentation and CI without changing the measured implementation. Canonical
model/data/policy identities are in [run.json](evidence/run.json); raw file
hashes are in [manifest.json](evidence/manifest.json). The runtime is Python
3.15.0, with standard-library math and no native model dependency.

Training uses 60 standard-vocabulary examples for baseline and 120 examples
for the improved model. The negative model rotates all training labels. Ten
controls shuffle baseline training with seeds 0 through 9. Inference runs on
300 development and 600 held-out evaluation texts. Numeric identifiers and
partition-specific prefixes ensure exact text disjointness, but these are not
independent semantic examples. Thresholds were specified before measurement,
not fitted on either partition. Development results did not alter the policy.

On the original host, a clean checkout can rerun the experiment with:

```sh
python3 /Users/luchoduran/Sites/open-source-projects/.coordination/benchmark_lock.py \
  .venv/bin/python scripts/experiment.py
```

This wrapper holds the shared exclusive lock for the full process lifetime and
sets CPU library thread limits to two. Actual computation is single-process,
sequential Python. The experiment requires a clean Git tree and overwrites
tracked evidence on rerun, so preserve the published evidence in Git. On other
hosts, provide your own exclusive lock wrapper; hardware capture currently
requires macOS `sysctl`. Lightweight replay works without that wrapper or macOS.

## Results

Recorded on November 9, 2026, using an Intel Xeon W-3223 at 3.50 GHz,
16 logical CPUs and 48 GiB RAM, macOS 26.7.1, Python 3.15.0.

| Evaluation candidate | Correct / 600 | Decision |
| --- | ---: | --- |
| Baseline against itself | 400 | PASS |
| Improved vocabulary | 600 | PASS |
| Rotated training labels | 0 | FAIL |
| Ten shuffled-order baseline controls | 400 each | PASS, all ten |

Each evaluation slice has 100 rows, tolerance 0.10 and minimum count 50.
Unchanged slice intervals are approximately [-0.07587, 0.07587]. Improvements
on billing/expanded and delivery/expanded are +1.0, with intervals
[0.84826, 1.0]. The corrupted model has four -1.0 slice regressions.

Missing and invalid outputs fail deterministically. Human-review, insufficient
sample and zero-tolerance uncertainty cases require review. At 50 rows per
slice, the development baseline and improved model require review: satisfying
the minimum sample count does not establish noninferiority. These negative
promotion results are retained in the evidence.

Observed control false rejections: 0/10; evaluation control nonpasses: 0/10.
Development controls have 0/10 failures but 10/10 review outcomes. A hypothetical
IID Wilson 95% interval for 0/10 is [0, 0.27753], but these order-invariance
controls are dependent and are the same effective model. This is a consistency
check, not an estimate of production false-positive probability.

The recorded experiment took 0.4965 seconds wall time and 0.4056 seconds process
CPU time, with peak process RSS 25,161,728 bytes. This includes training,
inference, report writing and hardware metadata queries; it is not isolated
inference latency. The finish timestamp marks completion of cases, before
hardware queries; timer readings are taken afterward and before manifest writing.
No monetary cost or energy measurement was performed.

## Testing

```sh
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/ruff check src scripts tests
.venv/bin/ruff format --check src scripts tests
.venv/bin/mypy src scripts
.venv/bin/python scripts/replay.py
```

GitHub CI runs tests, lint, formatting, strict source/script type checking,
evidence replay and a wheel-installed CLI smoke check on Python 3.15.0.
Tests cover contract failures, paired improvement/regression, uncertainty,
minimum counts, human-review precedence, model roundtrip and partition overlap.
No heavy model experiment runs in CI.

## Limitations

Only twelve intent words and shared templates underlie the fixtures. Numeric
variants do not provide independent evidence. The classifier learns lexical
routing, and the improved model has training vocabulary coverage by construction.
Neither perfect fixture accuracy nor the nominal intervals support claims about
natural support traffic, LLMs, distribution shift or adversarial robustness.
Wilson bounds are approximate, including after multiple-slice adjustment.
The gate compares relative accuracy, so a poor baseline can still pass unchanged.
Absolute quality requirements must be specified separately by the application.

SHA-256 manifests detect accidental changes relative to a trusted checkout;
they are neither signatures nor independent attestations. Replay checks saved
consistency only. There is no reviewed dataset, human approval, GPU result,
cross-hardware performance claim or production rollout in this release.

## Contributing

Add small, tested changes and retain explicit task and artifact versions.
Freeze new decision policies and partitions before measuring. Preserve negative
results. Use free local resources and an exclusive benchmark lock for training
or evaluation. Never silently revise published evidence or claim human review.

## License

[MIT](LICENSE) for code, documentation, model artifacts and owned fixtures.
The fixture license is also included at [data/LICENSE](data/LICENSE).
