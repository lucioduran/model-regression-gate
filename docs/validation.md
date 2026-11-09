# Validation history

The first locked run is preserved in commit 7ed1cd8. Its predictions and gate
results are real, but its manifest incorrectly included ignored egg-info files.
A wheel build changed generated SOURCES.txt and replay caught the mismatch.
The final experiment excludes generated package metadata from the manifest.
No model code, task, decision threshold or prediction was changed to fix this.
The full experiment was repeated under the shared lock after committing the fix.

The initially pinned build 1.3.0 frontend failed on Python 3.15.0 with
`TypeError: HelpFormatter.__init__() got an unexpected keyword argument 'color'`.
Packaging now uses pip wheel with the pinned setuptools backend. The resulting
wheel was installed into a separate environment and the CLI passed using the
saved held-out candidate predictions.

The final validation commands are documented in README.md and CI. Saved
measurement results are consistency evidence, not independently authenticated
benchmarks. The ten shuffled-order controls are deterministic consistency
controls, not ten independent trials. Development review outcomes are retained
separately from regression failures.
