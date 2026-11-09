# Decision design frozen before evaluation

Task support-intent-v1 routes owned synthetic English support tickets among
billing, access and delivery. This is a multinomial naive Bayes classifier,
not an LLM. Baseline training includes only standard vocabulary. Candidate
training adds expanded vocabulary. The deliberate regression trains with
rotated class labels. Ten controls shuffle baseline training order with seeds
0 through 9. Training order should have no effect on count-based estimation.

The policy is specified by engineering requirements, not fitted to evaluation:
at most 0.10 absolute accuracy loss per named slice, at least 50 observations
per slice, family-wise nominal alpha 0.05. Development is a separate diagnostic
partition, and does not change thresholds. Evaluation is frozen before training.
Exact text and identifier overlap between partitions is forbidden. Numeric
request identifiers create distinct texts but do not create independent semantic
examples. The fixtures contain only twelve intent words and shared templates.
No generalization or production false-rejection claim is supported.

For each paired prediction, count gains (candidate correct, reference wrong)
and losses (reference correct, candidate wrong). Compute Wilson intervals for
each proportion at alpha/(2 * number of slices), using a Bonferroni union bound.
The difference interval is [gain.lower - loss.upper, gain.upper - loss.lower].
This is a conservative construction from approximate Wilson intervals, not an
exact finite-sample guarantee. The normal quantile is computed by statistics.NormalDist.

PASS requires every slice lower bound >= negative tolerance. FAIL requires a
slice upper bound < negative tolerance or any deterministic contract violation.
Otherwise REVIEW_REQUIRED captures uncertainty or insufficient sample counts.
An explicit human_review_required flag forces REVIEW_REQUIRED unless a failure
already applies. This system never records or invents human approval.

Contracts require exact IDs, labels, versioned task identity, known slices,
finite valid policy values, unique task IDs and nonempty text. Extra, missing,
or invalid predictions fail closed. Empty expected slices require review.
Reports bind canonical SHA-256 hashes of all gate inputs. Saved inference
artifacts also bind the model, training data and task data. Replay verifies
consistency with saved files, not authenticity of the original measurement.
