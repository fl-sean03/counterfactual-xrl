# Metrics Specification — for report §IV.E

Three explanation-quality metrics. All take a
(`DecisionRecord`, `Explanation`) pair as input and are aggregated with
95% percentile bootstrap CIs (1000 resamples) per evidence source.

## 1. Fidelity

**Definition.** Fraction of numerical claims in the explanation that
match the evidence within tolerance $\epsilon = 0.1$.

**Operationalization** (`src/xrl/eval/metrics.py::fidelity_score`):

For each entry in `Explanation.claims`:
1. Look up the `(action, metric)` pair in the record's
   `per_action_stats`.
2. If the metric is one of `{mean_return, success_rate,
   collision_rate, std_return, mean_steps_to_end}`, compare the claimed
   `value` to the actual stat. Within $\epsilon$ → hit.
3. Claims citing out-of-scope metrics (e.g., `q_value`, `visit_count`)
   are skipped, not counted as failures — they can be informative but
   are not verifiable against rollout-style stats.

Score = hits / total_claims.

**Why bootstrap CI.** Per-decision scores are proportions on $[0, 1]$;
normal approximation is poor near 0 or 1. Percentile bootstrap with
1000 resamples handles both asymmetry and the small-n tails cleanly.

## 2. Soundness

**Definition.** How well the LLM explanation's rationale sentences are
supported by the evidence, rated by an independent LLM judge on a
3-point scale.

**Operationalization** (`src/xrl/eval/metrics.py::soundness_score`):

1. Split `Explanation.rationale` into sentences plus
   `Explanation.counterfactual` as one more sentence.
2. For each sentence, ask the judge (gpt-4o-mini) to output
   $\{0, 1, 2\}$ where 2 = fully supported by the evidence, 1 =
   partially supported, 0 = contradicted or invented.
3. Score = $\bar{s}/2 \in [0, 1]$.

**Judge prompt.** See `SOUNDNESS_SYSTEM` in metrics.py. The judge sees
the full evidence record and one sentence; it replies with only a
digit, which we parse robustly via regex.

**Guard against cheating.** The judge is a different-size model
(gpt-4o-mini) in the same family as the generator (gpt-4o). A
cross-family replication (Claude judge) is listed as future work.

## 3. Post-hoc inferability

**Definition.** Can a held-out judge recover the chosen action given
only the explanation?

**Operationalization** (`src/xrl/eval/metrics.py::inferability`):

1. Scrub every occurrence of `action {chosen_action}` from the
   rationale and counterfactual text using a case-insensitive regex.
2. Give the judge a minimal state description plus the scrubbed
   explanation and ask for the chosen action as a single digit
   $\{0, 1, 2\}$.
3. Score = 1 if judge's answer matches the true chosen action, else 0.

**Interpretation.** High inferability means the explanation carries
enough information for a reader to identify the decision; low
inferability means the explanation is too vague. We report it
alongside fidelity and soundness as a complement: an explanation can
be accurate (high fidelity) but uninformative (low inferability), or
vice versa.

## Reporting

The per-record CSV is at `results/metrics/per_record.csv`; the
summary JSON at `results/metrics/summary.json`. Both are regenerated
by `scripts/evaluate_explanations.py`.
