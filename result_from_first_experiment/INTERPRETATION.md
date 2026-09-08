# Headline numbers — auto-generated

This is a draft of the headline result table. Fill in narrative around it.

| Policy | Capacity | Lookups | Hit rate | Shared hit rate | P50 (ms) | P99 (ms) | Goodput |
|---|---|---:|---:|---:|---:|---:|---:|
| FIFO (naive) | constrained | 582 | 0.555 | 0.283 | 779 | 8280 | 0.99 |
| FIFO (naive) | generous | 582 | 0.644 | 0.378 | 760 | 7285 | 0.99 |
| Session-aware | constrained | 582 | 0.605 | 0.301 | 770 | 1119 | 0.99 |
| Session-aware | generous | 582 | 0.689 | 0.410 | 746 | 1092 | 0.99 |
| Sharing-aware | constrained | 582 | 0.687 | 0.331 | 745 | 1012 | 0.99 |
| Sharing-aware | generous | 582 | 0.653 | 0.235 | 724 | 1847 | 0.99 |
| Combined | constrained | 582 | 0.625 | 0.216 | 757 | 1644 | 0.99 |
| Combined | generous | 582 | 0.696 | 0.330 | 712 | 1168 | 0.99 |

## Diagnostic questions to address in the writeup

- Does the **Combined** policy beat both single-signal baselines on the constrained-capacity hit rate? By how much?
- Does the advantage shrink or disappear at the **generous** capacity? (expected: yes — at low pressure, ordering doesn't matter much.)
- Is the per-session fairness ECDF for Combined reasonable, or is it skewed toward protecting some sessions at the cost of others?
- Does the **Sharing-aware** policy's hit rate on shared content stay higher than **Session-aware**'s? (sanity check on the policy's design.)
- If Combined does NOT clearly beat both baselines, what does the diagnosis say? (e.g. 'combining helps only at high overlap_fraction; at low overlap, session-awareness alone is sufficient'.)