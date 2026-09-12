# Validation benchmark

The frozen seed-42 five-method benchmark completed without result-dependent reruns. Concise metrics are tracked in `results/validation_benchmark_seed42.csv`; detailed matrices, per-class metrics, replay audits, checkpoints, and resource records remain under ignored `artifacts/experiments/benchmark_seed42/`.

| Method | Accuracy | Balanced accuracy | Macro-F1 | Accuracy forgetting | BA forgetting | Forward transfer | Runtime (s) | Peak CPU MiB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Naive | 0.461185 | 0.451694 | 0.163005 | 0.471333 | 0.388005 | 0.244349 | 45.72 | 1000.79 |
| Uniform | 0.702777 | 0.661604 | 0.474125 | 0.214653 | 0.147955 | 0.242722 | 48.61 | 1057.93 |
| TAFR-F | 0.685764 | 0.648476 | 0.439875 | 0.237144 | 0.179534 | 0.242880 | 49.13 | 1058.27 |
| TAFR-FU | 0.709590 | 0.660261 | 0.488744 | 0.198369 | 0.132802 | 0.241928 | 50.86 | 1058.14 |
| TAFR | 0.697590 | **0.670742** | 0.475885 | 0.212300 | **0.121674** | 0.242007 | 50.08 | 1048.18 |

Full TAFR ranked first by the predeclared primary criterion, final average validation balanced accuracy. It beat Naive and Uniform Replay on that metric and also had the lowest balanced-accuracy forgetting. TAFR-FU had the highest validation accuracy and Macro-F1, which is retained rather than obscured.

The audit found all 71 tests passing, finite strict-JSON results, matching preparation/config fingerprints, exact E1 result equivalence, identical locked non-strategy settings, valid buffer capacity/uniqueness/eligibility/provenance, an empty project test-access log, unchanged source/config fingerprints, and a clean `git diff --check`. Every replay update retained exactly 2,000 unique eligible samples. TAFR-F used the specified uniform fallback at E1 because all forgetting priorities were zero; no cap relaxation was needed in the produced runs.
