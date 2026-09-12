# Final evaluation

The automated audit passed, then full TAFR was selected and frozen from validation evidence before any model evaluation on logical test. The verified 82,332-row logical test was loaded and transformed once with the E1-fitted preprocessor. Each finalized checkpoint was evaluated exactly once. No training, selection, thresholding, refitting, or rerun followed from these results.

| Method | Accuracy | Balanced accuracy | Macro-F1 | Weighted F1 |
| --- | ---: | ---: | ---: | ---: |
| Naive | 0.490150 | 0.273016 | 0.156247 | 0.420071 |
| Uniform | 0.731745 | 0.539980 | 0.396174 | 0.732427 |
| TAFR-F | **0.732899** | 0.535941 | 0.379418 | 0.724251 |
| TAFR-FU | 0.730396 | 0.532885 | **0.412951** | **0.741546** |
| TAFR | 0.704028 | **0.545146** | 0.391536 | 0.721409 |

Full TAFR beat Naive and Uniform Replay on the primary balanced-accuracy metric, but not on every metric: TAFR-F had the highest accuracy, TAFR-FU had the highest Macro-F1 and weighted F1, and Uniform exceeded full TAFR on those three aggregate metrics. Detailed per-class precision, recall, F1, support, and each 10×10 confusion matrix remain in ignored `artifacts/experiments/benchmark_seed42/<method>/test_metrics.json`; chart-ready copies are tracked in `results/dashboard_data.json`.

The official test is now consumed and finalized for protocol v1.0. The unchanged limitations remain: duplicate/conflicting-label rows were preserved, this is a single-seed study, official byte identity is unverified, and the replay forgetting signal is a retained-memory proxy.
