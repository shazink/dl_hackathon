# Phase 5 naive sequential baseline

The Phase 5 baseline is locked in `configs/experiments/naive.toml`. It uses one 156-input, ten-output MLP with hidden widths 256 and 128. Each hidden linear layer is followed by LayerNorm, ReLU, and dropout 0.20. Hidden weights use Kaiming initialization. Training uses unweighted cross-entropy and AdamW (learning rate 0.001, weight decay 0.0001), batch size 256, 20 epochs per experience, and gradient clipping at norm 5.0. The optimizer resets at each experience; model weights do not. There is no scheduler, early stopping, validation-based model selection, replay, or preprocessing fit.

The runner validates every prepared-file hash, both preparation configuration hashes, the pinned logical-training hash, assignment fingerprint, class mapping, dtypes, finite features, label ranges, shapes, stable 156-feature dimension, and mutually disjoint sample identifiers before constructing the model. It accepts only the exact training preparation artifact set and never loads logical-test data.

Run from the repository root:

```bash
python scripts/run_experiment.py \
  --method naive \
  --config configs/experiments/naive.toml \
  --prepared-dir artifacts/data/prepared \
  --output-dir artifacts/experiments/naive_seed42
```

To continue an interrupted run, repeat the command with `--resume`. Resume is allowed only when `checkpoints/latest.pt` matches both the complete experiment-config fingerprint and prepared-data fingerprint. Checkpoints are atomically replaced after every epoch and experience and contain model and optimizer states, progress, RNG states, full configuration, fingerprints, metric history, and accumulated runtimes. A new run refuses a nonempty output directory.

Before E1, the randomly initialized model is evaluated separately on E1–E4 validation. After each experience it is evaluated on every task validation subset and on the union of validation subsets seen so far. Validation never changes training. Task subsets contain mutually disjoint Normal rows and different newly introduced attacks. Per-evaluation JSON records loss, accuracy, balanced accuracy, Macro-F1 over target-present classes, global per-class recall/support with absent classes explicit, and a global-order confusion matrix.

For a metric, let `R[i,j]` be performance on validation task `j` after training task `i`; `R[0,j]` is random-initialization performance and there are `T=4` tasks:

- Final average: `(1/T) sum_j R[T,j]`.
- Per-task forgetting: `F_j = max_{i=j..T} R[i,j] - R[T,j]` (one-based training rows).
- Average forgetting: `(1/T) sum_j F_j`.
- Forward transfer: `(1/(T-1)) sum_{j=2..T} (R[j-1,j] - R[0,j])`.

Balanced accuracy is the primary forgetting and forward-transfer measure. Accuracy-based forgetting and summaries for all three matrices are retained. Final seen-class Macro-F1 is computed on the E1–E4 validation union after E4. `results.json` also records environment and Git state, complete config and fingerprints, epoch/experience/total runtime, process peak resident memory, CUDA peaks when applicable, parameter count, and checkpoint size. Unsupported resource measurements are `null` with an explanation.
