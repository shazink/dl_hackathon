# Configurations

`unsw_nb15_splits.toml` is the portable logical split manifest for the current Kaggle mirror. It records URLs, official counts, physical filenames, pinned SHA-256 values, resolution status, and provenance limitations. The physical presplit names are swapped; consumers must validate and use logical roles. Pass the dataset directory separately; never put machine-specific paths in the manifest.

`experiences.toml` locks protocol v1.0, multiclass indices, development-only attack counts and snake allocation, recurring Normal policy, source fingerprint, and assignment fingerprint. Preparation recomputes and validates it before fitting any transformer. Do not alter it in response to model results.

Training model configurations are not yet implemented.
