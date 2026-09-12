# Planned architecture

The complete pipeline is implemented: inventory/role verification, logical-training loading, deterministic splitting/experiences, frozen E1 preprocessing, Naive and replay training, audit-gated final evaluation, and a local dashboard. The component flow is:

`data ingestion → validation/splitting → preprocessing → experience construction → training strategy → replay memory → evaluation → artifact logging`

- **Data ingestion:** future loaders obtain logical paths through `verified_split_paths(data_dir, split_manifest)`, which validates current structure, hashes, counts, and roles without profiling targets. Never infer roles from physical filenames. File paths must be used immediately after validation; external file mutation afterward is not prevented by this API.
- **Validation/splitting:** validate structure and derive validation only from official training data.
- **Preprocessing:** fit transformations only on permitted training data and apply frozen transformations downstream.
- **Experience construction:** build the four class-incremental experiences after training class counts are inspected.
- **Training strategy:** provide a shared loop and controlled method-specific behavior.
- **Replay memory:** fixed 2,000-item complete replacement with Uniform Replay, TAFR-F, TAFR-FU, and full TAFR quota policies.
- **Evaluation:** produce required predictive and continual-learning metrics without leaking test information.
- **Artifact logging:** record configurations, provenance, runtime, memory, metrics, matrices, and generated outputs.

The data package now separates these implemented responsibilities into `inspection`, `loader`, `splits`, `experiences`, `preprocessing`, and `artifacts`. Non-data package subdirectories still contain import markers only.
