# Agent instructions

1. Read `.ai/context_manifest.md` first, then inspect `.ai/task_state.md` before changing code.
2. Inspect existing code before editing and make narrowly scoped changes.
3. Preserve the scientific constraints in `docs/experimental_protocol.md`. Never use test data for fitting, tuning, early stopping, task design, or feature selection.
4. Never fabricate or conceal results. Label unexecuted experiments clearly.
5. Run relevant tests after changes. Update documentation and `.ai/task_state.md` when behavior or project state changes.
6. Record material scientific or architectural decisions in `.ai/decisions.md`. Append to `.ai/experiment_log.md` only for commands and results actually executed.
7. Add dependencies only for a concrete need.
8. Never commit or push unless explicitly requested.

Detailed authority and reading order are defined in `.ai/context_manifest.md`.
