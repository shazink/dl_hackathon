# Context manifest

Read repository context in this order:

1. `AGENTS.md` — concise operating instructions.
2. The authoritative project specification, if one is added later — scientific goals and requirements; it overrides planning documents when they conflict.
3. `.ai/task_state.md` — current phase, completed work, next task, and blockers.
4. `.ai/decisions.md` — accepted and unresolved scientific or architectural decisions.
5. `docs/experimental_protocol.md` — fixed protocol, leakage boundaries, fairness rules, and metrics.
6. Relevant implementation documents and source files — planned architecture, phased work, configuration, and actual behavior.

Information ownership:

- `AGENTS.md` owns agent workflow rules.
- An authoritative project specification, once identified, owns research intent and method definitions.
- `.ai/task_state.md` owns current progress.
- `.ai/decisions.md` owns decision history.
- `.ai/data_card.md` owns verified dataset facts and inspection questions.
- `.ai/experiment_log.md` owns records of experiments actually run.
- `docs/experimental_protocol.md` owns the operational scientific protocol.
- Source, tests, and configurations own implemented behavior.

Keep these files synchronized. When implementation or project state changes, update the owning file and any directly affected summaries; do not resolve conflicts by silently rewriting fixed constraints.
