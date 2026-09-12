# TAFR replay method

All replay variants share the Naive model, initialization, optimizer reset, 20-epoch budget, loss, learning rate, weight decay, validation schedule, and final-epoch policy. Each starts independently from seed 42. E1 has an empty buffer and therefore uses the exact Naive batch path. From E2 onward, each epoch visits every current row once in deterministic chunks of at most 192, samples replay with replacement at the fixed 0.25 ratio, shuffles the combined batch deterministically, and performs one optimizer step.

After training each experience, the 2,000-item memory is completely rebuilt from the retained buffer and current development rows. Stable sample IDs are unique. Permanently discarded historical rows cannot reappear. Cumulative class counts use development rows only, including recurring Normal.

For class `c`, forgetting is `F_c = max(0, A_best_c - A_current_c)` using recall on the retained prior buffer. A missing retained class scores zero. Uncertainty is the mean candidate value `1 - max_k softmax(logits)_k`; rarity is `1 / sqrt(N_c)`. Each signal is min-max normalized across currently seen classes; a zero range becomes all zeros.

- Uniform Replay: `P_c = 1`
- TAFR-F: `P_c = F_hat_c`
- TAFR-FU: `P_c = 0.5 F_hat_c + 0.5 U_hat_c`
- Full TAFR: `P_c = (F_hat_c + U_hat_c + R_hat_c) / 3`

All-zero priorities trigger a recorded uniform fallback. Quotas start at one quarter of the uniform quota, are capped at twice the uniform quota where candidate supply permits, and use iterative deterministic largest-remainder allocation. Upper caps relax only when their sum cannot fill the eligible target. Within a class, candidates are stable-ID sorted and permuted from SHA-256 of seed, experience, canonical class ID, and buffer version. The method name is deliberately absent, so equal quotas select equal prefixes.

This forgetting estimate is a fixed-memory proxy, not an oracle over discarded data. Neither validation nor official-test samples influence buffer allocation.
