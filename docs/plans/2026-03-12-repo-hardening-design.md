# Repository Hardening Design

**Context**

The repository now contains a first-pass SVD-LLM implementation, but it is not yet portable or CI-friendly. The current test suite depends on a local 7B model path and CUDA, `pytest` cannot import `src` from a clean checkout, and `svd_llm_w` materializes full activation tensors in a way that is not viable for LLaMA-7B-scale runs.

**Goals**

- Make local development and CI reliable from a clean checkout.
- Preserve the current research-oriented workflow without forcing 7B-model integration tests into default CI.
- Refactor whitening-based compression to avoid caching full activation matrices in GPU memory.
- Align scripts, tests, and docs around one executable path.

**Non-Goals**

- Re-architect the entire compression pipeline.
- Add baseline methods not already implemented.
- Make full 7B compression run inside CI.

**Recommended Approach**

Use a layered hardening pass:

1. Add project-level Python test configuration so `pytest` can import `src` directly.
2. Split tests into lightweight default unit tests and opt-in integration tests.
3. Refactor activation collection from full activation capture to streaming covariance accumulation.
4. Rework `svd_llm_w` to collect whitening statistics in a first pass and persist per-layer stats to disk instead of caching all activations in memory.
5. Move optional heavyweight imports behind runtime boundaries so helper functions remain testable without full experiment dependencies.
6. Add a small GitHub Actions workflow that runs linting and unit tests only.

**Architecture**

The central change is data-flow oriented. Whitening only needs second-order activation statistics, not the full activation matrix after collection. The refactor therefore introduces a streaming collector that accumulates `X^T X` and sample counts from forward hooks. Compression then consumes covariance statistics to build the whitening matrix. For `svd_llm_w`, the original model remains untouched during a stats pass, and each layer/linear stat is serialized to a temporary file. A second pass loads one stat at a time and applies compression in-place.

Tests will be reorganized around a small fake LLaMA-like model fixture so default checks exercise tensor logic and layer replacement without requiring the local 7B checkpoint. Real-model tests remain available behind an explicit marker and environment variable.

**Error Handling**

- Invalid or unsupported datasets continue to raise `ValueError`.
- Missing `lm_eval` should only break downstream evaluation entrypoints, not helper imports.
- Temporary whitening stats should be created under a deterministic temp directory and cleaned up after use.

**Testing Strategy**

- Unit tests:
  - import path works in clean pytest runs
  - downstream formatting works without `lm_eval`
  - covariance accumulation matches full-activation covariance
  - whitening-only compression no longer preloads all layer activations into a single in-memory map
  - replacement and rank math still hold
- Integration tests:
  - gated behind markers and local model path env vars
  - optional CUDA gating

**CI Strategy**

GitHub Actions should run:

- `ruff check .`
- `pytest -m "not integration"`

This keeps CI fast and meaningful while preserving room for later GPU or scheduled jobs.
