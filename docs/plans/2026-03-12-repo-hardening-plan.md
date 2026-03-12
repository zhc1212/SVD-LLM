# Repository Hardening Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Make the repository portable, testable, and CI-ready while replacing the whitening path's full-activation caching with a streaming statistics workflow.

**Architecture:** Introduce project-level test configuration, split unit vs integration coverage, compute whitening from streamed second-order statistics, and persist whitening-only stats per layer to disk for a two-pass compression flow. Keep the user-facing scripts largely intact while removing brittle local assumptions from default tests and imports.

**Tech Stack:** Python, PyTorch, pytest, Ruff, GitHub Actions

---

### Task 1: Add failing tests and pytest configuration

**Files:**
- Create: `tests/conftest.py`
- Modify: `tests/test_downstream.py`
- Modify: `tests/test_whitening.py`
- Modify: `tests/test_sequential.py`
- Modify: `tests/test_e2e.py`
- Modify: `tests/test_model_loader.py`
- Modify: `tests/test_calibration.py`
- Create: `pyproject.toml`

**Step 1: Write failing tests for import-path-independent pytest runs**

Add project-level pytest config and update tests so default runs do not depend on a hard-coded 7B path or CUDA.

**Step 2: Run focused tests to verify red state**

Run: `pytest tests/test_downstream.py tests/test_whitening.py -q`
Expected: failures around optional imports and missing stats-based APIs.

**Step 3: Add minimal test scaffolding**

Create fake model fixtures and integration markers.

**Step 4: Re-run focused tests**

Run: `pytest tests/test_downstream.py tests/test_whitening.py -q`
Expected: import-path issue resolved, tests still failing only for missing implementation.

### Task 2: Refactor calibration and whitening to streaming statistics

**Files:**
- Modify: `src/data/calibration.py`
- Modify: `src/compress/whitening.py`
- Modify: `tests/test_whitening.py`
- Modify: `tests/test_calibration.py`

**Step 1: Write failing tests for covariance accumulation**

Add a test that streaming stats produce the same covariance as the full activation matrix on a fake model.

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_calibration.py::test_collect_linear_input_covariance_matches_full_activations -q`
Expected: FAIL due to missing function.

**Step 3: Implement minimal stats collector and covariance-based whitening**

Add streaming accumulation of `X^T X` and sample counts, plus whitening from covariance statistics.

**Step 4: Run focused tests**

Run: `pytest tests/test_calibration.py tests/test_whitening.py -q`
Expected: PASS.

### Task 3: Rework sequential and whitening-only compression flows

**Files:**
- Modify: `src/compress/sequential_update.py`
- Modify: `scripts/compress.py`
- Modify: `tests/test_sequential.py`
- Modify: `tests/test_e2e.py`

**Step 1: Write failing test for memory-safe whitening-only compression**

Test that whitening-only compression no longer caches an in-memory dict of all activations and instead uses a per-layer stats workflow.

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_sequential.py::test_compress_model_whitening_only_smoke -q`
Expected: FAIL on old behavior assumptions or missing injectable stats directory.

**Step 3: Implement minimal two-pass stats persistence**

Collect one layer/linear covariance stat at a time during the original-model pass, persist it to temp storage, then load and compress during the mutation pass.

**Step 4: Run focused tests**

Run: `pytest tests/test_sequential.py tests/test_e2e.py -q`
Expected: PASS for unit-safe fake-model tests; integration tests skipped by default.

### Task 4: Make evaluation imports and docs/scripts consistent

**Files:**
- Modify: `src/eval/downstream.py`
- Modify: `README.md`
- Modify: `scripts/run_all_experiments.sh`
- Modify: `scripts/run_experiments_gpu02.sh`

**Step 1: Write failing test for formatting helper without `lm_eval`**

Keep `tests/test_downstream.py` focused on helper behavior.

**Step 2: Implement lazy import**

Move `lm_eval` imports into the evaluation function.

**Step 3: Align docs and script entrypoints**

Use `scripts/eval_model.py` consistently and document `truthfulqa_gen`.

**Step 4: Verify**

Run: `pytest tests/test_downstream.py -q`
Expected: PASS.

### Task 5: Add CI and verify repository health

**Files:**
- Create: `.github/workflows/ci.yml`
- Modify: `pyproject.toml`
- Optionally create: `.gitignore`

**Step 1: Add lint/test workflow**

Run lint and non-integration tests on push and pull request.

**Step 2: Run local verification**

Run:
- `ruff check .`
- `pytest -m "not integration"`

**Step 3: Fix any fallout**

Keep changes minimal and targeted.

**Step 4: Final verification**

Run the full verification commands again and record exact outcomes before claiming completion.
