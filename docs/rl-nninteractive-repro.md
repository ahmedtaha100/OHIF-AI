# RL nnInteractive Repro README

This is the Phase 0 scaffold for the RL-over-nnInteractive tumor project. It is not a real benchmark, training run, or clinical result.

See `docs/data_manifest.md` for the current Phase 0 input/checkpoint manifest.

`make phase1-small` runs the local point-only Phase 1 proof-of-pipeline on
code-generated toy masks with the mock adapter. It exercises state encoding,
top-k FP/FN point candidates, behavior cloning, a tiny DQN-style update loop,
and a validation NoC table. The output is synthetic/mock only and must not be
reported as a real benchmark result.

`make phase2-smoke` runs the local Phase 2-4 code-surface smoke on the same
toy masks. It exercises deterministic component geometry for point/scribble/
lasso/box, multi-tool adapter dispatch, TTA-disagreement uncertainty, safety
reward terms, the OHIF-facing suggestion payload, interaction logging,
DAgger/preference helpers, STOP calibration, failure taxonomy, and ablation
grid scaffolding. It is also synthetic/mock only.

## Plan Scope

The authoritative execution plan is maintained outside the code repo in the Brain work docs at `work/OHIF-AI/Plan.md`. The current repo scaffold covers these checked Phase 0 items from that plan:

- Repo skeleton: package layout, deps/lock file, config, `make setup`, `make test`, repro README.
- Metric library + unit tests on SYNTHETIC masks: Dice, NSD, HD95, NoC@85, NoC@90, Dice@{1,3,5}.
- nnInteractive adapter interface: `set_image`, `set_target_buffer`, `target_buffer`, `add_*_interaction`.
- MOCK nnInteractive adapter for CI/unit tests, clearly labeled mock.
- REAL nnInteractive smoke test: actual `nnInteractive_v1.0` checkpoint plus one public Nibabel test NIfTI volume; wiring check only.
- Gymnasium env on synthetic/mock data: `reset(image + GT + optional initial seed)`, `step(point+ / point- / STOP)`, delta-Dice reward vs provided GT.
- FP/FN largest-component robot-user baseline on synthetic/mock data.

## Setup

```bash
make setup
```

The setup target creates `.venv`, installs the pinned build backend from `requirements.lock`, and installs the local package in editable mode.

Real nnInteractive dependencies are intentionally separate because they install torch and the inference stack:

```bash
make setup-real
```

The CUDA setup used for the 2026-07-05 local throughput run is pinned in
`requirements.real.txt`: `torch==2.12.1+cu126` and
`torchvision==0.27.1+cu126` from the PyTorch CUDA 12.6 wheel index.

## Tests

```bash
make test
```

The current tests validate the runtime config loader and run without public datasets, checkpoints, DICOM files, or PHI.

## Smoke Check

```bash
make smoke
```

The smoke target validates `configs/rl_nninteractive_skeleton.json` and prints a mock-scaffold summary. Real nnInteractive execution is intentionally blocked until later units add the adapter, checkpoint configuration, public de-identified dataset manifests, seeds, metrics, and saved outputs.

The config seed is recorded for reproducibility but no RNG-consuming code exists yet. Actual RNG seeding is deferred to the environment/training units that introduce stochastic sampling.

The mock adapter is deliberately non-representative: it paints binary voxels directly and does not run a neural forward pass, propagate prompts in 3D, decay prior interactions, expose logits, or measure changed-patch inference timing. It is only for CI/unit tests.

The real nnInteractive contract is currently a pinned stub targeting `nninteractive==2.5.0` (`https://pypi.org/project/nninteractive/2.5.0/`) and model/checkpoint name `nnInteractive_v1.0`. The real adapter smoke-test unit must install nnInteractive, enforce `RL_NNINTERACTIVE_REQUIRE_REAL=1`, inspect the live signatures, and retain the checkpoint license note: CC-BY-NC-SA 4.0, research/non-commercial use only.

## Real nnInteractive Smoke Check

```bash
make setup-real
make real-smoke
```

The real smoke target downloads or reuses `MIC-DKFZ/nnInteractive/nnInteractive_v1.0`, loads Nibabel's bundled public `anatomical.nii` test fixture, initializes a real `nnInteractiveInferenceSession`, applies one positive center point, and writes ignored outputs under `artifacts/rl_nninteractive/real_smoke/`.

Current local run result after the CUDA wheel install: `torch 2.12.1+cu126`,
CUDA available on `cuda:0`, image shape `[1, 33, 41, 25]`, point
`[16, 20, 12]`, changed bbox `[[0, 33], [0, 41], [0, 25]]`, mask sum
`126`. This is only a wiring smoke test, not a tumor benchmark or clinical
result.

## Gymnasium Env

`RlNnInteractiveEnv` is the current synthetic/mock RL environment. It requires `reset(options={"image": ..., "ground_truth": ...})`; standard wrappers that call option-less reset are unsupported in this phase. The image shape is `(1, z, y, x)` and the GT shape is `(z, y, x)`. An optional `initial_point` applies a positive seed during reset unless `initial_include=False` is passed for a negative seed. Actions are dictionaries with `action_type` (`STOP`, `POINT_POSITIVE`, `POINT_NEGATIVE`) and integer `coord` in `(z, y, x)` order. Point actions receive immediate Dice improvement vs GT as reward; `STOP` terminates with zero reward. Empty GT is allowed and follows Dice's empty-score convention, so adding a false-positive voxel produces negative reward. This is not a final training reward or evaluation harness.

## Robot-User Baseline

`largest_component_robot_action(current_mask, ground_truth)` is the deterministic FP/FN baseline for later comparisons. It uses GT during simulation/evaluation only: compute false negatives and false positives, find the largest connected component across both error masks, prefer a false-negative component on equal size, and click the component voxel nearest its centroid. False negatives emit `POINT_POSITIVE`, false positives emit `POINT_NEGATIVE`, and exact matches emit `STOP`. `run_largest_component_robot_user(...)` rolls that policy through `RlNnInteractiveEnv` and records point-interaction Dice values. This is currently validated on synthetic/mock masks only; real public cases and NoC harness wiring are separate Phase 0 items.

## Baseline Verification

```bash
make verify-baseline
make setup-real
make verify-baseline-public
```

`make verify-baseline` runs three synthetic tumor-mask cases through the mock Gymnasium env and writes `artifacts/rl_nninteractive/baseline_verification/synthetic/summary.json`. `make verify-baseline-public` also loads Nibabel's bundled public `anatomical.nii` volume and uses a tiny synthetic center GT mask to verify real public/de-identified image I/O. The public-image case is still a wiring verification only; it is not a tumor benchmark, clinical result, or real GT evaluation.

## NoC Evaluation Harness

`evaluate_interaction_trajectory(...)` is the current NoC/Dice summary helper. It consumes point-interaction Dice trajectories, delegates NoC@85, NoC@90, and Dice@{1,3,5} to the metric library, and returns JSON-friendly summaries. Baseline verification summaries include these fields under `evaluation`, so later real evaluation can reuse the same output schema instead of recomputing NoC separately.

## Remote Throughput Harness

```bash
make setup-real
NNINTERACTIVE_SERVER_URL=http://127.0.0.1:1527 make throughput-remote
```

`throughput-remote` is gated and expects an already-running `nninteractive-server`; it does not start Docker or claim GPU use by itself. The current local environment has an RTX 4080 visible to `nvidia-smi`, but the Python venv has CPU-only torch and Docker is not running, so the Phase 0 throughput checkbox is not complete until a CUDA-enabled server is available and an env-steps/sec result is logged.

Current local throughput artifact: the RTX 4080 remote server measured
`4.4877479714064235` point-interaction steps/sec on the public Nibabel fixture
shape `[1, 33, 41, 25]` with one server session. That timer excludes
`set_image`, target-buffer reads, metric/candidate generation, and real tumor
volume size. Treat it as a wiring lower bound only. Before any paid rental or
large run, rerun with a public/de-identified real-sized volume and
`THROUGHPUT_PARALLEL_SESSIONS=N` to validate actual concurrent sessions. The
throughput target preflights that `THROUGHPUT_PARALLEL_SESSIONS` does not
exceed `BLACKWELL_MAX_SESSIONS`; start the server with
`--max-sessions >= N` before requesting `N` concurrent sessions.

The versioned handoff target records the remaining GPU-gated commands and estimates:

```bash
make blackwell-handoff
```

`blackwell-handoff` writes `blackwell_handoff.json` and `blackwell_handoff.md` under `artifacts/rl_nninteractive/blackwell_handoff/`. It is a runbook generator only; it does not claim throughput, training, evaluation, or ablation results.

The manifest-driven point-policy large-run launcher is:

```bash
BLACKWELL_DATASET_MANIFEST=manifests/blackwell_datasets.json make phase1-real
```

`phase1-real` requires an already-running remote server and a JSON manifest of public/de-identified cases with `train` and `val` splits. It refuses to run without `--require-remote`; use `python -m rl_nninteractive.phase1_real --dry-run-manifest --dataset-manifest <manifest>` to validate manifest shape without touching a server. The current runner is gated plumbing-smoke for the point-only path, not the final training design. The default online fine-tune budget is deliberately small (`BLACKWELL_DQN_EPISODES=256`) and guarded by `BLACKWELL_MAX_REMOTE_ENV_STEPS=10000`; larger runs require an explicit `--allow-large-run` after real-sized-volume and parallel-session throughput are measured, and the reopened reward/STOP/GT-free candidate work must land before treating it as a go/no-go experiment.

## Metric Spacing

HD95 and NSD are distance metrics. Any real evaluation must pass voxel spacing from the source image header or manifest; the default unit spacing is only valid for synthetic tests with known isotropic voxels.
