# PI screen-recording guide

Use this page as the repository map during the live presentation. Navigate by symbol name, not line number, so the recording remains valid if formatting changes.

## Before recording

Use a clean checkout of public `main`. From the repository root, run:

```powershell
python scripts\present_pi_walkthrough.py --check-readiness
```

The command must end with `PI PRESENTATION READINESS: PASS`. It parses every Python file and verifies every file, function, class, and heading named below without importing the scientific stack or touching research data.

Preview the ordered file tour:

```powershell
python scripts\present_pi_walkthrough.py --section files --mode auto --delay 0 --no-color --width 112
```

Start the recording walkthrough:

```powershell
python scripts\present_pi_walkthrough.py --mode interactive --width 104
```

## Presentation order

Open with the paper evidence, then move into the technical file tour:

```powershell
python scripts\present_pi_walkthrough.py --section paper --section release --section code --section statistics --mode interactive --width 108
python scripts\present_pi_walkthrough.py --section files --section pipeline --section policy --section compute --section decision --mode interactive --width 108
```

The first command establishes why the public paper package is not reproducible. The second command explains what was actually implemented and tested in this repository.

## Exact live file order

### 1. Presenter map

- File: `docs/pi-presentation-guide.md`
- Show: `Exact live file order`, `Technical explanation`, and `Paper reproduction boundary`.
- Explain: this page is the navigation and speaking map. Keep the Explorer panel visible and use symbol search for every code stop.

### 2. AutoPET nnU-Net input

- File: `scripts/prepare_autopet_nnunet_input.py`
- Symbol: `main`.
- Show: CT, PET, and TTB are resampled to the PET reference grid. CT and PET become image channels and TTB becomes the offline lesion label.
- Explain: AutoPET provides the PET/CT imaging backbone. TTB is used offline for model labels, simulated expert corrections, development rewards, and final scoring. It is not presented as a deployable input.

### 3. Patient split and robot-user proposals

- File: `scripts/run_fusion_only_cohort_v2.py`
- Symbols: `split_for`, `freeze_contract`, `stage_prompt_round`, `run_prompt_no_score`.
- Show: the patient-disjoint 12 train, 4 calibration, 8 policy-validation, and 6 test split; paired FDG/PSMA studies; the AutoPET III ResEnc-L baseline; and two AutoPET V prompt rounds.
- Explain: foreground clicks are placed at deep voxels of the largest false-negative component and background clicks at deep voxels of the largest false-positive component. Clicks accumulate across rounds. Because errors are computed against TTB, the proposals are indirectly ground-truth-dependent and this remains an offline oracle-assisted experiment.

### 4. Safe fusion menu

- File: `scripts/finalize_fusion_only_cohort_v2.py`
- Symbols: `candidate_paths`, `generate_fusions`.
- Show: round-one intersection, round-one union, round-two intersection, and round-two union.
- Explain: KEEP retains the ResEnc-L mask. Direct replacement was removed after severe development harms, so the final selector could only KEEP or combine a prompt mask with the baseline.

### 5. Voxelwise three-class EDL critic

- File: `rl_nninteractive/evidential.py`
- Symbols: `EvidentialErrorNet3D`, `dirichlet_alpha`, `dirichlet_uncertainty`, `error_labels_from_masks`, `predict_error_maps`.
- Show: the network predicts nonnegative evidence for correct, false-negative, and false-positive voxels. For class `k`, `alpha_k = evidence_k + 1`, total strength is `S = sum(alpha)`, expected probability is `alpha_k / S`, and vacuity is `K / S` with `K = 3`.
- Explain: this critic proposes uncertain error regions for the lung and pancreas point-selection experiment. It is not the AutoPET component classifier and not the final route utility head.

### 6. Evidential point candidates

- File: `rl_nninteractive/evidential_candidates.py`
- Symbols: `evidential_candidates_topk`, `evidential_stop_decision`.
- Show: connected components of predicted false-negative and false-positive error become ranked positive or negative point candidates. A greedy evidential baseline can abstain.
- Explain: the coordinates and component statistics become candidate actions for the first RL formulation.

### 7. Lung and pancreas point-selection RL

- File: `rl_nninteractive/rl_policy.py`
- Symbols: `build_action_features`, `RealEdlEnv`, `TrainConfig`, `train`.
- Show: every action has 10 candidate features plus four state features. The candidate fields encode STOP, polarity, component size, predicted error mass, confidence, vacuity, and normalized location. The state fields encode predicted error fraction, largest error region, mean vacuity, and step fraction.
- Explain: the action is one ranked point or an explicit STOP. Reward is `new Dice - previous Dice - 0.01 click cost`, plus an optional uncertainty-resolution bonus whose coefficient was zero in the reported run. Training used 200 behavior-cloning episodes, then 1,200 REINFORCE episodes with a learned value baseline, `gamma = 0.99`, entropy regularization, and at most eight actions.
- Result: RL tied greedy EDL on lung at about 0.768 Dice, then fell below KEEP on pancreas. That showed that a capable critic did not guarantee a safe learned interaction policy.

### 8. AutoPET component classifier and accept-or-skip RL

- File: `rl_nninteractive/autopet_rl_recovery.py`
- Symbols: `extract_candidates`, `EvidentialCandidateClassifier`, `RecoveryPolicy`, `rollout_policy`, `train_rl`.
- Show: PET-hot connected components have 11 direct features: size, SUV statistics, CT statistics, normalized location, automated TotalSegmentator overlap, compactness, and SUV percentile. The binary EDL classifier produces `P(lesion)` and vacuity.
- Explain: each policy state is the 11 component features plus `P(lesion)`, vacuity, candidate-index fraction, and current union fraction. The policy makes one binary accept-or-skip decision per ranked candidate. Accept reward is change in whole-body Dice minus `0.002`; skip reward is zero. Training uses behavior-cloning warm start followed by REINFORCE. There is no separate learned STOP and no value network in this formulation.
- Integrity boundary: PET-hot candidate locations are GT-free, but training labels and rewards use TTB. The legacy 16-scan result also initialized each case from the largest TTB lesion. The current public file was later hardened to an evidential seed, so do not present it as the exact source of the legacy `0.350` artifact.

### 9. Final 70-feature route EDL head

- File: `rl_nninteractive/prompt_update_edl.py`
- Symbols: `EvidentialUtilityHead`, `extract_update_features`, `evidential_utility_loss`, `calibrate_temperature`, `decide_update`.
- Show: 70 direct features summarize PET, CT, current and proposed masks, added and removed regions, morphology, automated anatomical context, and prompt metadata. The head is a 48-hidden-unit MLP with binary Dirichlet evidence and a separate signed utility output.
- Explain: it trained for 300 CPU epochs on development data. Patient-disjoint calibration temperature-scales only `P(accept)`. Vacuity and signed utility are not separately calibrated. The frozen gate requires `P(accept) >= 0.5`, vacuity `<= 0.6`, predicted utility `>= 0`, and an actual mask change. Failure returns KEEP.
- Integrity boundary: no ground-truth array or score is a direct inference feature. TotalSegmentator overlap is automated anatomy, not tumor truth. The prompt coordinates and proposal masks are nevertheless indirectly TTB-derived through the robot user.

### 10. Frozen safety screen plus EDL veto

- File: `rl_nninteractive/edl_fusion_hybrid.py`
- Symbols: `fit_safe_rule_set`, `train_edl`, `edl_gate`, `nested_development_replay`, `select_frozen_policy_routes`.
- Show: the deterministic consensus and PET-uptake screen, the patient-disjoint EDL fit and calibration, 24-fold leave-one-patient-out replay, and the KEEP fallback.
- Explain: a rule was eligible only with at least four selected development studies, zero harms, positive patient mean, and a positive bootstrap lower bound. The full development fit used 19 patients and the calibration split used 5. The final primary selector is a fixed safety screen followed by the EDL veto. It is not an RL network.

### 11. One-shot sealed evaluation

- File: `scripts/run_edl_hybrid_test_once.py`
- Symbols: `validate_non_test_preflight`, `_score_both_frozen_policies_once`, `execute_once`.
- Show: frozen hashes and independent clearance are validated, routes are selected without labels, and only afterward is each test label loaded once to score both frozen policies.
- Explain: six patients and 12 paired studies were opened once. The transaction receipt prevents repeated test tuning or different scoring passes for the two frozen selectors.

### 12. Patient-level statistics

- File: `rl_nninteractive/route_policy_eval.py`
- Symbols: `_patient_rows`, `_policy_summary`, `_bootstrap_ci`.
- Show: FDG and PSMA studies are aggregated within patient, followed by coverage, harm, win/tie/loss, and 10,000-sample paired patient bootstrap summaries.
- Explain: patients, not scans, are the unit of uncertainty.

### 13. Paper reproduction audit

- File: `scripts/present_pi_walkthrough.py`
- Symbols: `_paper_claims`, `_release_forensics`, `_manuscript_code`, `_statistics`.
- Show: missing final policy weights, missing split manifests and paper-matching evaluator, the `0 / 154` U-Net key match, partial PPO restoration, manuscript-code differences, and statistical bounds.
- Explain: the public package is insufficient to reproduce or verify the headline result. This is a serious reproducibility discrepancy, but it is not evidence of fabrication or author intent.

## Technical explanation

### The full AutoPET pipeline

AutoPET contributed paired whole-body FDG and PSMA PET/CT studies and TTB lesion annotations. The AutoPET III ResEnc-L model produced the automatic baseline. The AutoPET V correction model consumed four channels: CT, PET, foreground-click heatmap, and background-click heatmap. A deterministic robot user generated cumulative positive and negative corrections from TTB error regions. We fused each prompt result with ResEnc-L using intersection or union and asked a frozen selector whether any proposal was safer than KEEP.

The experiment therefore answers a narrow offline question: can GT-free selector features identify when an oracle-assisted proposal improves the baseline? It does not test a deployable autonomous click generator.

### Three different EDL heads

1. The voxel error critic is a three-class Dirichlet model over correct, false-negative, and false-positive voxels. It generated uncertainty-guided point candidates in lung and pancreas.
2. The AutoPET recovery classifier is a binary Dirichlet model over non-lesion versus lesion for PET-hot components. Its `P(lesion)` and vacuity entered an accept-or-skip RL state.
3. The final route utility head is a binary Dirichlet help/harm model plus signed utility regression over 70 proposal features. It could veto a route proposed by a deterministic screen.

All three use nonnegative evidence and Dirichlet strength, but they operate at different spatial levels and were trained for different decisions.

### Two different RL policies

1. The lung and pancreas policy selected an evidential point candidate or STOP. It used behavior cloning, REINFORCE, and a learned value baseline.
2. The AutoPET component policy accepted or skipped each ranked PET-hot component. It used behavior-cloning warm start and REINFORCE without a value baseline or a separate STOP action.

Neither RL policy was the final sealed selector. Their failures motivated the conservative action menu, abstention, and KEEP fallback.

### What was actually sealed

The primary sealed policy was a deterministic consensus and PET-uptake safety screen followed by the 70-feature EDL veto. The secondary policy was the deterministic screen alone. Both were frozen before the six-patient test was opened. Calling the sealed primary an RL network would be technically wrong.

### The clearest failure example

`train_0025_PSMA` selected the round-two union route. The EDL head reported `P(accept) = 0.7377`, model vacuity `0.03146`, and predicted utility `+0.4182`. Actual Dice fell from `0.54172` to `0.50327`, a change of `-0.03845`. The model looked confident and predicted benefit, yet the action was harmful. This is direct evidence of a calibration and generalization failure on the selected case.

The sealed cohort result was baseline Dice `0.608227`, primary hybrid Dice `0.605023`, delta `-0.003204`, patient-bootstrap 95% CI `[-0.009612, 0]`, and one harmful action in 12 studies. The correct decision is KEEP.

## Paper reproduction boundary

Present the paper audit in this order:

1. State the exact headline claims.
2. Show what a reproducible package requires: final weights, exact cohort identities and splits, paper-matching preprocessing and evaluation, per-case outputs, and statistics code.
3. Show that those materials are missing or incompatible.
4. Show the `0 / 154` documented U-Net key match and the missing PPO heads.
5. Show the manuscript-to-code and statistical discrepancies.
6. End with the bounded conclusion: the headline result cannot be independently verified from the public release. Do not infer misconduct from missing evidence alone.

For the paper-focused sequence, run:

```powershell
python scripts\present_pi_walkthrough.py --section paper --section release --section code --section statistics --section decision --mode interactive --width 108
```
