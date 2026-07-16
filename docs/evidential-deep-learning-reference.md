# Evidential Deep Learning for 3D Segmentation Uncertainty — Methods Reference

Reference for `rl_nninteractive/evidential.py`. Formulas and citations verified
2026-07-14 against primary sources.

Notation: `K` = number of classes; `y` one-hot label; `⊙` elementwise product;
`ψ(·)` digamma; `Γ(·)` gamma.

## 1. Sensoy–Kaplan–Kandemir EDL (NeurIPS 2018)

Replace the softmax head with a non-negative activation producing per-class
**evidence** `e_k ≥ 0` (we use **softplus**, not ReLU — see §3), which
parameterizes a Dirichlet:

```
e_k = softplus(logit_k)                 evidence ≥ 0
α_k = e_k + 1                           Dirichlet parameters (+1 = uniform base prior)
S   = Σ_k α_k = K + Σ_k e_k             Dirichlet strength
p̂_k = α_k / S                          predicted probability (Dirichlet mean)
b_k = e_k / S = (α_k − 1)/S             belief mass for class k
u   = K / S                            vacuity / epistemic uncertainty,  u + Σ_k b_k = 1
```

**Loss — Bayes risk of the sum-of-squares (Sensoy Eq. 5, the recommended form):**

```
L_SS = Σ_k [ (y_k − p̂_k)^2 + p̂_k (1 − p̂_k)/(S + 1) ]
```

decomposes into an error term + a variance term minimized by accumulating
evidence (raising `S`). (This is distinct from the "Type-II MLE" form, Eq. 4,
`Σ_k y_k (log S − log α_k)`; Eq. 5 is the best-behaved and what we use.)

**KL regularizer toward uniform on the *misleading* evidence** (drives wrong-class
evidence to zero):

```
α̃ = y + (1 − y) ⊙ α
KL[ Dir(α̃) ‖ Dir(1) ] = log( Γ(Σ_k α̃_k) / (Γ(K) Π_k Γ(α̃_k)) )
                          + Σ_k (α̃_k − 1)[ ψ(α̃_k) − ψ(Σ_j α̃_j) ]
```

**Total, annealed** so the model accumulates correct evidence before the KL flattens
misleading evidence:

```
L = mean_voxels( L_SS + λ_t · KL ),   λ_t = min(1, epoch / 10)
```

Our implementation additionally applies inverse-frequency **class weights** per
voxel to counter the ~99% dominance of the `correct` class (the segmentation
literature alternatively adds a soft-Dice term; either counters imbalance).

## 2. Dense (per-voxel) EDL and uncertainty decomposition

Put the Dirichlet head on every voxel (K output channels → softplus), average the
loss over voxels. Uncertainty fields:

- **Vacuity** `u(v) = K/S(v)` — epistemic, high where total evidence is low
  (ambiguous / OOD regions).
- Finer split of total predictive uncertainty:
  `Total = H(p̂)`, `Aleatoric = E_{Dir(α)}[H(p)]`, `Epistemic = Total − Aleatoric`
  (mutual information). Vacuity and the MI term correlate but are not identical.

Empirical note from our synthetic run: the **expected error probability**
`p_error = p̂_FN + p̂_FP` is a near-perfect voxel-error localizer (AUROC ≈ 1.0),
whereas **vacuity does not localize errors** — the model concentrates evidence
(low vacuity) at the errors it recognizes, so vacuity instead flags genuinely
ambiguous regions. Candidate selection therefore uses `p_error`; vacuity is kept
as a separate "defer-to-human / ambiguity" signal.

## 3. Known failure modes & corrections

- **Evidence collapse (ReLU dead zones).** Prefer softplus/exp evidence (Pandey &
  Yu, ICML 2023) so every voxel keeps accumulating evidence. *(Adopted.)*
- **KL over-regularization → overconfidence / erased evidence magnitude.** If
  vacuity maps come out overconfident, use R-EDL (Chen et al., ICLR 2024):
  tunable prior `α = e + λ` and dropping the variance-minimizing KL for a
  direct-expectation objective. *(Noted for a v2 ablation.)*
- **Uninformative equal weighting / weak few-shot uncertainty.** I-EDL /
  Fisher-information EDL (Deng et al., **ICML 2023**). *(Only if rare-class
  uncertainty is unreliable.)*

## 4. Calibration metrics (the honesty check)

- **AUROC(uncertainty → voxel error)** — treat per-voxel uncertainty as a
  classifier of "voxel is wrong"; higher = uncertainty ranks errors above
  correct voxels. *(Logged every epoch.)*
- **ECE of the Dirichlet mean** — bin by `max_k p̂_k`, compare confidence to
  empirical accuracy. *(Logged every epoch.)*
- **Sparsification / AUSE** and **risk–coverage / AURC** — remove most-uncertain
  voxels and check error drops toward the oracle curve. *(Planned for the
  write-up.)*

## Selected references

- Sensoy, Kaplan, Kandemir. *Evidential Deep Learning to Quantify Classification Uncertainty.* NeurIPS 2018. arXiv:1806.01768.
- Han, Zhang, Fu, Zhou. *Trusted Multi-View Classification.* ICLR 2021 (TPAMI 2022 ext.). — Dempster–Shafer evidential fusion.
- Zou et al. *TBraTS: Trusted Brain Tumor Segmentation.* MICCAI 2022. arXiv:2206.09309. — first clean per-voxel EDL segmentation head.
- Zou et al. *DEviS: Reliable Medical Image Segmentation via Evidential Calibrated Uncertainty.* IEEE Trans. Cybernetics 2025. arXiv:2301.00349.
- *Uncertainty–Error correlations in EDL for biomedical segmentation.* 2024. arXiv:2410.18461. — EDL uncertainty correlates with error better than MC-Dropout/entropy.
- *Evidential Calibrated Uncertainty-Guided Interactive Segmentation for Ultrasound.* 2025. arXiv:2501.01072. — closest analog: vacuity guides the next interactive correction.
- Chen et al. *Think Twice Before Selection: Federated Evidential Active Learning …* MICCAI 2024. arXiv:2312.02567. — evidential uncertainty as an acquisition function.
- Deng et al. *I-EDL (Fisher-Information EDL).* ICML 2023. arXiv:2303.02045.
- Chen et al. *R-EDL: Relaxing Nonessential Settings of EDL.* ICLR 2024. arXiv:2410.00393.
- Pandey & Yu. *Learn to Accumulate Evidence from All Training Samples.* ICML 2023. arXiv:2306.11113.
