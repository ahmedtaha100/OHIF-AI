"""Evidential deep learning (EDL) for GT-free segmentation-error prediction.

This module trains a small 3D network to answer, *without ground truth at
inference time*, the question that the RL prompt policy needs at every step:

    "Given the image and nnInteractive's current mask, where is the mask most
     likely wrong, is the error a missed tumor (false negative) or leakage
     (false positive), and how confident is the model in that judgement?"

The network emits a per-voxel Dirichlet distribution over three *error*
classes -- {correct, false-negative, false-positive} -- following the
evidential framework of Sensoy, Kaplan & Kandemir, "Evidential Deep Learning to
Quantify Classification Uncertainty" (NeurIPS 2018), applied densely per voxel.

Why this matters for the project (see ``docs`` / Progress.md): the existing
``robot_user.largest_component_robot_action`` and ``multitool.multi_tool_candidates``
locate errors by comparing the mask to the ground-truth mask. That is an
*oracle* -- unavailable when a clinician is actually annotating. The Dirichlet
error map produced here is the ground-truth-free replacement that makes the
next-best-interaction recommender deployable, and its Dempster-Shafer vacuity
gives a calibrated, GT-free stopping signal.

Everything here depends only on ``torch`` and ``numpy``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

try:  # torch is a real runtime dependency for this module, but keep imports lazy-friendly.
    import torch
    from torch import nn
    from torch.nn import functional as F
except Exception as exc:  # pragma: no cover - exercised only when torch is absent
    raise ImportError(
        "rl_nninteractive.evidential requires PyTorch. Install the CUDA build "
        "used for training (torch>=2.2)."
    ) from exc


# Per-voxel error classes. The label at a voxel is a pure function of
# (current_mask, ground_truth):
#   correct           : current_mask == ground_truth
#   false_negative    : ground_truth == 1 and current_mask == 0  (missed tumor)
#   false_positive    : ground_truth == 0 and current_mask == 1  (leakage)
ERR_CORRECT = 0
ERR_FALSE_NEGATIVE = 1
ERR_FALSE_POSITIVE = 2
NUM_ERROR_CLASSES = 3
ERROR_CLASS_NAMES = ("correct", "false_negative", "false_positive")


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #
def _group_norm(channels: int) -> nn.GroupNorm:
    groups = 8 if channels % 8 == 0 else (4 if channels % 4 == 0 else 1)
    return nn.GroupNorm(groups, channels)


class _ConvBlock(nn.Module):
    """(Conv3d -> GroupNorm -> LeakyReLU) x 2, padding-preserving."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            _group_norm(out_channels),
            nn.LeakyReLU(0.01, inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            _group_norm(out_channels),
            nn.LeakyReLU(0.01, inplace=True),
        )

    def forward(self, x: "torch.Tensor") -> "torch.Tensor":
        return self.block(x)


class EvidentialErrorNet3D(nn.Module):
    """Lightweight 3D U-Net that outputs Dirichlet evidence per voxel.

    Input:  (B, in_channels, D, H, W). Default 2 channels = [image, current_mask].
    Output: (B, NUM_ERROR_CLASSES, D, H, W) of non-negative *evidence* e_k >= 0.
            The Dirichlet parameters are alpha_k = e_k + 1.

    The encoder/decoder use two 2x poolings, so spatial dims are internally
    padded to a multiple of 4 and cropped back, making the model safe on
    arbitrary input sizes.
    """

    def __init__(self, in_channels: int = 2, base_channels: int = 16, num_classes: int = NUM_ERROR_CLASSES) -> None:
        super().__init__()
        self.in_channels = int(in_channels)
        self.num_classes = int(num_classes)
        c = int(base_channels)
        self.enc1 = _ConvBlock(self.in_channels, c)
        self.enc2 = _ConvBlock(c, 2 * c)
        self.pool = nn.MaxPool3d(2)
        self.bottleneck = _ConvBlock(2 * c, 4 * c)
        self.up2 = nn.ConvTranspose3d(4 * c, 2 * c, kernel_size=2, stride=2)
        self.dec2 = _ConvBlock(4 * c, 2 * c)
        self.up1 = nn.ConvTranspose3d(2 * c, c, kernel_size=2, stride=2)
        self.dec1 = _ConvBlock(2 * c, c)
        self.head = nn.Conv3d(c, self.num_classes, kernel_size=1)

    @staticmethod
    def _pad_to_multiple(x: "torch.Tensor", multiple: int = 4) -> tuple["torch.Tensor", tuple[int, int, int]]:
        d, h, w = x.shape[-3:]
        pd = (multiple - d % multiple) % multiple
        ph = (multiple - h % multiple) % multiple
        pw = (multiple - w % multiple) % multiple
        if pd or ph or pw:
            # F.pad order is (W_left, W_right, H_left, H_right, D_left, D_right)
            x = F.pad(x, (0, pw, 0, ph, 0, pd))
        return x, (d, h, w)

    def forward(self, x: "torch.Tensor") -> "torch.Tensor":
        x, (d, h, w) = self._pad_to_multiple(x, 4)
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        b = self.bottleneck(self.pool(e2))
        d2 = self.dec2(torch.cat([self.up2(b), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        logits = self.head(d1)
        logits = logits[..., :d, :h, :w]
        # softplus keeps evidence non-negative and gradients stable vs exp/relu.
        evidence = F.softplus(logits)
        return evidence


def dirichlet_alpha(evidence: "torch.Tensor") -> "torch.Tensor":
    """alpha = evidence + 1 (a uniform Dirichlet prior of concentration 1)."""

    return evidence + 1.0


# --------------------------------------------------------------------------- #
# Uncertainty decomposition (Dempster-Shafer / subjective logic)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ErrorMaps:
    """Numpy views of the evidential prediction for one volume.

    All arrays share the input spatial shape (D, H, W), float32 in [0, 1].
    """

    prob: np.ndarray            # (K, D, H, W) expected class probabilities p_k = alpha_k / S
    p_error: np.ndarray         # (D, H, W)   p_fn + p_fp  -> P(voxel currently wrong)
    p_false_negative: np.ndarray
    p_false_positive: np.ndarray
    vacuity: np.ndarray         # (D, H, W)   u = K / S  -> epistemic (lack-of-evidence) uncertainty
    strength: np.ndarray        # (D, H, W)   S = sum_k alpha_k (total evidence + K)


def dirichlet_uncertainty(alpha: "torch.Tensor") -> dict[str, "torch.Tensor"]:
    """Return probability, belief, vacuity, and error-probability tensors.

    ``alpha`` has shape (B, K, ...). Vacuity ``u = K / S`` is the subjective-logic
    epistemic uncertainty: it is high wherever the network has produced little
    total evidence, which is exactly where an interactive correction (or a human
    check) is most warranted.
    """

    strength = alpha.sum(dim=1, keepdim=True)                 # S, shape (B,1,...)
    prob = alpha / strength                                   # p_k
    belief = (alpha - 1.0) / strength                         # b_k = e_k / S
    vacuity = NUM_ERROR_CLASSES / strength                    # u = K / S in (0, 1]
    p_error = prob[:, ERR_FALSE_NEGATIVE] + prob[:, ERR_FALSE_POSITIVE]
    return {
        "prob": prob,
        "belief": belief,
        "vacuity": vacuity.squeeze(1),
        "strength": strength.squeeze(1),
        "p_error": p_error,
        "p_false_negative": prob[:, ERR_FALSE_NEGATIVE],
        "p_false_positive": prob[:, ERR_FALSE_POSITIVE],
    }


# --------------------------------------------------------------------------- #
# Labels + loss
# --------------------------------------------------------------------------- #
def error_labels_from_masks(current_mask: Any, ground_truth: Any) -> np.ndarray:
    """Per-voxel error class in {0,1,2} from (current_mask, ground_truth).

    This uses ground truth and is therefore a *training-only* function; it never
    runs at deployment. Returned dtype is int64 for use as a class-index target.
    """

    current = np.asarray(current_mask).astype(bool)
    target = np.asarray(ground_truth).astype(bool)
    if current.shape != target.shape:
        raise ValueError(f"mask shapes differ: {current.shape} != {target.shape}")
    labels = np.zeros(current.shape, dtype=np.int64)
    labels[np.logical_and(target, ~current)] = ERR_FALSE_NEGATIVE
    labels[np.logical_and(~target, current)] = ERR_FALSE_POSITIVE
    return labels


def _kl_to_uniform_dirichlet(alpha: "torch.Tensor") -> "torch.Tensor":
    """KL( Dir(alpha) || Dir(1,...,1) ), reduced over the class dim.

    Closed form with beta = ones (Sensoy 2018, eq. for the regularizer).
    ``alpha`` shape (B, K, ...), returns (B, 1, ...).
    """

    k = alpha.shape[1]
    strength = alpha.sum(dim=1, keepdim=True)
    ones = torch.ones_like(alpha)
    sum_lgamma_alpha = torch.lgamma(alpha).sum(dim=1, keepdim=True)
    lgamma_sum_alpha = torch.lgamma(strength)
    lgamma_k = torch.lgamma(torch.tensor(float(k), device=alpha.device, dtype=alpha.dtype))
    term_gamma = lgamma_sum_alpha - lgamma_k - sum_lgamma_alpha
    digamma_diff = torch.digamma(alpha) - torch.digamma(strength)
    term_digamma = ((alpha - ones) * digamma_diff).sum(dim=1, keepdim=True)
    return term_gamma + term_digamma


def evidential_segmentation_loss(
    evidence: "torch.Tensor",
    target_labels: "torch.Tensor",
    *,
    epoch: int,
    anneal_epochs: int = 10,
    class_weights: "torch.Tensor | None" = None,
) -> dict[str, "torch.Tensor"]:
    """Sensoy Type-II MLE (expected sum-of-squares) + annealed KL regularizer.

    Args:
        evidence: (B, K, D, H, W) non-negative network evidence.
        target_labels: (B, D, H, W) int64 class indices in [0, K).
        epoch: current epoch (0-based) used for KL annealing.
        anneal_epochs: KL weight ramps linearly to 1 over this many epochs.
        class_weights: optional (K,) tensor; each voxel's loss is scaled by the
            weight of its true class. Used to counter the extreme dominance of
            the ``correct`` class in a mostly-right mask.

    Returns dict with ``loss`` (scalar), ``data`` (sum-of-squares term) and
    ``kl`` (regularizer term) for logging.
    """

    alpha = dirichlet_alpha(evidence)
    k = alpha.shape[1]
    strength = alpha.sum(dim=1, keepdim=True)
    prob = alpha / strength
    target = F.one_hot(target_labels, num_classes=k)          # (B, D, H, W, K)
    target = target.permute(0, 4, 1, 2, 3).to(alpha.dtype)    # (B, K, D, H, W)

    # Bayes risk of the sum-of-squares loss (Sensoy 2018, eq. 5):
    #   E[||y - p||^2] = sum_k (y_k - p_k)^2 + sum_k p_k (1 - p_k) / (S + 1)
    err = (target - prob).pow(2).sum(dim=1, keepdim=True)
    var = (prob * (1.0 - prob) / (strength + 1.0)).sum(dim=1, keepdim=True)
    data_term = err + var                                     # (B, 1, D, H, W)

    # KL regularizer on the *misleading* evidence only:
    #   alpha_tilde = y + (1 - y) * alpha   (evidence for the true class removed)
    alpha_tilde = target + (1.0 - target) * alpha
    kl_term = _kl_to_uniform_dirichlet(alpha_tilde)           # (B, 1, D, H, W)
    lam = min(1.0, float(epoch) / float(max(1, anneal_epochs)))

    per_voxel = data_term + lam * kl_term                     # (B, 1, D, H, W)

    if class_weights is not None:
        w = class_weights.to(alpha.device, alpha.dtype)[target_labels]  # (B, D, H, W)
        w = w.unsqueeze(1)
        loss = (per_voxel * w).sum() / w.sum().clamp_min(1e-8)
    else:
        loss = per_voxel.mean()

    return {
        "loss": loss,
        "data": data_term.mean().detach(),
        "kl": kl_term.mean().detach(),
        "lambda": torch.tensor(lam),
    }


def inverse_frequency_class_weights(
    target_labels: "torch.Tensor",
    *,
    num_classes: int = NUM_ERROR_CLASSES,
    clamp_max: float = 200.0,
) -> "torch.Tensor":
    """Inverse-frequency class weights (normalized to mean 1) for a label batch."""

    counts = torch.bincount(target_labels.reshape(-1), minlength=num_classes).float()
    counts = counts.clamp_min(1.0)
    inv = counts.sum() / (num_classes * counts)
    inv = inv.clamp_max(clamp_max)
    return inv / inv.mean().clamp_min(1e-8)


# --------------------------------------------------------------------------- #
# Inference helpers
# --------------------------------------------------------------------------- #
@torch.no_grad()
def predict_error_maps(
    model: EvidentialErrorNet3D,
    image: Any,
    current_mask: Any,
    *,
    device: "torch.device | str | None" = None,
) -> ErrorMaps:
    """Run the model on one (image, current_mask) volume and return numpy maps.

    ``image`` is (D, H, W) or (1, D, H, W); ``current_mask`` is (D, H, W). No
    ground truth is used -- this is the deployment-time path.
    """

    model.eval()
    dev = torch.device(device) if device is not None else next(model.parameters()).device
    img = np.asarray(image, dtype=np.float32)
    if img.ndim == 4 and img.shape[0] == 1:
        img = img[0]
    if img.ndim != 3:
        raise ValueError("image must be (D,H,W) or (1,D,H,W)")
    mask = np.asarray(current_mask).astype(np.float32)
    if mask.shape != img.shape:
        raise ValueError(f"current_mask shape {mask.shape} != image {img.shape}")

    x = np.stack([img, mask], axis=0)[None, ...]              # (1, 2, D, H, W)
    tensor = torch.from_numpy(x).to(dev)
    evidence = model(tensor)
    alpha = dirichlet_alpha(evidence)
    u = dirichlet_uncertainty(alpha)
    prob = u["prob"][0].detach().cpu().numpy().astype(np.float32)
    return ErrorMaps(
        prob=prob,
        p_error=u["p_error"][0].detach().cpu().numpy().astype(np.float32),
        p_false_negative=u["p_false_negative"][0].detach().cpu().numpy().astype(np.float32),
        p_false_positive=u["p_false_positive"][0].detach().cpu().numpy().astype(np.float32),
        vacuity=u["vacuity"][0].detach().cpu().numpy().astype(np.float32),
        strength=u["strength"][0].detach().cpu().numpy().astype(np.float32),
    )


def set_seed(seed: int) -> None:
    """Seed numpy + torch (CPU and CUDA) for reproducible runs."""

    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
