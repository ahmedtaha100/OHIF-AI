"""Datasets for training the evidential error model.

Two sources, one sample schema. Each sample is a mid-trajectory interactive
state -- an image, the *current* (imperfect) segmentation mask, and the
per-voxel error label -- so the evidential model learns to predict where a
partially-corrected mask is still wrong.

``SyntheticEvidentialDataset``
    Deterministic 3D tumor phantoms. Ellipsoidal lesions with an
    image-appearance signal, plus a randomized "partial segmentation" of that
    lesion that injects realistic false-negative (missed) and false-positive
    (leaked) regions. Fully reproducible from a base seed -- used to validate
    the whole pipeline before real data lands.

``MSDEvidentialDataset``
    Real tumor CT from a Medical Segmentation Decathlon task (e.g. Task06_Lung).
    Reads NIfTI image/label pairs, windows + normalizes the CT, extracts the
    tumor label, samples a patch around a lesion, and applies the *same*
    perturbation model to synthesize the current mask.

The "current mask" perturbations are the key modelling choice: they stand in
for the distribution of masks nnInteractive produces mid-trajectory. Later they
can be replaced by real nnInteractive rollout states (that only changes the
input distribution, not the schema).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from scipy import ndimage

try:
    import torch
    from torch.utils.data import Dataset
except Exception as exc:  # pragma: no cover
    raise ImportError("rl_nninteractive.evidential_dataset requires PyTorch") from exc

from .evidential import error_labels_from_masks


# --------------------------------------------------------------------------- #
# Synthetic phantoms
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class SyntheticTumorConfig:
    shape: tuple[int, int, int] = (64, 64, 64)
    min_tumors: int = 1
    max_tumors: int = 2
    min_radius: float = 5.0
    max_radius: float = 12.0
    tumor_contrast: float = 1.4      # mean intensity offset of tumor vs background
    boundary_blur: float = 1.2       # gaussian sigma applied to the intensity lesion
    noise_std: float = 0.45          # additive gaussian image noise
    background_lowfreq: float = 0.6  # amplitude of smooth background intensity drift


def _ellipsoid(shape: tuple[int, int, int], center, radii, rng) -> np.ndarray:
    zz, yy, xx = np.ogrid[: shape[0], : shape[1], : shape[2]]
    # slightly anisotropic, randomly oriented-ish via per-axis radius jitter
    rz, ry, rx = radii
    val = (
        ((zz - center[0]) / max(rz, 1e-3)) ** 2
        + ((yy - center[1]) / max(ry, 1e-3)) ** 2
        + ((xx - center[2]) / max(rx, 1e-3)) ** 2
    )
    return val <= 1.0


def make_synthetic_case(seed: int, cfg: SyntheticTumorConfig = SyntheticTumorConfig()):
    """Return (image (D,H,W) float32, gt (D,H,W) bool) for a phantom case."""

    rng = np.random.default_rng(seed)
    shape = cfg.shape
    gt = np.zeros(shape, dtype=bool)
    n = int(rng.integers(cfg.min_tumors, cfg.max_tumors + 1))
    margin = int(cfg.max_radius) + 2
    for _ in range(n):
        center = [int(rng.integers(margin, shape[a] - margin)) for a in range(3)]
        radii = [float(rng.uniform(cfg.min_radius, cfg.max_radius)) * float(rng.uniform(0.75, 1.25)) for _ in range(3)]
        gt |= _ellipsoid(shape, center, radii, rng)

    # Image: smooth low-frequency background drift + tumor contrast + noise.
    background = rng.standard_normal(shape).astype(np.float32)
    background = ndimage.gaussian_filter(background, sigma=6.0) * cfg.background_lowfreq
    lesion = ndimage.gaussian_filter(gt.astype(np.float32), sigma=cfg.boundary_blur)
    lesion = lesion / max(float(lesion.max()), 1e-6)
    image = background + cfg.tumor_contrast * lesion
    image = image + rng.standard_normal(shape).astype(np.float32) * cfg.noise_std
    # normalize to zero-mean unit-ish std
    image = (image - image.mean()) / (image.std() + 1e-6)
    return image.astype(np.float32), gt


# --------------------------------------------------------------------------- #
# Current-mask perturbation model (shared by synthetic + real)
# --------------------------------------------------------------------------- #
def _binary(x: np.ndarray) -> np.ndarray:
    return np.asarray(x).astype(bool)


def perturb_to_current(seed: int, gt: np.ndarray, *, progress: float | None = None) -> np.ndarray:
    """Synthesize an imperfect 'current mask' from ground truth.

    ``progress`` in [0,1] controls how close the current mask is to gt (0 = very
    partial/early, 1 = nearly done). If None it is sampled uniformly. The result
    is guaranteed to contain at least one error voxel (else the sample carries no
    learning signal), unless gt is empty.
    """

    rng = np.random.default_rng(seed)
    gt = _binary(gt)
    if not gt.any():
        # No lesion: current mask is empty or a small spurious FP blob.
        current = np.zeros_like(gt)
        if rng.random() < 0.5:
            c = [int(rng.integers(0, s)) for s in gt.shape]
            current |= _ellipsoid(gt.shape, c, [rng.uniform(2, 4)] * 3, rng)
        return current

    p = float(rng.uniform(0.0, 1.0)) if progress is None else float(np.clip(progress, 0.0, 1.0))
    current = gt.copy()
    struct = ndimage.generate_binary_structure(3, 1)

    # 1) Under-segmentation (false negatives): erode, stronger when early.
    erode_iters = int(round((1.0 - p) * rng.integers(1, 4)))
    if erode_iters > 0:
        current = ndimage.binary_erosion(current, structure=struct, iterations=erode_iters)

    # 2) Drop a slab of the lesion (a large contiguous FN), more likely early.
    if rng.random() < (0.5 * (1.0 - p) + 0.1):
        axis = int(rng.integers(0, 3))
        coords = np.argwhere(gt)
        lo, hi = coords[:, axis].min(), coords[:, axis].max()
        if hi > lo:
            cut = int(rng.integers(lo, hi + 1))
            keep_upper = bool(rng.random() < 0.5)
            slab = np.zeros_like(gt)
            idx = [slice(None)] * 3
            idx[axis] = slice(cut, None) if keep_upper else slice(None, cut)
            slab[tuple(idx)] = True
            current = np.logical_and(current, ~slab) if rng.random() < 0.5 else np.logical_and(current, slab | ~gt)

    # 3) Leakage (false positives): dilate then keep only a random fraction of the new rim.
    if rng.random() < (0.4 * p + 0.3):
        dil_iters = int(rng.integers(1, 3))
        dilated = ndimage.binary_dilation(gt, structure=struct, iterations=dil_iters)
        rim = np.logical_and(dilated, ~gt)
        keep = rng.random(size=gt.shape) < rng.uniform(0.2, 0.8)
        current = np.logical_or(current, np.logical_and(rim, keep))

    # 4) Spurious FP component near the lesion (disconnected leak / wrong structure).
    if rng.random() < 0.35:
        coords = np.argwhere(gt)
        anchor = coords[int(rng.integers(0, len(coords)))]
        offset = rng.integers(-14, 15, size=3)
        c = np.clip(anchor + offset, 1, np.array(gt.shape) - 2)
        blob = _ellipsoid(gt.shape, c.tolist(), [rng.uniform(2, 5)] * 3, rng)
        current = np.logical_or(current, np.logical_and(blob, ~gt))

    # Guarantee at least one error voxel so the label is informative.
    if bool(np.array_equal(current, gt)):
        current = ndimage.binary_erosion(gt, structure=struct, iterations=1)
        if bool(np.array_equal(current, gt)):
            # tiny lesion that erosion can't touch: drop a single voxel
            coords = np.argwhere(gt)
            v = coords[int(rng.integers(0, len(coords)))]
            current = gt.copy()
            current[tuple(v)] = False
    return _binary(current)


@dataclass
class EvidentialSample:
    image: np.ndarray        # (D,H,W) float32
    current_mask: np.ndarray  # (D,H,W) bool
    gt: np.ndarray            # (D,H,W) bool
    error_labels: np.ndarray  # (D,H,W) int64 in {0,1,2}
    case_id: str = ""

    def to_tensors(self) -> dict[str, "torch.Tensor"]:
        inp = np.stack([self.image.astype(np.float32), self.current_mask.astype(np.float32)], axis=0)
        return {
            "input": torch.from_numpy(inp),                          # (2,D,H,W)
            "labels": torch.from_numpy(self.error_labels),            # (D,H,W)
            "current_mask": torch.from_numpy(self.current_mask.astype(np.uint8)),
            "gt": torch.from_numpy(self.gt.astype(np.uint8)),
        }


def build_sample(image: np.ndarray, gt: np.ndarray, current: np.ndarray, case_id: str = "") -> EvidentialSample:
    labels = error_labels_from_masks(current, gt)
    return EvidentialSample(
        image=np.asarray(image, dtype=np.float32),
        current_mask=_binary(current),
        gt=_binary(gt),
        error_labels=labels,
        case_id=case_id,
    )


class SyntheticEvidentialDataset(Dataset):
    """Deterministic synthetic dataset; case i is seeded from ``base_seed + i``."""

    def __init__(
        self,
        length: int,
        *,
        base_seed: int = 0,
        cfg: SyntheticTumorConfig = SyntheticTumorConfig(),
    ) -> None:
        self.length = int(length)
        self.base_seed = int(base_seed)
        self.cfg = cfg

    def __len__(self) -> int:
        return self.length

    def make(self, index: int) -> EvidentialSample:
        seed = self.base_seed + int(index)
        image, gt = make_synthetic_case(seed, self.cfg)
        current = perturb_to_current(seed * 7919 + 1, gt)
        return build_sample(image, gt, current, case_id=f"synthetic_{seed}")

    def __getitem__(self, index: int) -> dict[str, "torch.Tensor"]:
        return self.make(index).to_tensors()


# --------------------------------------------------------------------------- #
# Medical Segmentation Decathlon (real CT)
# --------------------------------------------------------------------------- #
def find_msd_cases(root: str | Path) -> list[tuple[Path, Path]]:
    """Return (image, label) NIfTI pairs from an extracted MSD task directory.

    Looks for ``imagesTr/*.nii.gz`` with matching ``labelsTr`` files, skipping
    macOS ``._`` sidecar files.
    """

    root = Path(root)
    images_dir = root / "imagesTr"
    labels_dir = root / "labelsTr"
    if not images_dir.is_dir() or not labels_dir.is_dir():
        raise FileNotFoundError(f"MSD task dir missing imagesTr/labelsTr under {root}")
    pairs: list[tuple[Path, Path]] = []
    for img in sorted(images_dir.glob("*.nii.gz")):
        if img.name.startswith("._"):
            continue
        label = labels_dir / img.name
        if label.exists():
            pairs.append((img, label))
    return pairs


def _window_ct(volume: np.ndarray, lo: float = -1000.0, hi: float = 400.0) -> np.ndarray:
    v = np.clip(volume.astype(np.float32), lo, hi)
    v = (v - lo) / (hi - lo)          # [0,1]
    return (v * 2.0 - 1.0).astype(np.float32)  # [-1,1]


def load_msd_case(image_path: str | Path, label_path: str | Path, *, tumor_label: int = 1):
    """Load one MSD case; return (image float32 windowed, gt bool tumor mask).

    Arrays are returned in (D,H,W) axis order.
    """

    from .medical_geometry import load_nifti_on_reference_grid

    image_aligned = load_nifti_on_reference_grid(
        image_path,
        reference_path=image_path,
        channel_index=0,
        reference_channel_index=0,
    )
    label_aligned = load_nifti_on_reference_grid(
        label_path,
        reference_path=image_path,
        is_label=True,
        reference_channel_index=0,
    )
    image = image_aligned.data_zyx
    gt = label_aligned.data_zyx == tumor_label
    return _window_ct(image), gt.astype(bool)


def _sample_patch_bounds(shape, patch, center, rng) -> tuple[slice, slice, slice]:
    slices = []
    for a in range(3):
        size = min(patch[a], shape[a])
        lo = int(np.clip(center[a] - size // 2, 0, shape[a] - size))
        slices.append(slice(lo, lo + size))
    return tuple(slices)


def _pad_to(array: np.ndarray, target: tuple[int, int, int], *, pad_value) -> np.ndarray:
    """Pad a (D,H,W) array up to ``target`` (never crops), constant pad_value."""

    pads = [(0, max(0, target[a] - array.shape[a])) for a in range(3)]
    if all(hi == 0 for _, hi in pads):
        return array
    return np.pad(array, pads, mode="constant", constant_values=pad_value)


class MSDEvidentialDataset(Dataset):
    """Patch dataset over MSD tumor CT with perturbed current masks.

    Each __getitem__ picks a case (round-robin by index), samples a tumor-centred
    patch when a lesion exists, and synthesizes a current mask via
    ``perturb_to_current``. Deterministic given ``base_seed`` and index.
    """

    def __init__(
        self,
        pairs: Sequence[tuple[Path, Path]],
        *,
        patch: tuple[int, int, int] = (64, 64, 64),
        tumor_label: int = 1,
        base_seed: int = 0,
        samples_per_case: int = 4,
    ) -> None:
        if not pairs:
            raise ValueError("no MSD (image,label) pairs provided")
        self.pairs = list(pairs)
        self.patch = tuple(int(p) for p in patch)
        self.tumor_label = int(tumor_label)
        self.base_seed = int(base_seed)
        self.samples_per_case = int(samples_per_case)
        self._cache: dict[int, tuple[np.ndarray, np.ndarray]] = {}

    def __len__(self) -> int:
        return len(self.pairs) * self.samples_per_case

    def _load(self, case_idx: int) -> tuple[np.ndarray, np.ndarray]:
        if case_idx not in self._cache:
            # bound the cache to avoid unbounded RAM growth on big tasks
            if len(self._cache) > 6:
                self._cache.pop(next(iter(self._cache)))
            img_path, lab_path = self.pairs[case_idx]
            self._cache[case_idx] = load_msd_case(img_path, lab_path, tumor_label=self.tumor_label)
        return self._cache[case_idx]

    def make(self, index: int) -> EvidentialSample:
        case_idx = index % len(self.pairs)
        seed = self.base_seed + index
        rng = np.random.default_rng(seed)
        image, gt = self._load(case_idx)
        if gt.any():
            coords = np.argwhere(gt)
            center = coords[int(rng.integers(0, len(coords)))]
        else:
            center = [int(rng.integers(0, s)) for s in gt.shape]
        bounds = _sample_patch_bounds(gt.shape, self.patch, center, rng)
        # pad up to a fixed patch size so samples batch cleanly (-1 = air in windowed CT).
        img_patch = _pad_to(image[bounds], self.patch, pad_value=-1.0)
        gt_patch = _pad_to(gt[bounds], self.patch, pad_value=False)
        current = perturb_to_current(seed * 7919 + 1, gt_patch)
        return build_sample(img_patch, gt_patch, current, case_id=f"msd_{case_idx}_{index}")

    def __getitem__(self, index: int) -> dict[str, "torch.Tensor"]:
        return self.make(index).to_tensors()


class InMemoryEvidentialDataset(Dataset):
    """Holds pre-built ``EvidentialSample`` objects (fast, no reload thrash)."""

    def __init__(self, samples: Sequence[EvidentialSample]) -> None:
        self.samples = list(samples)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, "torch.Tensor"]:
        return self.samples[index].to_tensors()


def materialize_msd_samples(
    pairs: Sequence[tuple[Path, Path]],
    *,
    patch: tuple[int, int, int] = (64, 64, 64),
    tumor_label: int = 1,
    base_seed: int = 0,
    samples_per_case: int = 8,
) -> list[EvidentialSample]:
    """Extract all patches once, iterating case-by-case to reuse the volume cache."""

    ds = MSDEvidentialDataset(
        pairs, patch=patch, tumor_label=tumor_label, base_seed=base_seed, samples_per_case=samples_per_case
    )
    n_cases = len(pairs)
    samples: list[EvidentialSample] = []
    for case_idx in range(n_cases):
        for j in range(samples_per_case):
            index = j * n_cases + case_idx  # index % n_cases == case_idx -> reuse cached volume
            samples.append(ds.make(index))
    return samples
