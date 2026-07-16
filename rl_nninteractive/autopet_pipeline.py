"""AutoPET / DEEP-PSMA whole-body PET/CT pipeline for multifocal lesion recovery.

The lung/pancreas experiments showed the limit of single-mask correction on a
strong segmentator. Whole-body PET/CT is the setting with *safe* headroom: a scan
has many scattered lesions, so one seed prompt captures only one lesion and misses
the rest. Adding a foreground prompt at a MISSED lesion can only raise Dice /
detection -- it is monotone-safe, unlike perturbing one fragile mask.

This module:
  * discovers DEEP-PSMA/AutoPET cases (CT, PET/SUV, TTB lesion mask) robustly by
    filename keyword (the release does not fix exact names);
  * preprocesses a whole-body case to a fixed grid (CT windowed to [-1,1], PET
    SUV log-normalized, lesion mask at PET resolution);
  * provides a multifocal rollout where the evidential model's predicted
    false-negative map is used to place prompts that RECOVER missed lesions.

Intensity conventions (from the DEEP-PSMA data card): PET is SUV (body weight),
CT is HU, masks are TTB (binary total tumour burden at PET resolution).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .medical_geometry import GeometryMetadata, load_nifti_on_reference_grid

try:
    from scipy import ndimage
except Exception as exc:  # pragma: no cover
    raise ImportError("autopet_pipeline requires scipy") from exc


# --------------------------------------------------------------------------- #
# Case discovery
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class AutoPetCase:
    case_id: str
    ct: Path
    pet: Path
    mask: Path
    tracer: str  # "FDG" or "PSMA"


def _classify(name: str) -> str | None:
    n = name.lower()
    if n.startswith("._"):
        return None
    if any(k in n for k in ("totalseg", "totseg", "organ")):  # TotalSegmentator aux labels
        return "aux"
    if "ct" in n and "ttb" not in n:
        return "ct"
    if any(k in n for k in ("suv", "pet")):
        return "pet"
    if any(k in n for k in ("ttb", "tumou", "tumor", "lesion", "label", "seg", "gt", "mask")):
        return "mask"
    return None


def _tracer_of(path: Path) -> str:
    s = str(path).lower()
    if "psma" in s:
        return "PSMA"
    if "fdg" in s:
        return "FDG"
    return "FDG"


def find_autopet_cases(root: str | Path, *, tracer: str = "FDG") -> list[AutoPetCase]:
    """Discover cases under ``root``. Groups NIfTIs by their parent directory and
    classifies CT/PET/mask by filename keyword; keeps cases with all three for the
    requested tracer.
    """

    root = Path(root)
    by_dir: dict[Path, dict[str, list[Path]]] = {}
    for p in root.rglob("*.nii.gz"):
        if p.name.startswith("._"):
            continue
        kind = _classify(p.name)
        if kind in (None, "aux"):
            continue
        d = p.parent
        by_dir.setdefault(d, {}).setdefault(kind, []).append(p)

    cases: list[AutoPetCase] = []
    for d, kinds in sorted(by_dir.items()):
        # DEEP-PSMA lays out cases as train_XXXX/{FDG,PSMA}/{CT,PET,TTB}.nii.gz;
        # keep only the requested tracer's sub-directory.
        if tracer != "ANY" and d.name.upper() != tracer.upper():
            continue
        cts, pets, masks = kinds.get("ct", []), kinds.get("pet", []), kinds.get("mask", [])
        if cts and pets and masks:
            case_id = f"{d.parent.name}_{d.name}"
            cases.append(AutoPetCase(case_id=case_id, ct=sorted(cts)[0], pet=sorted(pets)[0],
                                     mask=sorted(masks)[0], tracer=tracer))
    return cases


# --------------------------------------------------------------------------- #
# Preprocessing
# --------------------------------------------------------------------------- #
def _load_nii(path: str | Path):
    import nibabel as nib
    img = nib.load(str(path))
    return np.asarray(img.dataobj, dtype=np.float32), img


def _resample_to(vol: np.ndarray, target: tuple[int, int, int], *, order: int) -> np.ndarray:
    factors = [t / s for t, s in zip(target, vol.shape)]
    return ndimage.zoom(vol, factors, order=order).astype(np.float32)


def _window_ct(v: np.ndarray, lo: float = -1000.0, hi: float = 400.0) -> np.ndarray:
    v = np.clip(v, lo, hi)
    return ((v - lo) / (hi - lo) * 2.0 - 1.0).astype(np.float32)


def _norm_suv(v: np.ndarray, cap: float = 25.0) -> np.ndarray:
    # SUV is heavy-tailed; log1p then scale to ~[-1,1] with a physiological cap.
    v = np.clip(v, 0.0, cap)
    v = np.log1p(v) / np.log1p(cap)
    return (v * 2.0 - 1.0).astype(np.float32)


@dataclass
class AutoPetVolume:
    case_id: str
    ct: np.ndarray       # (Z,Y,X) [-1,1]
    pet: np.ndarray      # (Z,Y,X) [-1,1] (log-SUV)
    gt: np.ndarray       # (Z,Y,X) bool lesion mask
    n_lesions: int
    geometry: dict[str, GeometryMetadata]


def load_autopet_volume(case: AutoPetCase, *, target: tuple[int, int, int] = (256, 128, 128)) -> AutoPetVolume:
    """Load a case on a PET-referenced physical grid in ``(Z,Y,X)`` order."""

    pet_aligned = load_nifti_on_reference_grid(
        case.pet, reference_path=case.pet, target_shape_zyx=target
    )
    ct_aligned = load_nifti_on_reference_grid(
        case.ct, reference_path=case.pet, target_shape_zyx=target
    )
    mask_aligned = load_nifti_on_reference_grid(
        case.mask, reference_path=case.pet, target_shape_zyx=target, is_label=True
    )
    pet = pet_aligned.data_zyx
    ct = ct_aligned.data_zyx
    gt = mask_aligned.data_zyx > 0.5

    _, n = ndimage.label(gt, structure=np.ones((3, 3, 3), dtype=bool))
    return AutoPetVolume(
        case_id=case.case_id,
        ct=_window_ct(ct),
        pet=_norm_suv(pet),
        gt=gt.astype(bool),
        n_lesions=int(n),
        geometry={
            "pet": pet_aligned.geometry,
            "ct": ct_aligned.geometry,
            "ground_truth": mask_aligned.geometry,
        },
    )


def lesion_components(gt: np.ndarray, *, min_size: int = 5) -> list[np.ndarray]:
    labels, n = ndimage.label(gt, structure=np.ones((3, 3, 3), dtype=bool))
    out = []
    for i in range(1, n + 1):
        comp = labels == i
        if comp.sum() >= min_size:
            out.append(comp)
    return out


def _representative_coord(mask: np.ndarray) -> tuple[int, int, int]:
    coords = np.argwhere(mask)
    c = coords.mean(axis=0)
    d = np.square(coords - c).sum(axis=1)
    order = np.lexsort((coords[:, 2], coords[:, 1], coords[:, 0], d))
    v = coords[int(order[0])]
    return (int(v[0]), int(v[1]), int(v[2]))
