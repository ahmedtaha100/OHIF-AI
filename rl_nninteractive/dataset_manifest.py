"""Dataset manifest loader for public/de-identified large-run handoff."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .adapter import VoxelCoord, as_voxel_coord
from .medical_geometry import GeometryMetadata, load_nifti_on_reference_grid
from .provenance import require_sha256, sha256_file
from .real_adapter import load_nifti_image


@dataclass(frozen=True)
class ManifestCase:
    name: str
    dataset: str
    split: str
    image: np.ndarray
    ground_truth: np.ndarray
    initial_point: VoxelCoord
    initial_include: bool = True
    patient_id: str = ""
    site: str = ""
    tracer: str = ""
    modalities: tuple[str, ...] = ()
    reference_modality: str = ""
    modality_paths: dict[str, str] | None = None
    source_dataset_version: str = ""
    annotation_version: str = ""
    target_label: str = ""
    preprocessing_hash: str = ""
    inclusion_hash: str = ""
    image_sha256: dict[str, str] | None = None
    ground_truth_sha256: str = ""
    geometry: dict[str, Any] | None = None
    prior_exposure: bool = False

    @property
    def case_id(self) -> str:
        return self.name


@dataclass(frozen=True)
class StudyManifest:
    version: int
    split_seed: int | None
    split_provenance: str
    preprocessing_hash: str
    manifest_sha256: str
    cases: tuple[ManifestCase, ...]


def load_manifest_cases(manifest_path: Path, *, split: str | None = None) -> tuple[ManifestCase, ...]:
    """Load public/de-identified array cases from a JSON manifest.

    Manifest shape:

    ```json
    {
      "version": 1,
      "datasets": [{
        "name": "dataset_name",
        "cases": [{
          "case_id": "case001",
          "split": "train",
          "image": "relative-or-absolute.npy",
          "ground_truth": "relative-or-absolute.npy",
          "initial_point": [10, 20, 30]
        }]
      }]
    }
    ```

    `.npy`, one-array `.npz`, and 3D NIfTI paths are supported. The loader does
    not download data and does not permit DICOM inputs.
    """

    study = load_study_manifest(manifest_path)
    if split is None:
        return study.cases
    return tuple(case for case in study.cases if case.split == split)


def load_study_manifest(manifest_path: Path) -> StudyManifest:
    """Load a legacy v1 or scientifically hardened v2 StudyManifest.

    Version 2 binds ordered modalities, patient grouping, split seed and
    provenance, dataset/annotation versions, geometry, preprocessing/inclusion
    hashes, prior exposure, and source-file digests. It also rejects patient
    leakage across train, policy-validation, calibration, and test cohorts.
    Version 1 remains readable for historical smoke fixtures but does not
    satisfy the scientific-hardening gate.
    """

    manifest_path = Path(manifest_path)
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    version = payload.get("version")
    if version not in (1, 2):
        raise ValueError("dataset manifest version must be 1 or 2")
    base_dir = manifest_path.parent
    datasets = payload.get("datasets")
    if not isinstance(datasets, list) or not datasets:
        raise ValueError("dataset manifest requires non-empty datasets")

    if version == 2:
        split_seed = payload.get("split_seed")
        if not isinstance(split_seed, int) or isinstance(split_seed, bool):
            raise ValueError("v2 StudyManifest requires integer split_seed")
        preprocessing_hash = require_sha256(
            _required_str(payload, "preprocessing_hash"), field="preprocessing_hash"
        )
        split_provenance = _required_str(payload, "split_provenance")
    else:
        split_seed = None
        split_provenance = "legacy-unspecified"
        preprocessing_hash = ""

    cases: list[ManifestCase] = []
    case_ids: set[str] = set()
    patient_splits: dict[str, str] = {}
    for dataset in datasets:
        dataset_name = _required_str(dataset, "name")
        if version == 2:
            dataset_version = _required_str(dataset, "version")
            annotation_version = _required_str(dataset, "annotation_version")
            modalities = _required_str_sequence(dataset, "modalities")
            reference_modality = _required_str(dataset, "reference_modality")
            if reference_modality not in modalities:
                raise ValueError("reference_modality must appear in modalities")
        else:
            dataset_version = "legacy-unspecified"
            annotation_version = "legacy-unspecified"
            modalities = ("image",)
            reference_modality = "image"
        raw_cases = dataset.get("cases")
        if not isinstance(raw_cases, list) or not raw_cases:
            raise ValueError(f"dataset {dataset_name} requires non-empty cases")
        for raw_case in raw_cases:
            case_id = _required_str(raw_case, "case_id")
            if case_id in case_ids:
                raise ValueError(f"duplicate case_id: {case_id}")
            case_ids.add(case_id)
            case_split = _required_str(raw_case, "split")
            if version == 2 and case_split not in {
                "train",
                "policy_validation",
                "calibration",
                "test",
            }:
                raise ValueError(f"invalid v2 StudyManifest split: {case_split}")

            if version == 2:
                patient_id = _required_str(raw_case, "patient_id")
                site = _required_str(raw_case, "site")
                tracer = _required_str(raw_case, "tracer")
                target_label = _required_str(raw_case, "target_label")
                prior_exposure = raw_case.get("prior_exposure")
                if not isinstance(prior_exposure, bool):
                    raise ValueError("v2 StudyManifest case requires boolean prior_exposure")
                if (
                    split_provenance == "prospectively_frozen_untouched"
                    and case_split == "test"
                    and prior_exposure
                ):
                    raise ValueError("an untouched test case cannot have prior_exposure=true")
                inclusion_hash = require_sha256(
                    _required_str(raw_case, "inclusion_hash"), field="inclusion_hash"
                )
                raw_image_hashes = _required_str_mapping(raw_case, "image_sha256")
                if set(raw_image_hashes) != set(modalities):
                    raise ValueError("image_sha256 keys must exactly match dataset modalities")
                image_sha256 = {
                    modality: require_sha256(
                        raw_image_hashes[modality], field=f"image_sha256.{modality}"
                    )
                    for modality in modalities
                }
                ground_truth_sha256 = require_sha256(
                    _required_str(raw_case, "ground_truth_sha256"),
                    field="ground_truth_sha256",
                )
                geometry = _required_geometry(raw_case, modalities=modalities)
                prior_split = patient_splits.setdefault(patient_id, case_split)
                if prior_split != case_split:
                    raise ValueError(
                        f"patient {patient_id} appears in multiple splits: "
                        f"{prior_split}, {case_split}"
                    )
            else:
                patient_id = case_id
                site = "legacy-unspecified"
                tracer = "legacy-unspecified"
                target_label = "legacy-binary"
                prior_exposure = False
                inclusion_hash = ""
                image_sha256 = {}
                ground_truth_sha256 = ""
                geometry = None

            ground_truth_path = _resolve(base_dir, _required_str(raw_case, "ground_truth"))
            if version == 2:
                raw_images = _required_str_mapping(raw_case, "images")
                if set(raw_images) != set(modalities):
                    raise ValueError("images keys must exactly match dataset modalities")
                modality_paths = {
                    modality: _resolve(base_dir, raw_images[modality]) for modality in modalities
                }
                for modality, image_path in modality_paths.items():
                    _verify_file_digest(
                        image_path,
                        image_sha256[modality],
                        case_id=case_id,
                        role=modality,
                    )
                _verify_file_digest(
                    ground_truth_path,
                    ground_truth_sha256,
                    case_id=case_id,
                    role="ground_truth",
                )
                nifti_flags = {_is_nifti(path) for path in modality_paths.values()}
                nifti_flags.add(_is_nifti(ground_truth_path))
                if len(nifti_flags) != 1:
                    raise ValueError(
                        f"case {case_id} cannot mix NIfTI and array inputs in a v2 StudyManifest"
                    )
            else:
                image_path = _resolve(base_dir, _required_str(raw_case, "image"))
                modality_paths = {"image": image_path}

            if version == 2 and _is_nifti(ground_truth_path):
                reference_path = modality_paths[reference_modality]
                aligned_modalities = []
                for modality in modalities:
                    image_aligned = load_nifti_on_reference_grid(
                        modality_paths[modality],
                        reference_path=reference_path,
                        channel_index=0,
                        reference_channel_index=0,
                    )
                    _verify_geometry_metadata(
                        image_aligned.geometry,
                        geometry[modality],
                        case_id=case_id,
                        role=modality,
                    )
                    aligned_modalities.append(image_aligned.data_zyx)
                ground_truth_aligned = load_nifti_on_reference_grid(
                    ground_truth_path,
                    reference_path=reference_path,
                    is_label=True,
                    reference_channel_index=0,
                )
                _verify_geometry_metadata(
                    ground_truth_aligned.geometry,
                    geometry["ground_truth"],
                    case_id=case_id,
                    role="ground_truth",
                )
                image = _as_image4d(np.stack(aligned_modalities, axis=0))
                ground_truth = _as_binary_volume(
                    ground_truth_aligned.data_zyx,
                    name="ground_truth",
                )
            elif version == 2:
                raw_modality_arrays = [
                    np.asarray(_load_array(modality_paths[modality])) for modality in modalities
                ]
                if any(array.ndim != 3 for array in raw_modality_arrays):
                    raise ValueError("each v2 array modality must be a 3D volume")
                if any(array.shape != raw_modality_arrays[0].shape for array in raw_modality_arrays[1:]):
                    raise ValueError("v2 array modalities must share one voxel grid")
                image = _as_image4d(np.stack(raw_modality_arrays, axis=0))
                ground_truth = _as_binary_volume(
                    _load_array(ground_truth_path),
                    name="ground_truth",
                )
            else:
                image = _as_image4d(_load_array(image_path))
                ground_truth = _as_binary_volume(
                    _load_array(ground_truth_path),
                    name="ground_truth",
                )
            if image.shape[1:] != ground_truth.shape:
                raise ValueError(
                    f"case {case_id} image shape "
                    f"{image.shape[1:]} != ground_truth shape {ground_truth.shape}"
                )
            point = (
                as_voxel_coord(raw_case["initial_point"])
                if "initial_point" in raw_case
                else _default_initial_point(ground_truth)
            )
            if any(point[axis] >= ground_truth.shape[axis] for axis in range(3)):
                raise ValueError("initial_point must be inside ground_truth shape")
            cases.append(
                ManifestCase(
                    name=case_id,
                    dataset=dataset_name,
                    split=case_split,
                    image=image,
                    ground_truth=ground_truth,
                    initial_point=point,
                    initial_include=bool(raw_case.get("initial_include", True)),
                    patient_id=patient_id,
                    site=site,
                    tracer=tracer,
                    modalities=modalities,
                    reference_modality=reference_modality,
                    modality_paths={
                        modality: str(modality_paths[modality]) for modality in modalities
                    },
                    source_dataset_version=dataset_version,
                    annotation_version=annotation_version,
                    target_label=target_label,
                    preprocessing_hash=preprocessing_hash,
                    inclusion_hash=inclusion_hash,
                    image_sha256=image_sha256,
                    ground_truth_sha256=ground_truth_sha256,
                    geometry=geometry,
                    prior_exposure=prior_exposure,
                )
            )
    return StudyManifest(
        version=version,
        split_seed=split_seed,
        split_provenance=split_provenance,
        preprocessing_hash=preprocessing_hash,
        manifest_sha256=sha256_file(manifest_path),
        cases=tuple(cases),
    )


def _required_str(payload: dict[str, Any], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"manifest field {key} must be a non-empty string")
    return value


def _required_str_sequence(payload: dict[str, Any], key: str) -> tuple[str, ...]:
    values = payload.get(key)
    if not isinstance(values, list) or not values or any(not isinstance(value, str) or not value for value in values):
        raise ValueError(f"manifest field {key} must be a non-empty string list")
    if len(set(values)) != len(values):
        raise ValueError(f"manifest field {key} must not contain duplicates")
    return tuple(values)


def _required_str_mapping(payload: dict[str, Any], key: str) -> dict[str, str]:
    values = payload.get(key)
    if not isinstance(values, dict) or not values:
        raise ValueError(f"manifest field {key} must be a non-empty object")
    if any(
        not isinstance(map_key, str)
        or not map_key
        or not isinstance(value, str)
        or not value
        for map_key, value in values.items()
    ):
        raise ValueError(f"manifest field {key} must map non-empty strings to non-empty strings")
    return dict(values)


def _required_geometry(
    raw_case: dict[str, Any], *, modalities: tuple[str, ...]
) -> dict[str, Any]:
    geometry = raw_case.get("geometry")
    if not isinstance(geometry, dict):
        raise ValueError("v2 StudyManifest case requires geometry")
    expected_roles = set(modalities) | {"ground_truth"}
    if set(geometry) != expected_roles:
        raise ValueError("geometry keys must exactly match modalities plus ground_truth")
    for role in (*modalities, "ground_truth"):
        record = geometry.get(role)
        if not isinstance(record, dict):
            raise ValueError(f"geometry.{role} must be an object")
        affine = np.asarray(record.get("affine"), dtype=np.float64)
        spacing = np.asarray(record.get("spacing"), dtype=np.float64)
        orientation = record.get("orientation")
        if affine.shape != (4, 4) or not bool(np.isfinite(affine).all()):
            raise ValueError(f"geometry.{role}.affine must be a finite 4x4 matrix")
        if spacing.shape != (3,) or not bool(np.isfinite(spacing).all()) or bool((spacing <= 0).any()):
            raise ValueError(f"geometry.{role}.spacing must contain three positive values")
        if not _valid_orientation(orientation):
            raise ValueError(f"geometry.{role}.orientation must be a three-letter axis code")
    return geometry


def _valid_orientation(value: Any) -> bool:
    if not isinstance(value, str) or len(value) != 3:
        return False
    return all(sum(letter in group for letter in value) == 1 for group in ("LR", "PA", "IS"))


def _verify_file_digest(path: Path, expected: str, *, case_id: str, role: str) -> None:
    observed = sha256_file(path)
    if observed != expected:
        raise ValueError(
            f"case {case_id} {role} SHA-256 mismatch: expected {expected}, observed {observed}"
        )


def _verify_geometry_metadata(
    observed: GeometryMetadata,
    expected: dict[str, Any],
    *,
    case_id: str,
    role: str,
) -> None:
    expected_affine = np.asarray(expected["affine"], dtype=np.float64)
    observed_affine = np.asarray(observed.source_affine_xyz, dtype=np.float64)
    expected_spacing = np.asarray(expected["spacing"], dtype=np.float64)
    observed_spacing = np.asarray(observed.source_spacing_xyz, dtype=np.float64)
    if not np.allclose(observed_affine, expected_affine, rtol=0.0, atol=1e-5):
        raise ValueError(f"case {case_id} {role} affine does not match StudyManifest")
    if not np.allclose(observed_spacing, expected_spacing, rtol=0.0, atol=1e-5):
        raise ValueError(f"case {case_id} {role} spacing does not match StudyManifest")
    if observed.source_orientation != expected["orientation"]:
        raise ValueError(f"case {case_id} {role} orientation does not match StudyManifest")


def _resolve(base_dir: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else base_dir / path


def _is_nifti(path: Path) -> bool:
    suffixes = "".join(path.suffixes).lower()
    return suffixes.endswith(".nii") or suffixes.endswith(".nii.gz")


def _load_array(path: Path) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(path)
    suffixes = "".join(path.suffixes).lower()
    if suffixes.endswith(".nii") or suffixes.endswith(".nii.gz"):
        return load_nifti_image(path)
    if path.suffix.lower() == ".npy":
        return np.load(path, allow_pickle=False)
    if path.suffix.lower() == ".npz":
        loaded = np.load(path, allow_pickle=False)
        try:
            keys = list(loaded.keys())
            if len(keys) != 1:
                raise ValueError(f"npz input must contain exactly one array: {path}")
            return np.asarray(loaded[keys[0]])
        finally:
            loaded.close()
    if suffixes.endswith(".dcm") or suffixes.endswith(".dicom"):
        raise ValueError("DICOM inputs are not allowed in the public handoff manifest")
    raise ValueError(f"unsupported array path: {path}")


def _as_image4d(image: np.ndarray) -> np.ndarray:
    array = np.asarray(image, dtype=np.float32)
    if array.ndim == 3:
        array = array[None]
    if array.ndim != 4 or array.shape[0] < 1:
        raise ValueError(f"image must be 3D or shape (c, z, y, x) with c >= 1, got {array.shape}")
    if not bool(np.isfinite(array).all()):
        raise ValueError("image contains non-finite values")
    return array.copy()


def _as_binary_volume(mask: np.ndarray, *, name: str) -> np.ndarray:
    array = np.asarray(mask)
    if array.ndim != 3:
        raise ValueError(f"{name} must be a 3D volume")
    if array.dtype == np.bool_:
        return array.astype(bool, copy=True)
    if not np.issubdtype(array.dtype, np.number):
        raise ValueError(f"{name} must be boolean or binary numeric")
    if not bool(np.isfinite(array).all()):
        raise ValueError(f"{name} contains non-finite values")
    if not bool(np.isin(array, (0, 1)).all()):
        raise ValueError(f"{name} numeric values must be in {{0, 1}}")
    return array.astype(bool, copy=True)


def _default_initial_point(ground_truth: np.ndarray) -> VoxelCoord:
    coords = np.argwhere(ground_truth)
    if coords.size == 0:
        raise ValueError("initial_point is required when ground_truth is empty")
    centroid = coords.mean(axis=0)
    distances = np.square(coords - centroid).sum(axis=1)
    order = np.lexsort((coords[:, 2], coords[:, 1], coords[:, 0], distances))
    chosen = coords[int(order[0])]
    return (int(chosen[0]), int(chosen[1]), int(chosen[2]))
