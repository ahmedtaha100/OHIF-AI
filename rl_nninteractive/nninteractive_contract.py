"""Pinned nnInteractive API contract used by the local adapter Protocol.

The real package is intentionally not installed for the scaffold tests because
it pulls the heavyweight inference stack. These expected signatures are a pinned
stub for `nninteractive==2.5.0`; tests also check the real class when available.
"""

from __future__ import annotations

NNINTERACTIVE_REQUIREMENT = "nninteractive==2.5.0"
NNINTERACTIVE_SOURCE_URL = "https://pypi.org/project/nninteractive/2.5.0/"
NNINTERACTIVE_REQUIREMENT_SOURCE = (
    "unverified pinned-stub target for scaffold contract tests. Source URL: "
    f"{NNINTERACTIVE_SOURCE_URL}. The real package is not installed in this "
    "scaffold venv, so this remains unverified until the real adapter smoke-test "
    "unit installs and inspects nnInteractive."
)
NNINTERACTIVE_MODEL_NAME = "nnInteractive_v1.0"
NNINTERACTIVE_MODEL_SOURCE = (
    "Observed in monai-label/monailabel/tasks/infer/basic_infer.py as MODEL_NAME. "
    "Checkpoint/model availability must be revalidated in the real smoke-test unit."
)
NNINTERACTIVE_CHECKPOINT_LICENSE = "CC-BY-NC-SA 4.0; research/non-commercial use only"

EXPECTED_NNINTERACTIVE_METHOD_PARAMS = {
    # Positional argument names are intentionally omitted unless basic_infer.py
    # calls them by keyword. This avoids over-asserting names from an uninstalled
    # external package.
    "set_image": (),
    "set_target_buffer": (),
    "add_point_interaction": ("include_interaction",),
    "add_bbox_interaction": ("include_interaction",),
    "add_scribble_interaction": (
        "scribble_image",
        "include_interaction",
        "interaction_bbox",
    ),
    "add_lasso_interaction": (
        "include_interaction",
        "interaction_bbox",
    ),
    "reset_interactions": (),
}
