"""RL interaction-policy scaffold for frozen nnInteractive sessions."""

from .adapter import (
    AxisRange,
    Box3D,
    ImageArray,
    InteractionResult,
    MaskArray,
    NnInteractiveSession,
    VoxelCoord,
    as_box3d,
    as_voxel_coord,
)
from .config import RuntimeConfig, load_config
from .env import (
    POINT_NEGATIVE,
    POINT_POSITIVE,
    STOP,
    RlNnInteractiveEnv,
)
from .evaluation import (
    DEFAULT_DICE_STEPS,
    InteractionEvaluation,
    evaluate_interaction_trajectory,
    summarize_interaction_evaluations,
)
from .mock_adapter import MockNnInteractiveSession
from .nninteractive_contract import (
    EXPECTED_NNINTERACTIVE_METHOD_PARAMS,
    NNINTERACTIVE_CHECKPOINT_LICENSE,
    NNINTERACTIVE_MODEL_NAME,
    NNINTERACTIVE_MODEL_SOURCE,
    NNINTERACTIVE_REQUIREMENT,
    NNINTERACTIVE_REQUIREMENT_SOURCE,
    NNINTERACTIVE_SOURCE_URL,
)
from .metrics import (
    SurfaceDistances,
    dice_at_steps,
    dice_score,
    hd95,
    noc_at_85,
    noc_at_90,
    noc_at_threshold,
    normalized_surface_dice,
    surface_distances,
)
from .robot_user import (
    RobotUserDecision,
    RobotUserEpisode,
    largest_component_robot_action,
    run_largest_component_robot_user,
)

__all__ = [
    "Box3D",
    "AxisRange",
    "ImageArray",
    "InteractionResult",
    "MaskArray",
    "MockNnInteractiveSession",
    "NnInteractiveSession",
    "EXPECTED_NNINTERACTIVE_METHOD_PARAMS",
    "NNINTERACTIVE_CHECKPOINT_LICENSE",
    "NNINTERACTIVE_MODEL_NAME",
    "NNINTERACTIVE_MODEL_SOURCE",
    "NNINTERACTIVE_REQUIREMENT",
    "NNINTERACTIVE_REQUIREMENT_SOURCE",
    "NNINTERACTIVE_SOURCE_URL",
    "RuntimeConfig",
    "RlNnInteractiveEnv",
    "RobotUserDecision",
    "RobotUserEpisode",
    "DEFAULT_DICE_STEPS",
    "InteractionEvaluation",
    "POINT_NEGATIVE",
    "POINT_POSITIVE",
    "STOP",
    "SurfaceDistances",
    "VoxelCoord",
    "as_box3d",
    "as_voxel_coord",
    "dice_at_steps",
    "dice_score",
    "evaluate_interaction_trajectory",
    "hd95",
    "load_config",
    "largest_component_robot_action",
    "noc_at_85",
    "noc_at_90",
    "noc_at_threshold",
    "normalized_surface_dice",
    "run_largest_component_robot_user",
    "summarize_interaction_evaluations",
    "surface_distances",
]
