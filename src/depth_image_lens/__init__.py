"""Static depth shape heuristics for deformable objects."""
from .depth import background_difference, depth_pair_16uc1_to_meters, positive_difference_16uc1
from .bridge_drape import BridgeDrapeDetector, BridgeDrapeResult, __version__

__all__ = ["BridgeDrapeDetector", "BridgeDrapeResult", "__version__", "background_difference", "depth_pair_16uc1_to_meters", "positive_difference_16uc1"]

from .geometry import CameraIntrinsics, TablePlane
from .lift import LiftDetector, LiftResult

__all__ += ["CameraIntrinsics", "TablePlane", "LiftDetector", "LiftResult"]

from .plane import fit_table_plane, PlaneFitResult

__all__ += ["fit_table_plane", "PlaneFitResult"]

from .color import SeededColorAssistant, ColorAssistResult

__all__ += ["SeededColorAssistant", "ColorAssistResult"]

from .corner import OverheadCornerDetector, CornerResult

__all__ += ["OverheadCornerDetector", "CornerResult"]

from .rgbd import analyze_color_geometry

__all__ += ["analyze_color_geometry"]

from .grasp import IndependentGraspDetector, GraspResult

__all__ += ["IndependentGraspDetector", "GraspResult"]

from .object_grasp import ObjectOnlyGraspDetector, ObjectOnlyGraspResult

__all__ += ["ObjectOnlyGraspDetector", "ObjectOnlyGraspResult"]

from .reference import ReferenceGripperSegmenter, GripperReference, ReferenceSegmentationResult
from .surface import DepthSurfaceDetector, SurfaceResult

__all__ += ["ReferenceGripperSegmenter", "GripperReference", "ReferenceSegmentationResult", "DepthSurfaceDetector", "SurfaceResult"]

from .boundary import OverheadBoundaryDetector, BoundaryResult

__all__ += ["OverheadBoundaryDetector", "BoundaryResult"]

from .outline import OuterContourDetector, OuterContourResult

__all__ += ["OuterContourDetector", "OuterContourResult"]

from .boundary_proximity import BoundaryProximityResult

__all__ += ["BoundaryProximityResult"]
