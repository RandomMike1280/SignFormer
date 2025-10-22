import enum
import os
import mediapipe as mp
from pathlib import Path


class ModelNotFoundError(FileNotFoundError):
    pass


def resolve_model(relative_path: str) -> str:
    """sigma resolve model function"""
    base_dir = Path(mp.__file__).parent
    full_path = base_dir / relative_path
    if not full_path.exists():
        raise ModelNotFoundError(f"model file not found: {full_path}")
    return str(full_path)


class ModelEnum(enum.Enum):
    POSE_LANDMARK_FULL = resolve_model("modules/pose_landmark/pose_landmark_full.tflite")
    # POSE_LANDMARK_LITE = resolve_model("modules/pose_landmark/pose_landmark_lite.tflite")
    # POSE_LANDMARK_HEAVY = resolve_model("modules/pose_landmark/pose_landmark_heavy.tflite")
    FACE_LANDMARK = resolve_model("modules/face_landmark/face_landmark.tflite")
    HAND_LANDMARK_FULL = resolve_model("modules/hand_landmark/hand_landmark_full.tflite")
    HAND_LANDMARK_LITE = resolve_model("modules/hand_landmark/hand_landmark_lite.tflite")
    # HAND_LANDMARK_SPARSE = resolve_model("modules/hand_landmark/hand_landmark_sparse.tflite")
    POSE_DETECTION = resolve_model("modules/pose_detection/pose_detection.tflite")
    PALM_DETECTION_FULL = resolve_model("modules/palm_detection/palm_detection_full.tflite")
    PALM_DETECTION_LITE = resolve_model("modules/palm_detection/palm_detection_lite.tflite")


r"modules\face_landmark\face_landmark.tflite"