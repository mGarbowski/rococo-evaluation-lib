"""Rococo Evaluation Library

A library for face recognition system validation based on the Rococo2 dataset.
"""

from .validation_results import ValidationResults
from .rococo_validation import RococoValidation
from .rococo_dataset import RococoDataset, Frame, FrameSequence, Face
from .face_recognition_system import (
    FaceRecognitionSystem,
    FaceRecognitionException,
    NoFaceDetectedException,
)


__version__ = "0.1.1"
__all__ = [
    "RococoValidation",
    "ValidationResults",
    "RococoDataset",
    "Frame",
    "FrameSequence",
    "Face",
    "FaceRecognitionSystem",
    "FaceRecognitionException",
    "NoFaceDetectedException",
]
