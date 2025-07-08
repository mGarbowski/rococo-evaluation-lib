"""Rococo Evaluation Library

A library for face recognition system validation based on the Rococo2 dataset.
"""

from .rococo_validation import RococoValidation
from .face_recognition_system import (
    FaceRecognitionSystem,
    FaceRecognitionException,
    NoFaceDetectedException,
)


__version__ = "0.1.0"
__all__ = [
    "RococoValidation",
    "FaceRecognitionSystem",
    "FaceRecognitionException",
    "NoFaceDetectedException",
]
