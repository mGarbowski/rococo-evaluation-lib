from typing import Protocol
import numpy as np


class FaceRecognitionException(Exception):
    """Base exception for face recognition errors."""

    pass


class NoFaceDetectedException(FaceRecognitionException):
    """Exception raised when no face is detected in an image."""

    pass


class FaceRecognitionSystem(Protocol):
    def feature_vector_length(self) -> int:
        """Returns the length of the feature vector used by the system."""
        ...

    def compute_feature_vector(self, image: np.ndarray) -> np.ndarray:
        """Computes the feature vector for a given image.

        May use a face detection model to find faces in the image.

        Raises:
            NoFaceDetectedException: If no face is detected in the image.

        Args:
            image (np.ndarray): The input image.

        Returns:
            np.ndarray: The computed feature vector.
        """
        ...
