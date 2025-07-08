import numpy as np
from rococo_evaluation_lib import RococoValidation


class MockFaceRecognitionSystem:

    def feature_vector_length(self) -> int:
        return 128

    def compute_feature_vector(self, image: np.ndarray) -> np.ndarray:
        return np.random.rand(128)


def test_stores_parameter():

    validator = RococoValidation(MockFaceRecognitionSystem())
    assert validator.embedding_size == 128

    validator.validate()
