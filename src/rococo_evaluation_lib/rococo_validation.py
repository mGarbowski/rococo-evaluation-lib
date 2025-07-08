from rococo_evaluation_lib.face_recognition_system import FaceRecognitionSystem


class RococoValidation:
    """Validation procedure for a face recognition system.

    Based on the Rococo2 dataset.
    """

    def __init__(self, face_recognition_system: FaceRecognitionSystem):
        self.face_recognition_system: FaceRecognitionSystem = face_recognition_system
        self.embedding_size: int = face_recognition_system.feature_vector_length()

    def validate(self):
        print("Validating...")
