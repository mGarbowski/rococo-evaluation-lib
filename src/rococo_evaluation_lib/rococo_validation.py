from dataclasses import dataclass
from enum import Enum
from typing import List, Union

import cv2
import numpy as np
from rococo_evaluation_lib.face_recognition_system import FaceRecognitionException, FaceRecognitionSystem
from rococo_evaluation_lib.rococo_dataset import Frame, RococoDataset
from rococo_evaluation_lib.sequence_match_algorithm import sequence_match


@dataclass(frozen=True)
class SingleThresholdResult:
    threshold: float
    n_true_positives: int
    n_false_accepts: int
    n_false_rejects: int
    n_total: int

    @property
    def far(self) -> float:
        if self.n_total == 0:
            return 0.0
        return self.n_false_accepts / self.n_total

    @property
    def frr(self) -> float:
        if self.n_total == 0:
            return 0.0
        return self.n_false_rejects / self.n_total


@dataclass
class ValidationResults:
    threshold: List[float]
    far: List[float]
    frr: List[float]


class MatchResult(Enum):
    TRUE_POSITIVE = "True Positive"
    FALSE_ACCEPT = "False Accept"
    FALSE_REJECT = "False Reject"


EmbeddingVec = np.ndarray
SimilarityVec = np.ndarray


class RococoValidation:
    """Validation procedure for a face recognition system.

    Based on the Rococo2 dataset.
    """

    face_recognition_system: FaceRecognitionSystem
    dataset: RococoDataset
    embedding_size: int
    min_sequence_length: int
    face_embeddings: Union[np.ndarray, None] = None
    frame_seq_similarities: Union[List[List[Union[SimilarityVec, None]]], None] = None

    def __init__(
        self,
        face_recognition_system: FaceRecognitionSystem,
        dataset: RococoDataset,
        min_sequence_length: int = 3,
    ):
        self.face_recognition_system = face_recognition_system
        self.dataset = dataset
        self.embedding_size = face_recognition_system.feature_vector_length()
        self.min_sequence_length = min_sequence_length

    def similarity_score(
        self, embedding_a: EmbeddingVec, embedding_b: EmbeddingVec
    ) -> float:
        """Compute similarity score between two embedding vectors.

        Score is a number in range [0, 1].
        Closer to 1 means more similar.
        Closer to 0 means less similar.
        Score is computed as the cosine similarity scaled to [0, 1].
        """
        norm_a = np.linalg.norm(embedding_a)
        norm_b = np.linalg.norm(embedding_b)

        if norm_a == 0 or norm_b == 0:
            return 0.0

        return np.dot(embedding_a, embedding_b) / (norm_a * norm_b)

    def _compute_embedding_for_file(self, filename: str) -> Union[EmbeddingVec, None]:
        """Compute embedding for a given file using face recognition system.

        Returns None if the recognition system fails (e.g. no face is detected).
        """
        image = cv2.imread(filename, cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Could not read image from {filename}")

        try:
            embedding = self.face_recognition_system.compute_feature_vector(image)
            assert embedding.shape == (
                self.embedding_size,
            ), f"Unexpected embedding shape: {embedding.shape}, expected: ({self.embedding_size},)"
            return embedding
        except FaceRecognitionException as e:
            print(f"WARNING Failed to compute embedding for {filename}: {e}")
            return None

    def _compute_face_embeddings(self):
        """Compute embeddings for all faces in the dataset.

        Embeddings are stored in `self.face_embeddings` for further comparisons.
        `face_embeddings` is a matrix of shape (n_all_faces, embedding_size).
        """
        if self.face_embeddings is not None:
            return

        self.face_embeddings = np.zeros(
            (self.dataset.n_all_faces, self.embedding_size), dtype=np.float32
        )

        for i, face in enumerate(self.dataset.all_faces):
            print(f"Processing face {i + 1} of {self.dataset.n_all_faces}")
            face_path = self.dataset.face_abspath(face)
            embedding = self._compute_embedding_for_file(face_path)
            assert (
                embedding is not None
            ), f"Failed to compute embedding for face {face_path}"
            self.face_embeddings[i] = embedding

    def _compute_similarity_vector_for_frame(
        self, frame: Frame
    ) -> Union[SimilarityVec, None]:
        """Compute similarity vector for a given frame.

        Similarity vector is a vector of similarity scores of frame to all faces.
        Return None if the embedding cannot be computed (e.g. frame does not contain a face).
        """

        # Ensure embeddings are computed
        if self.face_embeddings is None:
            self._compute_face_embeddings()
        assert self.face_embeddings is not None

        frame_path = self.dataset.frame_abspath(frame)
        frame_embedding = self._compute_embedding_for_file(frame_path)

        if frame_embedding is None:
            return None

        similarity_vec = np.zeros(self.dataset.n_all_faces, dtype=np.float32)
        for idx in range(self.dataset.n_all_faces):
            similarity_vec[idx] = self.similarity_score(
                frame_embedding, self.face_embeddings[idx]
            )

        return similarity_vec

    def _compute_all_sequence_similarities(self):
        """Compute similarity vectors for all frames of all sequences in the dataset.

        Similarity vectors are stored in `self.frame_seq_similarities`.
        `frame_seq_similarities` is a list with element for each sequence.
        Each sequence corresponds to a list of similarity vectors for each frame in that sequence (or None if the frame does not contain a face).
        """
        if self.frame_seq_similarities is not None:
            return

        self.frame_seq_similarities = []

        for idx, (_, frame_seq) in enumerate(self.dataset.elements):
            print(
                f"Processing frame sequence {idx + 1} of {len(self.dataset.elements)}"
            )
            sequence_similarities = [
                self._compute_similarity_vector_for_frame(frame) for frame in frame_seq
            ]
            self.frame_seq_similarities.append(sequence_similarities)

    @staticmethod
    def _get_most_similar_face_indices(
        sequence_similarities: List[Union[SimilarityVec, None]],
        similarity_threshold: float,
    ) -> List[Union[int, None]]:
        """Transform a list of similarity vectors for a frame sequence into a list of most similar face indices for each frame.

        Each element in the list corresponds to a frame in the sequence.
        Element is None if the frame does not contain a face at all or if the similarity score is below the threshold.
        """
        return [
            (
                int(np.argmax(similarity_vector))
                if similarity_vector is not None
                and np.max(similarity_vector) > similarity_threshold
                else None
            )
            for similarity_vector in sequence_similarities
        ]

    @staticmethod
    def _compute_match_result_for_sequence(
        sequence_similarities: List[Union[SimilarityVec, None]],
        expected_face_idx: int,
        similarity_threshold: float,
        min_sequence_length: int,
    ) -> MatchResult:

        most_similar_face_idxs = RococoValidation._get_most_similar_face_indices(
            sequence_similarities, similarity_threshold
        )

        matched_idx = sequence_match(most_similar_face_idxs, min_sequence_length)

        print(
            f"Matched index: {matched_idx}, Expected index: {expected_face_idx}, Threshold: {similarity_threshold:.2f}, most similar indices: {most_similar_face_idxs}"
        )

        if matched_idx is None:
            return MatchResult.FALSE_REJECT
        elif matched_idx == expected_face_idx:
            return MatchResult.TRUE_POSITIVE
        else:
            return MatchResult.FALSE_ACCEPT

    def _compute_results_for_threshold(
        self,
        similarity_threshold: float,
        all_seq_similarities: List[List[Union[SimilarityVec, None]]],
        min_sequence_length: int,
    ):
        counts = {
            MatchResult.TRUE_POSITIVE: 0,
            MatchResult.FALSE_ACCEPT: 0,
            MatchResult.FALSE_REJECT: 0,
        }

        for face_idx, seq_similarities in enumerate(all_seq_similarities):
            # For cases when the order of all_faces is not the same as the order of elements in the dataset
            # TODO check if it works correctly!
            expected_face_idx = self.dataset.face_index(
                self.dataset.all_faces[face_idx]
            )
            match_result = self._compute_match_result_for_sequence(
                sequence_similarities=seq_similarities,
                expected_face_idx=expected_face_idx,
                similarity_threshold=similarity_threshold,
                min_sequence_length=min_sequence_length,
            )
            counts[match_result] += 1

        return SingleThresholdResult(
            threshold=similarity_threshold,
            n_true_positives=counts[MatchResult.TRUE_POSITIVE],
            n_false_accepts=counts[MatchResult.FALSE_ACCEPT],
            n_false_rejects=counts[MatchResult.FALSE_REJECT],
            n_total=len(all_seq_similarities),
        )

    def validate(self) -> ValidationResults:
        print("Starting validation...")
        print("Computing face embeddings...")
        self._compute_face_embeddings()
        print("Computing sequence similarities...")
        self._compute_all_sequence_similarities()
        assert self.frame_seq_similarities is not None

        print("Computing validation results for different thresholds...")
        thresholds = list(np.linspace(0.0, 1.00, 101))
        fars = []
        frrs = []

        for threshold in thresholds:
            print(f"Validating for threshold: {threshold:.2f}")
            result = self._compute_results_for_threshold(
                similarity_threshold=threshold,
                all_seq_similarities=self.frame_seq_similarities,
                min_sequence_length=self.min_sequence_length,
            )
            fars.append(result.far)
            frrs.append(result.frr)

        return ValidationResults(threshold=thresholds, far=fars, frr=frrs)
