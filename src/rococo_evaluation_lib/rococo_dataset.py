# TODO: cleanup this file, field names, compatibility with thesis code etc

import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple


@dataclass
class Face:
    filename: str  # relative
    frame_number_1: int
    face_number: int

    @classmethod
    def from_filename(cls, filename: str) -> 'Face':
        parts = filename.split("_")
        assert len(parts) == 4, f"Unexpected filename format: {filename}"
        assert parts[0] == "frame", f"Unexpected filename format: {filename}"
        assert parts[2] == "face", f"Unexpected filename format: {filename}"
        assert parts[3].endswith(".jpg"), f"Unexpected filename format: {filename}"

        frame_number_1 = int(parts[1])
        face_number = int(parts[3].split(".")[0])
        return cls(filename, frame_number_1, face_number)

    @staticmethod
    def make_filename(frame_number_1: int, face_number: int) -> str:
        return f"frame_{frame_number_1}_face_{face_number}.jpg"


@dataclass
class Frame:
    filename: str  # relative
    frame_number: int
    face_number: int

    @classmethod
    def from_filename(cls, filename: str) -> 'Frame':
        return cls._from_filename_without_sequence(filename)

    @staticmethod
    def make_filename(
        frame_number: int,
        face_number: int,
    ) -> str:
        return (
            f"frame_{frame_number}_with_face_{face_number}.jpg"
        )
        
    @classmethod
    def _from_filename_without_sequence(cls, filename: str) -> 'Frame':
        parts = filename.split("_")
        assert len(parts) == 5, f"Unexpected filename format: {filename}"
        assert parts[0] == "frame", f"Unexpected filename format: {filename}"
        assert parts[2] == "with", f"Unexpected filename format: {filename}"
        assert parts[3] == "face", f"Unexpected filename format: {filename}"
        assert parts[4].endswith(".jpg"), f"Unexpected filename format: {filename}"

        frame_number = int(parts[1])
        face_number = int(parts[4].split(".")[0])
        return cls(filename, frame_number, face_number)


FrameSequence = List[Frame]


@dataclass
class RococoDataset:
    """Representation of the Rococo dataset

    Only stores filenames and information on the directory structure, does not interact with the filesystem

    The dataset is structured as follows:

    ```
    <root_dir>
        ├── <faces_dir>
        │   ├── frame_1_face_1.jpg
        │   ├── frame_1_face_2.jpg
        │   └── ...
        └── <frames_dir>
            ├── frame_1_face_1_frame_2.jpg
            ├── frame_1_face_2_frame_3.jpg
            └── ...
    ```

    Elements are pairs of (face, frame sequence), only those with at least 3 frames (not all faces).
    All faces are needed for search and comparison in the validation procedure.

    Face files are matched with corresponding frame files based on filename prefix.
    The filenames are misleading - face is identified by `frame_X_face_Y`,
    frames corresponding to the same face are identified by `frame_X_face_Y_frame_Z` and differ only by `Z`.

    Also supports a dataset with multiple sequences of frames for the same face.
    The filenames of frames are in the format `frame_X_face_Y_sequence_Z_frame_W`.
    If there are multiple sequences, there will be multiple elements with the same face (one element for each sequence).
    """

    elements: List[Tuple[Face, FrameSequence]]
    all_faces: List[Face]
    root_dir: str
    faces_dir: str = "faces"
    frames_dir: str = "frames"

    @classmethod
    def from_directory(
        cls, root_dir: str, faces_dir: str = "faces", frames_dir: str = "frames"
    ) -> 'RococoDataset':
        faces_path = os.path.join(root_dir, faces_dir)
        frames_path = os.path.join(root_dir, frames_dir)

        face_filenames = sorted(list(os.listdir(faces_path)))
        frame_filenames = sorted(list(os.listdir(frames_path)))

        faces = {filename: Face.from_filename(filename) for filename in face_filenames}
        frames = {
            filename: Frame.from_filename(filename) for filename in frame_filenames
        }

        face_frames = cls.get_matching_frames_for_faces(faces, frames)

        elements = [
            (faces[face_filename], face_frames[face_filename])
            for face_filename in face_frames.keys()
        ]

        all_faces = cls.get_all_faces(faces_dict=faces, elements=elements)
        all_faces = sorted(all_faces, key=lambda face: face.filename)

        # TODO verify if this is a good idea
        assert len(elements) == len(all_faces)

        return cls(
            elements=elements,
            all_faces=all_faces,
            root_dir=root_dir,
            faces_dir=faces_dir,
            frames_dir=frames_dir,
        )


    def face_abspath(self, face: Face) -> str:
        return os.path.join(self.root_dir, self.faces_dir, face.filename)

    def frame_abspath(self, frame: Frame) -> str:
        return os.path.join(self.root_dir, self.frames_dir, frame.filename)

    def face_index(self, face: Face) -> int:
        """Return the position of given face in all_faces"""
        return self.all_faces.index(face)

    def __getitem__(self, index: int) -> Tuple[Face, List[Frame]]:
        return self.elements[index]

    def __len__(self) -> int:
        return len(self.elements)

    @staticmethod
    def get_matching_frames_for_faces(
        faces: Dict[str, Face], frames: Dict[str, Frame], minimum_frames: int = 3
    ) -> Dict[str, FrameSequence]:
        faces_with_enough_frames = [
            face
            for face in faces.values()
            if len(
                [
                    frame
                    for frame in frames.values()
                    if frame.face_number == face.face_number
                ]
            )
            >= minimum_frames
        ]

        return {
            face.filename: [
                frame
                for frame in frames.values()
                if frame.face_number == face.face_number
            ]
            for face in faces_with_enough_frames
        }

    @staticmethod
    def get_all_faces(
        faces_dict: Dict[str, Face], elements: List[Tuple[Face, FrameSequence]]
    ) -> List[Face]:
        """Return a list of faces for searching matches

        All faces present in the `faces` directory
        """
        return list(faces_dict.values())
    
    @property
    def n_all_faces(self) -> int:
        """Number of all faces in the dataset"""
        return len(self.all_faces)

