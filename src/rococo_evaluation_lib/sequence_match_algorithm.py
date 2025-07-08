from typing import List, Union


def sequence_match(indices: List[Union[int, None]], sequence_length: int) -> Union[int, None]:
    n_in_sequence = 0
    matched_idx = None

    for face_idx in indices:
        if face_idx is None:
            n_in_sequence = 0
            matched_idx = None
            continue
            
        if matched_idx is None:
            matched_idx = face_idx

        if face_idx == matched_idx:
            n_in_sequence += 1
        else:
            n_in_sequence = 1
            matched_idx = face_idx

        if n_in_sequence == sequence_length:
            return matched_idx

    return None