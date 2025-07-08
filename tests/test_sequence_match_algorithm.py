import pytest

from rococo_evaluation_lib.sequence_match_algorithm import sequence_match


@pytest.mark.parametrize("indices,sequence_length,expected", [
    # Test cases for sequence_length = 1
    ((1, 1, 1, 1, 1), 1, 1),
    ((1, 2, 1, 2, 1), 1, 1),
    ((2, 2, 1, 1, 1), 1, 2),
    ((None, None, None, None, None), 1, None),
    ((None, None, None, 1, None), 1, 1),
    ((1, 2, 3, 4, 5), 1, 1),
    ((1, 1, None, 1, 1), 1, 1),
    ((1, 1, 1, None, 1), 1, 1),
    ((1, 1, 1, 1, None), 1, 1),
    
    # Test cases for sequence_length = 2
    ((1, 1, 1, 1, 1), 2, 1),
    ((1, 2, 1, 2, 1), 2, None),
    ((2, 2, 1, 1, 1), 2, 2),
    ((None, None, None, None, None), 2, None),
    ((1, 2, 3, 4, 5), 2, None),
    ((1, 1, None, 1, 1), 2, 1),
    ((1, 1, 1, None, 1), 2, 1),
    ((1, 1, 1, 1, None), 2, 1),
    
    # Test cases for sequence_length = 3
    ((1, 1, 1, 1, 1), 3, 1),
    ((1, 2, 1, 2, 1), 3, None),
    ((2, 2, 1, 1, 1), 3, 1),
    ((None, None, None, None, None), 3, None),
    ((1, 2, 3, 4, 5), 3, None),
    ((1, 1, None, 1, 1), 3, None),
    ((1, 1, 1, None, 1), 3, 1),
    ((1, 1, 1, 1, None), 3, 1),
    ((1, 1, None, 2, 2, 2, 2), 3, 2),
])
def test_sequence_match(indices, sequence_length, expected):
    """Test sequence_match function with various inputs and sequence lengths."""
    result = sequence_match(indices, sequence_length)
    assert result == expected, f"sequence_match({indices}, {sequence_length}) returned {result}, expected {expected}"