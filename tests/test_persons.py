"""Person label validation tests."""

import pytest

from face_presence.persons import validate_label


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("David", "David"),
        ("  David  ", "David"),
        ("Ana   Maria", "Ana Maria"),
        ("O'Connor", "O'Connor"),
        ("Person_A", "Person_A"),
        ("A.B_C-1", "A.B_C-1"),
        ("123A", "123A"),
        ("A" * 32, "A" * 32),
    ],
)
def test_valid_labels_are_normalized(value: str, expected: str) -> None:
    assert validate_label(value) == expected


@pytest.mark.parametrize(
    "value",
    [
        "",
        "   ",
        "123",
        "José",
        "bad/name",
        "David!",
        "David\nAdmin",
        "David\tAdmin",
        "A" * 33,
        "Person_001",
        "person_1000",
    ],
)
def test_invalid_labels_are_rejected(value: str) -> None:
    with pytest.raises(ValueError):
        validate_label(value)
