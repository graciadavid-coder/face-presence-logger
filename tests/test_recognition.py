"""SFace gallery and matching tests."""

from pathlib import Path

import numpy as np
import pytest

from face_presence.recognition import (
    FaceRecognizer,
    Gallery,
    RecognitionError,
    average_embeddings,
    normalize,
)
from face_presence.storage import PersonRecord


def one_hot(index: int = 0) -> np.ndarray:
    vector = np.zeros(128, dtype=np.float32)
    vector[index] = 1.0
    return vector


def record(person_id: int, label: str, embedding: np.ndarray) -> PersonRecord:
    return PersonRecord(person_id, label, embedding, "2026-01-01T00:00:00Z")


def test_normalize_returns_a_unit_vector() -> None:
    vector = normalize(np.array([3.0, 4.0], dtype=np.float32))

    assert np.linalg.norm(vector) == pytest.approx(1.0)
    assert vector.tolist() == pytest.approx([0.6, 0.8])


def test_normalize_handles_zero_vector() -> None:
    vector = normalize(np.zeros(4, dtype=np.float32))

    assert np.array_equal(vector, np.zeros(4, dtype=np.float32))


def test_average_embeddings_returns_a_unit_vector() -> None:
    average = average_embeddings([one_hot(0), one_hot(1)])

    assert np.linalg.norm(average) == pytest.approx(1.0)
    assert average[0] == pytest.approx(1.0 / np.sqrt(2.0))
    assert average[1] == pytest.approx(1.0 / np.sqrt(2.0))


def test_average_embeddings_requires_samples() -> None:
    with pytest.raises(ValueError, match="at least one"):
        average_embeddings([])


def test_empty_gallery_returns_no_match() -> None:
    result = Gallery([]).match(one_hot(), threshold=0.38, margin_threshold=0.05)

    assert result.person is None
    assert result.score == 0.0
    assert result.margin == 1.0


def test_exact_embedding_matches() -> None:
    person = record(1, "Person_001", one_hot(0))

    result = Gallery([person]).match(one_hot(0), threshold=0.38, margin_threshold=0.05)

    assert result.person is person
    assert result.score == pytest.approx(1.0)
    assert result.margin == pytest.approx(1.0)


def test_orthogonal_embedding_does_not_match() -> None:
    person = record(1, "Person_001", one_hot(0))

    result = Gallery([person]).match(one_hot(1), threshold=0.38, margin_threshold=0.05)

    assert result.person is None
    assert result.score == pytest.approx(0.0)


def test_margin_rejects_ambiguous_matches() -> None:
    close = normalize(np.array([1.0, 0.05] + [0.0] * 126, dtype=np.float32))
    people = [
        record(1, "Person_001", one_hot(0)),
        record(2, "Person_002", close),
    ]

    result = Gallery(people).match(one_hot(0), threshold=0.38, margin_threshold=0.05)

    assert result.person is None
    assert result.score == pytest.approx(1.0)
    assert result.margin < 0.05


def test_gallery_add_increases_count() -> None:
    gallery = Gallery([])
    gallery.add(record(1, "Person_001", one_hot(0)))

    assert gallery.count == 1
    assert gallery.match(one_hot(0), 0.38, 0.05).person is not None


def test_face_recognizer_requires_a_model(tmp_path: Path) -> None:
    with pytest.raises(RecognitionError, match="not found"):
        FaceRecognizer(tmp_path / "missing.onnx")
