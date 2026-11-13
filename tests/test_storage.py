"""SQLite storage tests."""

from pathlib import Path

import numpy as np
import pytest

from face_presence.storage import Database, StorageError, utc_now


def make_embedding(seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    vector = rng.normal(size=128).astype(np.float32)
    return vector / np.linalg.norm(vector)


def test_new_database_is_empty(tmp_path: Path) -> None:
    with Database(tmp_path / "people.db") as database:
        assert database.load_persons() == []


def test_add_person_assigns_sequential_labels(tmp_path: Path) -> None:
    with Database(tmp_path / "people.db") as database:
        first = database.add_person(make_embedding(1))
        second = database.add_person(make_embedding(2))

    assert first.id == 1
    assert first.label == "Person_001"
    assert second.id == 2
    assert second.label == "Person_002"
    assert np.linalg.norm(first.embedding) == pytest.approx(1.0)
    assert first.created_at.endswith("Z")


def test_people_persist_across_reopen(tmp_path: Path) -> None:
    path = tmp_path / "people.db"
    vector = make_embedding(3)

    with Database(path) as database:
        database.add_person(vector)
    with Database(path) as database:
        people = database.load_persons()

    assert len(people) == 1
    assert people[0].label == "Person_001"
    np.testing.assert_allclose(people[0].embedding, vector, atol=1e-6)


def test_add_person_normalizes_the_embedding(tmp_path: Path) -> None:
    with Database(tmp_path / "people.db") as database:
        person = database.add_person(make_embedding(4) * 7.0)

    assert np.linalg.norm(person.embedding) == pytest.approx(1.0)


def test_rejects_wrong_embedding_size(tmp_path: Path) -> None:
    with Database(tmp_path / "people.db") as database, pytest.raises(
        StorageError, match="128-dimensional"
    ):
        database.add_person(np.zeros(16, dtype=np.float32))


def test_utc_now_uses_iso_format() -> None:
    value = utc_now()

    assert value.endswith("Z")
    assert "T" in value
