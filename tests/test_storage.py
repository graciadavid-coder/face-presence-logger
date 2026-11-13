"""SQLite storage tests."""

import csv
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


def test_session_lifecycle_and_csv_export(tmp_path: Path) -> None:
    database_path = tmp_path / "people.db"
    csv_path = tmp_path / "exports" / "sessions.csv"

    with Database(database_path) as database:
        person = database.add_person(make_embedding(7))
        session_id = database.start_session(
            person.id, 0, "2026-11-01T10:00:00Z"
        )
        database.touch_session(session_id, "2026-11-01T10:00:30Z", 30.0)
        database.close_session(
            session_id,
            person.id,
            "2026-11-01T10:00:30Z",
            30.0,
            "timeout",
        )

        sessions = database.load_sessions()
        assert len(sessions) == 1
        assert sessions[0].person_label == "Person_001"
        assert sessions[0].duration_s == 30.0
        assert sessions[0].status == "closed"

        assert database.export_sessions(csv_path) == 1

    with csv_path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.reader(handle))

    assert rows[0][0] == "session_id"
    assert rows[1][0] == str(session_id)
    assert rows[1][1] == "Person_001"
    assert rows[1][7] == "00:30"
    assert rows[1][8] == "closed"


def test_close_stale_sessions_on_reopen(tmp_path: Path) -> None:
    path = tmp_path / "people.db"

    with Database(path) as database:
        person = database.add_person(make_embedding(8))
        session_id = database.start_session(
            person.id, 0, "2026-11-01T10:00:00Z"
        )
        database.touch_session(session_id, "2026-11-01T10:00:05Z", 5.0)

    with Database(path) as database:
        assert database.close_stale_sessions() == 1
        sessions = database.load_sessions()
        assert sessions[0].status == "closed"
        assert sessions[0].duration_s == 5.0
        assert sessions[0].ended_at == "2026-11-01T10:00:05Z"
        assert database.close_stale_sessions() == 0


def test_load_sessions_filters_by_start_date(tmp_path: Path) -> None:
    with Database(tmp_path / "people.db") as database:
        person = database.add_person(make_embedding(9))
        first = database.start_session(person.id, 0, "2026-11-01T10:00:00Z")
        second = database.start_session(person.id, 1, "2026-11-02T10:00:00Z")

        filtered = database.load_sessions(since="2026-11-02T00:00:00Z")

    assert [session.id for session in filtered] == [second]
    assert first not in [session.id for session in filtered]


def test_list_and_find_people(tmp_path: Path) -> None:
    with Database(tmp_path / "people.db") as database:
        first = database.add_person(make_embedding(10))
        second = database.add_person(make_embedding(11))
        session_id = database.start_session(
            first.id, 0, "2026-11-01T10:00:00Z"
        )
        database.touch_session(session_id, "2026-11-01T10:00:05Z", 5.0)

        people = database.list_persons()

        assert [person.id for person in people] == [first.id, second.id]
        assert people[0].session_count == 1
        assert people[0].last_seen_at == "2026-11-01T10:00:05Z"
        assert people[1].session_count == 0
        assert people[1].last_seen_at is None

        assert database.find_person(str(first.id)).label == "Person_001"
        assert database.find_person("Person_001").id == first.id
        assert database.find_person("person_001").id == first.id
        assert database.find_person("missing") is None


def test_rename_person_rejects_case_insensitive_duplicates(tmp_path: Path) -> None:
    with Database(tmp_path / "people.db") as database:
        first = database.add_person(make_embedding(12))
        second = database.add_person(make_embedding(13))

        renamed = database.rename_person(first.id, "David")

        assert renamed is not None
        assert renamed.label == "David"
        assert database.find_person("Person_001") is None
        assert database.find_person("david").id == first.id

        with pytest.raises(StorageError, match="already in use"):
            database.rename_person(second.id, "david")

        assert database.rename_person(999, "Nobody") is None


def test_delete_person_cascades_sessions(tmp_path: Path) -> None:
    with Database(tmp_path / "people.db") as database:
        person = database.add_person(make_embedding(14))
        database.start_session(person.id, 0, "2026-11-01T10:00:00Z")

        assert database.delete_person(person.id)
        assert database.load_persons() == []
        assert database.load_sessions() == []
        assert not database.delete_person(person.id)
