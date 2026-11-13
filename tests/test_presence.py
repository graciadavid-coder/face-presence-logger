"""Presence session tracking tests."""

from pathlib import Path

import numpy as np
import pytest

from face_presence.config import PresenceSettings
from face_presence.presence import PresenceManager
from face_presence.storage import Database
from face_presence.tracking import Track


def make_embedding(seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    vector = rng.normal(size=128).astype(np.float32)
    return vector / np.linalg.norm(vector)


def make_track(
    person_id: int | None, track_id: int = 1, label: str | None = None
) -> Track:
    return Track(
        track_id=track_id,
        box=(0.0, 0.0, 100.0, 100.0),
        last_seen=0.0,
        person_id=person_id,
        label=label or (f"Person_{person_id:03d}" if person_id else None),
    )


def test_session_starts_extends_and_closes(tmp_path: Path) -> None:
    with Database(tmp_path / "presence.db") as database:
        person = database.add_person(make_embedding())
        manager = PresenceManager(
            database,
            0,
            PresenceSettings(absence_timeout_s=5.0, heartbeat_s=2.0),
        )
        track = make_track(person.id)

        manager.update([track], now=0.0)
        assert manager.active_count == 1
        assert track.timer_s == 0.0

        manager.update([track], now=2.5)
        assert track.timer_s == pytest.approx(2.5)
        active = database.load_sessions()[0]
        assert active.status == "active"
        assert active.duration_s == pytest.approx(2.5)

        manager.update([], now=7.6)
        assert manager.active_count == 0
        closed = database.load_sessions()[0]
        assert closed.status == "closed"
        assert closed.duration_s == pytest.approx(2.5)
        assert closed.ended_at is not None


def test_reappearance_after_timeout_starts_a_new_session(tmp_path: Path) -> None:
    with Database(tmp_path / "presence.db") as database:
        person = database.add_person(make_embedding())
        manager = PresenceManager(
            database,
            0,
            PresenceSettings(absence_timeout_s=5.0, heartbeat_s=5.0),
        )

        manager.update([make_track(person.id)], now=0.0)
        manager.update([], now=6.0)
        assert manager.active_count == 0

        manager.update([make_track(person.id)], now=7.0)
        assert manager.active_count == 1
        assert len(database.load_sessions()) == 2


def test_close_all_finalizes_active_sessions(tmp_path: Path) -> None:
    with Database(tmp_path / "presence.db") as database:
        person = database.add_person(make_embedding())
        manager = PresenceManager(
            database,
            0,
            PresenceSettings(absence_timeout_s=5.0, heartbeat_s=5.0),
        )
        track = make_track(person.id)

        manager.update([track], now=0.0)
        manager.update([track], now=2.0)
        manager.close_all()

        sessions = database.load_sessions()
        assert len(sessions) == 1
        assert sessions[0].status == "closed"
        assert sessions[0].duration_s == pytest.approx(2.0)


def test_multiple_tracks_for_the_same_person_share_one_session(
    tmp_path: Path,
) -> None:
    with Database(tmp_path / "presence.db") as database:
        person = database.add_person(make_embedding())
        manager = PresenceManager(
            database,
            0,
            PresenceSettings(absence_timeout_s=5.0, heartbeat_s=5.0),
        )
        first = make_track(person.id, track_id=1)
        second = make_track(person.id, track_id=2)

        manager.update([first, second], now=0.0)

        assert manager.active_count == 1
        assert len(database.load_sessions()) == 1
        assert first.timer_s == second.timer_s == 0.0


def test_unrecognized_tracks_do_not_open_sessions(tmp_path: Path) -> None:
    with Database(tmp_path / "presence.db") as database:
        manager = PresenceManager(
            database,
            0,
            PresenceSettings(absence_timeout_s=5.0, heartbeat_s=5.0),
        )

        manager.update([make_track(None)], now=0.0)

        assert manager.active_count == 0
        assert database.load_sessions() == []
