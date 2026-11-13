"""Presence session tracking and live timers."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass

from face_presence.config import PresenceSettings
from face_presence.storage import Database, utc_now
from face_presence.tracking import Track

logger = logging.getLogger(__name__)


@dataclass
class ActiveSession:
    session_id: int
    person_id: int
    started_monotonic: float
    started_at: str
    last_seen_monotonic: float
    last_seen_at: str
    last_heartbeat: float


class PresenceManager:
    """Open, extend and close presence sessions for recognized people."""

    def __init__(
        self,
        database: Database,
        camera_index: int,
        settings: PresenceSettings,
    ) -> None:
        self._database = database
        self._camera_index = camera_index
        self._settings = settings
        self._active: dict[int, ActiveSession] = {}

    @property
    def active_count(self) -> int:
        return len(self._active)

    def update(self, tracks: Sequence[Track], now: float) -> None:
        """Close expired sessions, then extend the sessions of visible people."""
        for person_id, session in list(self._active.items()):
            if now - session.last_seen_monotonic > self._settings.absence_timeout_s:
                self._close(person_id, detail="timeout")

        seen: dict[int, Track] = {}
        for track in tracks:
            if track.person_id is not None:
                seen.setdefault(track.person_id, track)

        for person_id in seen:
            session = self._active.get(person_id)
            if session is None:
                session = self._open(person_id, now)
            session.last_seen_monotonic = now
            session.last_seen_at = utc_now()
            if now - session.last_heartbeat >= self._settings.heartbeat_s:
                duration = max(0.0, now - session.started_monotonic)
                self._database.touch_session(
                    session.session_id, session.last_seen_at, duration
                )
                session.last_heartbeat = now

        for track in tracks:
            if track.person_id is None:
                continue
            session = self._active.get(track.person_id)
            if session is not None:
                track.timer_s = max(0.0, now - session.started_monotonic)

    def close_all(self, detail: str = "shutdown") -> None:
        """Close every active session, normally during a graceful shutdown."""
        for person_id in list(self._active):
            self._close(person_id, detail=detail)

    def _open(self, person_id: int, now: float) -> ActiveSession:
        started_at = utc_now()
        session_id = self._database.start_session(
            person_id, self._camera_index, started_at
        )
        session = ActiveSession(
            session_id=session_id,
            person_id=person_id,
            started_monotonic=now,
            started_at=started_at,
            last_seen_monotonic=now,
            last_seen_at=started_at,
            last_heartbeat=now,
        )
        self._active[person_id] = session
        logger.info("Presence session started for person %d", person_id)
        return session

    def _close(self, person_id: int, detail: str) -> None:
        session = self._active.pop(person_id)
        duration = max(0.0, session.last_seen_monotonic - session.started_monotonic)
        self._database.close_session(
            session.session_id,
            session.person_id,
            session.last_seen_at,
            duration,
            detail,
        )
        logger.info("Presence session closed after %.1f s", duration)
