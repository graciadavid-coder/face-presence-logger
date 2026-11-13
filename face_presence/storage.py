"""SQLite persistence for people, presence sessions and exports."""

from __future__ import annotations

import csv
import logging
import sqlite3
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import numpy as np

from face_presence.formatting import format_duration

logger = logging.getLogger(__name__)

EMBEDDING_SIZE = 128

_SCHEMA = """
CREATE TABLE IF NOT EXISTS persons (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    label       TEXT NOT NULL UNIQUE,
    embedding   BLOB NOT NULL,
    created_at  TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS sessions (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    person_id     INTEGER NOT NULL REFERENCES persons(id) ON DELETE CASCADE,
    camera_index  INTEGER NOT NULL,
    started_at    TEXT NOT NULL,
    last_seen_at  TEXT NOT NULL,
    ended_at      TEXT,
    duration_s    REAL NOT NULL DEFAULT 0,
    status        TEXT NOT NULL CHECK (status IN ('active', 'closed'))
);

CREATE TABLE IF NOT EXISTS events (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    person_id   INTEGER REFERENCES persons(id) ON DELETE SET NULL,
    type        TEXT NOT NULL,
    created_at  TEXT NOT NULL,
    detail      TEXT
);

CREATE INDEX IF NOT EXISTS idx_sessions_person
    ON sessions(person_id, started_at);
CREATE INDEX IF NOT EXISTS idx_sessions_status
    ON sessions(status);
CREATE INDEX IF NOT EXISTS idx_events_person
    ON events(person_id, created_at);
"""


class StorageError(RuntimeError):
    """Raised when the local database cannot be read or written."""


@dataclass(frozen=True)
class PersonRecord:
    id: int
    label: str
    embedding: np.ndarray
    created_at: str


@dataclass(frozen=True)
class SessionRecord:
    id: int
    person_id: int
    person_label: str
    camera_index: int
    started_at: str
    last_seen_at: str
    ended_at: str | None
    duration_s: float
    status: str


def utc_now() -> str:
    """Return the current UTC time as an ISO-8601 string."""
    return (
        datetime.now(UTC)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


class Database:
    """Local SQLite store for face templates, sessions and exports."""

    def __init__(self, path: Path) -> None:
        self._path = path
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            self._connection = sqlite3.connect(str(path), timeout=5.0)
        except (OSError, sqlite3.Error) as exc:
            raise StorageError(f"Could not open the database at {path}: {exc}") from exc

        self._connection.row_factory = sqlite3.Row
        try:
            self._connection.execute("PRAGMA journal_mode = WAL")
            self._connection.execute("PRAGMA synchronous = NORMAL")
            self._connection.execute("PRAGMA foreign_keys = ON")
            self._connection.executescript(_SCHEMA)
            self._connection.commit()
        except sqlite3.Error as exc:
            self._connection.close()
            raise StorageError(f"Could not initialize the database: {exc}") from exc

    @property
    def path(self) -> Path:
        return self._path

    def load_persons(self) -> list[PersonRecord]:
        """Load every stored person ordered by identifier."""
        try:
            rows = self._connection.execute(
                "SELECT id, label, embedding, created_at FROM persons ORDER BY id"
            ).fetchall()
        except sqlite3.Error as exc:
            raise StorageError(f"Could not read known people: {exc}") from exc

        people: list[PersonRecord] = []
        for row in rows:
            try:
                vector = np.frombuffer(row["embedding"], dtype=np.float32).copy()
            except ValueError as exc:
                raise StorageError(f"Corrupt embedding for {row['label']}") from exc
            if vector.size != EMBEDDING_SIZE:
                raise StorageError(f"Unexpected embedding size for {row['label']}")
            people.append(
                PersonRecord(row["id"], row["label"], vector, row["created_at"])
            )
        return people

    def add_person(self, embedding: np.ndarray) -> PersonRecord:
        """Store a normalized face embedding and return the new person record."""
        vector = _normalize_embedding(embedding)
        created_at = utc_now()
        temporary_label = f"pending-{uuid4().hex}"

        try:
            with self._connection:
                cursor = self._connection.execute(
                    "INSERT INTO persons(label, embedding, created_at) VALUES(?, ?, ?)",
                    (temporary_label, vector.tobytes(), created_at),
                )
                person_id = int(cursor.lastrowid)
                label = f"Person_{person_id:03d}"
                self._connection.execute(
                    "UPDATE persons SET label = ? WHERE id = ?", (label, person_id)
                )
                self._connection.execute(
                    "INSERT INTO events(person_id, type, created_at, detail) "
                    "VALUES(?, ?, ?, ?)",
                    (person_id, "person_enrolled", created_at, None),
                )
        except sqlite3.Error as exc:
            raise StorageError(f"Could not store a new person: {exc}") from exc

        return PersonRecord(person_id, label, vector, created_at)

    def start_session(
        self, person_id: int, camera_index: int, started_at: str
    ) -> int:
        """Insert an active session and return its identifier."""
        try:
            with self._connection:
                cursor = self._connection.execute(
                    "INSERT INTO sessions(person_id, camera_index, started_at, "
                    "last_seen_at, ended_at, duration_s, status) "
                    "VALUES(?, ?, ?, ?, NULL, 0, 'active')",
                    (person_id, camera_index, started_at, started_at),
                )
                session_id = int(cursor.lastrowid)
                self._connection.execute(
                    "INSERT INTO events(person_id, type, created_at, detail) "
                    "VALUES(?, ?, ?, ?)",
                    (person_id, "session_started", started_at, None),
                )
        except sqlite3.Error as exc:
            raise StorageError(f"Could not start a presence session: {exc}") from exc
        return session_id

    def touch_session(
        self, session_id: int, last_seen_at: str, duration_s: float
    ) -> None:
        """Update the heartbeat of an active session."""
        try:
            with self._connection:
                self._connection.execute(
                    "UPDATE sessions SET last_seen_at = ?, duration_s = ? "
                    "WHERE id = ? AND status = 'active'",
                    (last_seen_at, max(0.0, duration_s), session_id),
                )
        except sqlite3.Error as exc:
            raise StorageError(f"Could not update a presence session: {exc}") from exc

    def close_session(
        self,
        session_id: int,
        person_id: int,
        ended_at: str,
        duration_s: float,
        detail: str | None = None,
    ) -> None:
        """Close an active session using its last observed timestamp."""
        try:
            with self._connection:
                cursor = self._connection.execute(
                    "UPDATE sessions SET last_seen_at = ?, ended_at = ?, "
                    "duration_s = ?, status = 'closed' "
                    "WHERE id = ? AND status = 'active'",
                    (ended_at, ended_at, max(0.0, duration_s), session_id),
                )
                if cursor.rowcount:
                    self._connection.execute(
                        "INSERT INTO events(person_id, type, created_at, detail) "
                        "VALUES(?, ?, ?, ?)",
                        (person_id, "session_closed", ended_at, detail),
                    )
        except sqlite3.Error as exc:
            raise StorageError(f"Could not close a presence session: {exc}") from exc

    def close_stale_sessions(self) -> int:
        """Close sessions that were left active by an abrupt shutdown."""
        try:
            rows = self._connection.execute(
                "SELECT id, person_id, started_at, last_seen_at "
                "FROM sessions WHERE status = 'active'"
            ).fetchall()
        except sqlite3.Error as exc:
            raise StorageError(f"Could not read active sessions: {exc}") from exc

        closed = 0
        for row in rows:
            try:
                started = _parse_utc(row["started_at"])
                last_seen = _parse_utc(row["last_seen_at"])
                duration = max(0.0, (last_seen - started).total_seconds())
            except ValueError:
                logger.warning("Invalid timestamp in session %s", row["id"])
                duration = 0.0

            try:
                with self._connection:
                    cursor = self._connection.execute(
                        "UPDATE sessions SET ended_at = ?, duration_s = ?, "
                        "status = 'closed' "
                        "WHERE id = ? AND status = 'active'",
                        (row["last_seen_at"], duration, row["id"]),
                    )
                    if cursor.rowcount:
                        self._connection.execute(
                            "INSERT INTO events(person_id, type, created_at, detail) "
                            "VALUES(?, ?, ?, ?)",
                            (
                                row["person_id"],
                                "session_closed",
                                row["last_seen_at"],
                                "startup_recovery",
                            ),
                        )
                        closed += 1
            except sqlite3.Error as exc:
                raise StorageError(
                    f"Could not recover a presence session: {exc}"
                ) from exc
        return closed

    def load_sessions(
        self, since: str | None = None, until: str | None = None
    ) -> list[SessionRecord]:
        """Load sessions ordered by start time, optionally filtered by start date."""
        query = (
            "SELECT s.id, s.person_id, p.label AS person_label, s.camera_index, "
            "s.started_at, s.last_seen_at, s.ended_at, s.duration_s, s.status "
            "FROM sessions s "
            "JOIN persons p ON p.id = s.person_id "
            "WHERE (? IS NULL OR s.started_at >= ?) "
            "AND (? IS NULL OR s.started_at <= ?) "
            "ORDER BY s.started_at, s.id"
        )
        try:
            rows = self._connection.execute(
                query, (since, since, until, until)
            ).fetchall()
        except sqlite3.Error as exc:
            raise StorageError(f"Could not read presence sessions: {exc}") from exc

        return [
            SessionRecord(
                id=row["id"],
                person_id=row["person_id"],
                person_label=row["person_label"],
                camera_index=row["camera_index"],
                started_at=row["started_at"],
                last_seen_at=row["last_seen_at"],
                ended_at=row["ended_at"],
                duration_s=float(row["duration_s"]),
                status=row["status"],
            )
            for row in rows
        ]

    def export_sessions(
        self,
        output: Path,
        since: str | None = None,
        until: str | None = None,
    ) -> int:
        """Write sessions to a UTF-8 CSV file and return the number of rows."""
        sessions = self.load_sessions(since=since, until=until)
        try:
            output.parent.mkdir(parents=True, exist_ok=True)
            with output.open("w", newline="", encoding="utf-8-sig") as handle:
                writer = csv.writer(handle)
                writer.writerow(
                    [
                        "session_id",
                        "person_label",
                        "camera_index",
                        "started_at_utc",
                        "last_seen_at_utc",
                        "ended_at_utc",
                        "duration_seconds",
                        "duration_hhmmss",
                        "status",
                    ]
                )
                for session in sessions:
                    writer.writerow(
                        [
                            session.id,
                            session.person_label,
                            session.camera_index,
                            session.started_at,
                            session.last_seen_at,
                            session.ended_at or "",
                            f"{session.duration_s:.2f}",
                            format_duration(session.duration_s),
                            session.status,
                        ]
                    )
        except OSError as exc:
            raise StorageError(f"Could not write the export file: {exc}") from exc
        return len(sessions)

    def close(self) -> None:
        self._connection.close()

    def __enter__(self) -> Database:
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()


def _normalize_embedding(embedding: np.ndarray) -> np.ndarray:
    vector = np.asarray(embedding, dtype=np.float32).reshape(-1)
    if vector.size != EMBEDDING_SIZE:
        raise StorageError(
            f"Expected a {EMBEDDING_SIZE}-dimensional embedding, got {vector.size}"
        )
    norm = float(np.linalg.norm(vector))
    if not np.isfinite(norm) or norm == 0.0:
        raise StorageError("The embedding must contain finite, non-zero values")
    return vector / norm


def _parse_utc(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))
