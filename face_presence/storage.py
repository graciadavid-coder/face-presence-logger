"""SQLite persistence for known people."""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import numpy as np

EMBEDDING_SIZE = 128

_SCHEMA = """
CREATE TABLE IF NOT EXISTS persons (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    label       TEXT NOT NULL UNIQUE,
    embedding   BLOB NOT NULL,
    created_at  TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS events (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    person_id   INTEGER REFERENCES persons(id) ON DELETE SET NULL,
    type        TEXT NOT NULL,
    created_at  TEXT NOT NULL,
    detail      TEXT
);

CREATE INDEX IF NOT EXISTS idx_events_person ON events(person_id, created_at);
"""


class StorageError(RuntimeError):
    """Raised when the local database cannot be read or written."""


@dataclass(frozen=True)
class PersonRecord:
    id: int
    label: str
    embedding: np.ndarray
    created_at: str


def utc_now() -> str:
    """Return the current UTC time as an ISO-8601 string."""
    return (
        datetime.now(UTC)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


class Database:
    """Local SQLite store for face templates and enrollment events."""

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
