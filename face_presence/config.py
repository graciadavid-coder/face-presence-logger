"""Application configuration loading and validation."""

from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

BACKENDS = {"auto", "default", "dshow", "msmf", "avfoundation", "v4l2"}


class ConfigError(ValueError):
    """Raised when the configuration file is missing or invalid."""


@dataclass(frozen=True)
class ModelSettings:
    directory: Path


@dataclass(frozen=True)
class CameraSettings:
    index: int = 0
    width: int = 640
    height: int = 480
    fps: int = 30
    backend: str = "auto"


@dataclass(frozen=True)
class DetectionSettings:
    input_width: int = 320
    score_threshold: float = 0.85
    nms_threshold: float = 0.3
    interval_frames: int = 2
    adaptive: bool = True


@dataclass(frozen=True)
class RecognitionSettings:
    match_threshold: float = 0.38
    match_margin: float = 0.05
    enroll_threshold: float = 0.36
    enroll_margin: float = 0.03
    confirm_votes: int = 2
    enroll_samples: int = 3
    min_face_px: int = 80
    interval_s: float = 0.5


@dataclass(frozen=True)
class PresenceSettings:
    absence_timeout_s: float = 5.0
    heartbeat_s: float = 5.0


@dataclass(frozen=True)
class StorageSettings:
    db_path: Path
    export_dir: Path


@dataclass(frozen=True)
class UISettings:
    show_fps: bool = True
    show_landmarks: bool = False
    window_title: str = "Face Presence Logger"


@dataclass(frozen=True)
class Settings:
    models: ModelSettings
    camera: CameraSettings
    detection: DetectionSettings
    recognition: RecognitionSettings
    presence: PresenceSettings
    storage: StorageSettings
    ui: UISettings
    config_path: Path
    base_dir: Path


def load_settings(path: str | Path = "config.toml") -> Settings:
    """Load and validate a TOML configuration file."""
    config_path = _resolve_config_path(path)
    raw = _read_toml(config_path)
    base_dir = config_path.parent

    models_table = _section(raw, "models")
    camera_table = _section(raw, "camera")
    detection_table = _section(raw, "detection")
    recognition_table = _section(raw, "recognition")
    presence_table = _section(raw, "presence")
    storage_table = _section(raw, "storage")
    ui_table = _section(raw, "ui")

    camera = CameraSettings(
        index=_int(camera_table, "index", 0, "camera"),
        width=_int(camera_table, "width", 640, "camera"),
        height=_int(camera_table, "height", 480, "camera"),
        fps=_int(camera_table, "fps", 30, "camera"),
        backend=_str(camera_table, "backend", "auto", "camera").lower(),
    )
    detection = DetectionSettings(
        input_width=_int(detection_table, "input_width", 320, "detection"),
        score_threshold=_float(detection_table, "score_threshold", 0.85, "detection"),
        nms_threshold=_float(detection_table, "nms_threshold", 0.3, "detection"),
        interval_frames=_int(detection_table, "interval_frames", 2, "detection"),
        adaptive=_bool(detection_table, "adaptive", True, "detection"),
    )
    recognition = RecognitionSettings(
        match_threshold=_float(recognition_table, "match_threshold", 0.38, "recognition"),
        match_margin=_float(recognition_table, "match_margin", 0.05, "recognition"),
        enroll_threshold=_float(
            recognition_table, "enroll_threshold", 0.36, "recognition"
        ),
        enroll_margin=_float(recognition_table, "enroll_margin", 0.03, "recognition"),
        confirm_votes=_int(recognition_table, "confirm_votes", 2, "recognition"),
        enroll_samples=_int(recognition_table, "enroll_samples", 3, "recognition"),
        min_face_px=_int(recognition_table, "min_face_px", 80, "recognition"),
        interval_s=_float(recognition_table, "interval_s", 0.5, "recognition"),
    )
    presence = PresenceSettings(
        absence_timeout_s=_float(presence_table, "absence_timeout_s", 5.0, "presence"),
        heartbeat_s=_float(presence_table, "heartbeat_s", 5.0, "presence"),
    )
    ui = UISettings(
        show_fps=_bool(ui_table, "show_fps", True, "ui"),
        show_landmarks=_bool(ui_table, "show_landmarks", False, "ui"),
        window_title=_str(ui_table, "window_title", "Face Presence Logger", "ui"),
    )

    _validate_camera(camera)
    _validate_detection(detection)
    _validate_recognition(recognition)
    _validate_presence(presence)

    storage = StorageSettings(
        db_path=_path(storage_table, "db_path", "data/presence.db", "storage", base_dir),
        export_dir=_path(storage_table, "export_dir", "data/exports", "storage", base_dir),
    )

    return Settings(
        models=ModelSettings(
            directory=_path(models_table, "directory", "models", "models", base_dir)
        ),
        camera=camera,
        detection=detection,
        recognition=recognition,
        presence=presence,
        storage=storage,
        ui=ui,
        config_path=config_path,
        base_dir=base_dir,
    )


def _resolve_config_path(path: str | Path) -> Path:
    candidate = Path(path).expanduser()
    if not candidate.is_absolute():
        candidate = Path.cwd() / candidate
    candidate = candidate.resolve()
    if not candidate.is_file():
        raise ConfigError(f"Configuration file not found: {candidate}")
    return candidate


def _read_toml(path: Path) -> dict[str, Any]:
    try:
        with path.open("rb") as handle:
            return tomllib.load(handle)
    except tomllib.TOMLDecodeError as exc:
        raise ConfigError(f"Invalid TOML in {path}: {exc}") from exc
    except OSError as exc:
        raise ConfigError(f"Could not read {path}: {exc}") from exc


def _section(raw: dict[str, Any], name: str) -> dict[str, Any]:
    value = raw.get(name, {})
    if not isinstance(value, dict):
        raise ConfigError(f"[{name}] must be a table")
    return value


def _int(table: dict[str, Any], key: str, default: int, section: str) -> int:
    value = table.get(key, default)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ConfigError(f"{section}.{key} must be an integer")
    return value


def _float(table: dict[str, Any], key: str, default: float, section: str) -> float:
    value = table.get(key, default)
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ConfigError(f"{section}.{key} must be a number")
    return float(value)


def _bool(table: dict[str, Any], key: str, default: bool, section: str) -> bool:
    value = table.get(key, default)
    if not isinstance(value, bool):
        raise ConfigError(f"{section}.{key} must be true or false")
    return value


def _str(table: dict[str, Any], key: str, default: str, section: str) -> str:
    value = table.get(key, default)
    if not isinstance(value, str) or not value.strip():
        raise ConfigError(f"{section}.{key} must be a non-empty string")
    return value.strip()


def _path(
    table: dict[str, Any], key: str, default: str, section: str, base_dir: Path
) -> Path:
    value = table.get(key, default)
    if not isinstance(value, str) or not value.strip():
        raise ConfigError(f"{section}.{key} must be a non-empty path")
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = base_dir / path
    return path.resolve()


def _validate_camera(camera: CameraSettings) -> None:
    if camera.index < 0:
        raise ConfigError("camera.index must be zero or greater")
    if camera.width <= 0 or camera.height <= 0:
        raise ConfigError("camera.width and camera.height must be positive")
    if camera.fps <= 0:
        raise ConfigError("camera.fps must be positive")
    if camera.backend not in BACKENDS:
        allowed = ", ".join(sorted(BACKENDS))
        raise ConfigError(f"camera.backend must be one of: {allowed}")


def _validate_detection(detection: DetectionSettings) -> None:
    if detection.input_width < 64:
        raise ConfigError("detection.input_width must be at least 64")
    if not 0.0 < detection.score_threshold <= 1.0:
        raise ConfigError("detection.score_threshold must be between 0 and 1")
    if not 0.0 <= detection.nms_threshold <= 1.0:
        raise ConfigError("detection.nms_threshold must be between 0 and 1")
    if detection.interval_frames < 1:
        raise ConfigError("detection.interval_frames must be at least 1")


def _validate_recognition(recognition: RecognitionSettings) -> None:
    if not 0.0 <= recognition.match_threshold <= 1.0:
        raise ConfigError("recognition.match_threshold must be between 0 and 1")
    if recognition.match_margin < 0.0:
        raise ConfigError("recognition.match_margin cannot be negative")
    if not 0.0 <= recognition.enroll_threshold <= 1.0:
        raise ConfigError("recognition.enroll_threshold must be between 0 and 1")
    if recognition.enroll_margin < 0.0:
        raise ConfigError("recognition.enroll_margin cannot be negative")
    if recognition.confirm_votes < 1:
        raise ConfigError("recognition.confirm_votes must be at least 1")
    if recognition.enroll_samples < 1:
        raise ConfigError("recognition.enroll_samples must be at least 1")
    if recognition.min_face_px <= 0:
        raise ConfigError("recognition.min_face_px must be positive")
    if recognition.interval_s <= 0.0:
        raise ConfigError("recognition.interval_s must be positive")


def _validate_presence(presence: PresenceSettings) -> None:
    if presence.absence_timeout_s <= 0.0:
        raise ConfigError("presence.absence_timeout_s must be positive")
    if presence.heartbeat_s <= 0.0:
        raise ConfigError("presence.heartbeat_s must be positive")
