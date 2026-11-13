"""Configuration loading and validation tests."""

from pathlib import Path

import pytest

from face_presence.config import ConfigError, load_settings


def test_empty_config_uses_defaults(tmp_path: Path) -> None:
    config = tmp_path / "config.toml"
    config.write_text("", encoding="utf-8")

    settings = load_settings(config)

    assert settings.camera.index == 0
    assert settings.camera.width == 640
    assert settings.camera.backend == "auto"
    assert settings.detection.input_width == 320
    assert settings.recognition.match_threshold == 0.38
    assert settings.recognition.enroll_threshold == 0.36
    assert settings.recognition.enroll_margin == 0.03
    assert settings.presence.absence_timeout_s == 5.0
    assert settings.models.directory == (tmp_path / "models").resolve()
    assert settings.storage.db_path == (tmp_path / "data" / "presence.db").resolve()
    assert settings.ui.window_title == "Face Presence Logger"
    assert settings.config_path == config


def test_custom_values_are_loaded(tmp_path: Path) -> None:
    config = tmp_path / "config.toml"
    config.write_text(
        """
[models]
directory = "assets/models"

[camera]
index = 2
width = 1280
height = 720
fps = 15
backend = "dshow"

[detection]
input_width = 256
score_threshold = 0.9
nms_threshold = 0.4
interval_frames = 3
adaptive = false

[recognition]
match_threshold = 0.42
match_margin = 0.08
enroll_threshold = 0.34
enroll_margin = 0.02
confirm_votes = 3
enroll_samples = 5
min_face_px = 96
interval_s = 0.75

[presence]
absence_timeout_s = 8.0
heartbeat_s = 2.5

[storage]
db_path = "state/presence.db"
export_dir = "state/exports"

[ui]
show_fps = false
show_landmarks = true
window_title = "Custom Window"
""",
        encoding="utf-8",
    )

    settings = load_settings(config)

    assert settings.models.directory == (tmp_path / "assets/models").resolve()
    assert settings.camera.index == 2
    assert settings.camera.backend == "dshow"
    assert settings.detection.adaptive is False
    assert settings.recognition.enroll_threshold == 0.34
    assert settings.recognition.confirm_votes == 3
    assert settings.presence.heartbeat_s == 2.5
    assert settings.storage.export_dir == (tmp_path / "state/exports").resolve()
    assert settings.ui.show_landmarks is True
    assert settings.ui.window_title == "Custom Window"


def test_paths_are_relative_to_the_config_file(tmp_path: Path) -> None:
    config_dir = tmp_path / "project"
    config_dir.mkdir()
    config = config_dir / "config.toml"
    config.write_text(
        '[storage]\ndb_path = "var/presence.db"\n',
        encoding="utf-8",
    )

    settings = load_settings(config)

    assert settings.storage.db_path == (config_dir / "var" / "presence.db").resolve()


def test_missing_config_file_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="not found"):
        load_settings(tmp_path / "missing.toml")


def test_invalid_toml_is_rejected(tmp_path: Path) -> None:
    config = tmp_path / "config.toml"
    config.write_text("[camera\n", encoding="utf-8")

    with pytest.raises(ConfigError, match="Invalid TOML"):
        load_settings(config)


def test_wrong_value_type_is_rejected(tmp_path: Path) -> None:
    config = tmp_path / "config.toml"
    config.write_text('[camera]\nwidth = "wide"\n', encoding="utf-8")

    with pytest.raises(ConfigError, match="camera.width must be an integer"):
        load_settings(config)


def test_invalid_threshold_is_rejected(tmp_path: Path) -> None:
    config = tmp_path / "config.toml"
    config.write_text("[recognition]\nmatch_threshold = 1.4\n", encoding="utf-8")

    with pytest.raises(ConfigError, match="recognition.match_threshold"):
        load_settings(config)


def test_negative_camera_index_is_rejected(tmp_path: Path) -> None:
    config = tmp_path / "config.toml"
    config.write_text("[camera]\nindex = -1\n", encoding="utf-8")

    with pytest.raises(ConfigError, match="camera.index"):
        load_settings(config)


def test_unknown_backend_is_rejected(tmp_path: Path) -> None:
    config = tmp_path / "config.toml"
    config.write_text('[camera]\nbackend = "bogus"\n', encoding="utf-8")

    with pytest.raises(ConfigError, match="camera.backend"):
        load_settings(config)
