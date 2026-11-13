"""Command line parser and persons command tests."""

from pathlib import Path

import numpy as np

from face_presence.cli import build_parser, main
from face_presence.storage import Database


def test_run_accepts_a_camera_override() -> None:
    args = build_parser().parse_args(["run", "--camera", "2"])

    assert args.command == "run"
    assert args.camera == 2


def test_run_uses_the_config_camera_by_default() -> None:
    args = build_parser().parse_args(["run"])

    assert args.camera is None


def test_cameras_accepts_a_max_index() -> None:
    args = build_parser().parse_args(["cameras", "--max-index", "3"])

    assert args.command == "cameras"
    assert args.max_index == 3


def test_export_accepts_dates_and_output_path() -> None:
    args = build_parser().parse_args(
        ["export", "--since", "2026-11-01", "--until", "2026-11-30", "--out", "sessions.csv"]
    )

    assert args.command == "export"
    assert args.since == "2026-11-01"
    assert args.until == "2026-11-30"
    assert args.out == Path("sessions.csv")


def test_persons_accepts_list_rename_and_delete() -> None:
    rename = build_parser().parse_args(
        ["persons", "rename", "Person_001", "David"]
    )
    assert rename.command == "persons"
    assert rename.persons_command == "rename"
    assert rename.person == "Person_001"
    assert rename.label == "David"

    delete = build_parser().parse_args(["persons", "delete", "1", "--yes"])
    assert delete.persons_command == "delete"
    assert delete.person == "1"
    assert delete.yes is True

    listing = build_parser().parse_args(["persons", "list"])
    assert listing.persons_command == "list"


def test_persons_cli_roundtrip(tmp_path: Path, capsys) -> None:
    config = tmp_path / "config.toml"
    database_path = tmp_path / "people.db"
    config.write_text(
        f'[storage]\ndb_path = "{database_path.as_posix()}"\n',
        encoding="utf-8",
    )

    vector = np.zeros(128, dtype=np.float32)
    vector[0] = 1.0
    database = Database(database_path)
    database.add_person(vector)
    database.close()

    assert main(["--config", str(config), "persons", "list"]) == 0
    output = capsys.readouterr().out
    assert "Person_001" in output
    assert "Sessions" in output

    assert (
        main(["--config", str(config), "persons", "rename", "Person_001", "David"])
        == 0
    )
    database = Database(database_path)
    assert database.find_person("David").id == 1
    database.close()

    assert main(["--config", str(config), "persons", "delete", "David"]) == 2
    database = Database(database_path)
    assert database.find_person("David") is not None
    database.close()

    assert (
        main(["--config", str(config), "persons", "delete", "David", "--yes"])
        == 0
    )
    database = Database(database_path)
    assert database.load_persons() == []
    database.close()
