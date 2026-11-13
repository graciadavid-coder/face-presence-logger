"""Command line parser tests."""

from pathlib import Path

from face_presence.cli import build_parser


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
