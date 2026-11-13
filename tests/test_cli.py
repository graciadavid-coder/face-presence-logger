"""Command line parser tests."""

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
