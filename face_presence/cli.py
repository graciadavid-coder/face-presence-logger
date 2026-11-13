"""Command line entry point."""

from __future__ import annotations

import argparse
import logging
from datetime import date
from pathlib import Path

from face_presence import __version__
from face_presence.camera import CameraError, probe_cameras
from face_presence.config import ConfigError, Settings, load_settings
from face_presence.detection import DetectionError
from face_presence.model_assets import ModelError, ensure_models
from face_presence.pipeline import run_app
from face_presence.recognition import RecognitionError
from face_presence.storage import Database, StorageError

logger = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="face_presence",
        description="CPU-only face recognition and presence logging.",
    )
    parser.add_argument(
        "--config",
        default="config.toml",
        help="Path to the TOML configuration file (default: config.toml).",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Enable debug logging."
    )
    parser.add_argument(
        "--version", action="version", version=f"%(prog)s {__version__}"
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser(
        "setup",
        help=(
            "Download and verify the face models, create runtime folders "
            "and initialize the database."
        ),
    )

    cameras_parser = subparsers.add_parser(
        "cameras", help="List the cameras available on this machine."
    )
    cameras_parser.add_argument(
        "--max-index",
        type=int,
        default=5,
        help="Highest camera index to probe (default: 5).",
    )

    run_parser = subparsers.add_parser(
        "run", help="Open the camera and start the live preview."
    )
    run_parser.add_argument(
        "--camera",
        type=int,
        default=None,
        help="Camera index to use instead of the value in config.toml.",
    )

    export_parser = subparsers.add_parser(
        "export", help="Export presence sessions to a CSV file."
    )
    export_parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output path (default: data/exports/sessions.csv).",
    )
    export_parser.add_argument(
        "--since",
        default=None,
        help="Include sessions that started on or after this date (YYYY-MM-DD, UTC).",
    )
    export_parser.add_argument(
        "--until",
        default=None,
        help="Include sessions that started on or before this date (YYYY-MM-DD, UTC).",
    )

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    try:
        settings = load_settings(args.config)
        if args.command == "setup":
            return _run_setup(settings)
        if args.command == "cameras":
            return _run_cameras(settings, args.max_index)
        if args.command == "run":
            if args.camera is not None and args.camera < 0:
                print("--camera must be zero or greater.")
                return 2
            return run_app(settings, camera_index=args.camera)
        if args.command == "export":
            return _run_export(settings, args.out, args.since, args.until)
    except (
        ConfigError,
        ModelError,
        CameraError,
        DetectionError,
        RecognitionError,
        StorageError,
    ) as exc:
        logger.error("%s", exc)
        return 1

    parser.error(f"Unknown command: {args.command}")
    return 2


def _run_setup(settings: Settings) -> int:
    models = ensure_models(settings.models.directory)
    for path in models:
        print(f"  {path.name}  {path.stat().st_size / 1_000_000:.1f} MB")

    for directory in (
        settings.storage.db_path.parent,
        settings.storage.export_dir,
        settings.storage.captures_dir,
        settings.storage.log_dir,
    ):
        directory.mkdir(parents=True, exist_ok=True)

    database = Database(settings.storage.db_path)
    database.close()
    print(f"Database ready: {settings.storage.db_path}")

    print("Setup complete.")
    return 0


def _run_export(
    settings: Settings,
    output: Path | None,
    since: str | None,
    until: str | None,
) -> int:
    try:
        since_iso = _parse_day(since, end_of_day=False)
        until_iso = _parse_day(until, end_of_day=True)
    except ValueError as exc:
        print(str(exc))
        return 2

    target = output or settings.storage.export_dir / "sessions.csv"
    database = Database(settings.storage.db_path)
    try:
        count = database.export_sessions(target, since_iso, until_iso)
    finally:
        database.close()

    print(f"Exported {count} session(s) to {target}")
    return 0


def _parse_day(value: str | None, end_of_day: bool) -> str | None:
    if value is None:
        return None
    try:
        day = date.fromisoformat(value)
    except ValueError as exc:
        raise ValueError(f"Invalid date '{value}'. Use YYYY-MM-DD.") from exc
    suffix = "T23:59:59Z" if end_of_day else "T00:00:00Z"
    return f"{day.isoformat()}{suffix}"


def _run_cameras(settings: Settings, max_index: int) -> int:
    if max_index < 0:
        print("--max-index must be zero or greater.")
        return 2

    print(f"Scanning camera indexes 0-{max_index}...")
    cameras = probe_cameras(
        max_index=max_index,
        width=settings.camera.width,
        height=settings.camera.height,
        backend=settings.camera.backend,
    )

    if not cameras:
        print("No cameras detected.")
        print(
            "Check that the device is connected and that this application has "
            "camera permission."
        )
        return 1

    print(f"{'Index':<6}{'Backend':<16}Resolution")
    for camera in cameras:
        print(f"{camera.index:<6}{camera.backend:<16}{camera.width}x{camera.height}")
    print()
    print("Set [camera].index in config.toml to the device you want to use.")
    return 0
