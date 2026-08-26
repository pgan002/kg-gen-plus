"""Logging configuration shared by the API and the worker.

``logging.yaml`` stays the declarative source of truth for formatters, levels and
which logger writes where. This module supplies the parts that depend on the
runtime environment:

* **where** the files go (``KGGEN_LOG_DIR``) rather than a relative ``logs/``
  resolved against whatever the current working directory happens to be;
* **rotation**, so files cannot grow without bound;
* **whether files are written at all** -- and a soft fallback to stdout when the
  directory is not writable, instead of `dictConfig` raising and taking the whole
  process down at startup.

That last point is not hypothetical: the container runs as uid 1000 while a
developer's account is typically some other uid, so log files created by one are
not appendable by the other. Losing file logs is an acceptable degradation;
refusing to boot is not.
"""

from __future__ import annotations

import logging
import logging.config
import os
from pathlib import Path
from typing import Any

import yaml

from app import settings

_CONFIG_PATH = Path(__file__).parent / "logging.yaml"


def _probe_writable(directory: Path, targets: list[Path]) -> str | None:
    """Return None if logging to ``targets`` will work, else why not.

    Both checks matter and fail differently. The directory has to be creatable and
    writable (else the handler cannot make its file), and each *existing* target
    has to be appendable -- a file left behind by a process running under a
    different uid is writable-by-nobody-else even in a world-writable directory.
    Checking only the directory is how you end up with a handler that raises on
    every single record instead of degrading once, at startup.
    """
    try:
        directory.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        return f"cannot create {directory}: {exc}"

    probe = directory / ".write-probe"
    try:
        with open(probe, "a"):
            pass
        probe.unlink(missing_ok=True)
    except OSError as exc:
        return f"cannot write in {directory}: {exc}"

    for target in targets:
        if not target.exists():
            continue
        try:
            with open(target, "a"):
                pass
        except OSError as exc:
            return f"cannot append to {target}: {exc}"
    return None


def _file_handler_names(config: dict[str, Any]) -> list[str]:
    return [
        name
        for name, handler in config.get("handlers", {}).items()
        if str(handler.get("class", "")).endswith("FileHandler")
    ]


def _drop_handlers(config: dict[str, Any], names: list[str]) -> None:
    """Remove handlers and every reference to them from loggers/root."""
    for name in names:
        config.get("handlers", {}).pop(name, None)

    dropped = set(names)
    for logger_config in list(config.get("loggers", {}).values()) + [
        config.get("root", {})
    ]:
        handlers = logger_config.get("handlers")
        if handlers:
            logger_config["handlers"] = [h for h in handlers if h not in dropped]


def _resolve_filenames(
    config: dict[str, Any], directory: Path, role: str
) -> dict[str, Path]:
    """Map each file handler to the absolute path it should write.

    The role is mixed into the filename so an API process and a worker sharing one
    log volume do not write to (and rotate) the same file.
    """
    resolved: dict[str, Path] = {}
    for name in _file_handler_names(config):
        handler = config["handlers"][name]
        stem = Path(str(handler.get("filename", f"{name}.log"))).name
        if role:
            stem = f"{Path(stem).stem}-{role}{Path(stem).suffix or '.log'}"
        resolved[name] = directory / stem
    return resolved


def _apply_rotation(config: dict[str, Any], filenames: dict[str, Path]) -> None:
    """Point file handlers at their resolved paths and make them rotate."""
    for name, path in filenames.items():
        handler = config["handlers"][name]
        handler["class"] = "logging.handlers.RotatingFileHandler"
        handler["filename"] = str(path)
        handler["maxBytes"] = settings.LOG_MAX_BYTES
        handler["backupCount"] = settings.LOG_BACKUP_COUNT
        handler.setdefault("encoding", "utf-8")
        # delay=True so opening the file is deferred to the first record. Handlers
        # belonging to loggers that never fire then never create an empty file.
        handler["delay"] = True
        handler.pop("mode", None)


def configure_logging(role: str = "", *, files_by_default: bool = True) -> None:
    """Configure logging for this process.

    Args:
        role: short process label ("api", "worker", ...) mixed into log filenames
            so co-located processes keep separate files. Empty means no suffix.
        files_by_default: whether this process writes log files when
            ``KGGEN_LOG_TO_FILES`` is not set explicitly.
    """
    with open(_CONFIG_PATH) as f:
        config = yaml.safe_load(f.read())

    problems: list[str] = []
    want_files = settings.log_to_files(files_by_default)

    if want_files:
        directory = Path(settings.LOG_DIR).expanduser()
        filenames = _resolve_filenames(config, directory, role)
        reason = _probe_writable(directory, list(filenames.values()))
        if reason is None:
            _apply_rotation(config, filenames)
        else:
            problems.append(reason)
            want_files = False

    if not want_files:
        _drop_handlers(config, _file_handler_names(config))

    logging.config.dictConfig(config)

    logger = logging.getLogger(__name__)
    for problem in problems:
        logger.warning("File logging disabled (%s); logging to stdout only", problem)
    if want_files:
        logger.info(
            "Logging to stdout and rotating files in %s "
            "(max %d bytes x %d backups per file)",
            Path(settings.LOG_DIR).expanduser(),
            settings.LOG_MAX_BYTES,
            settings.LOG_BACKUP_COUNT,
        )
    elif not problems:
        logger.info("Logging to stdout only (KGGEN_LOG_TO_FILES is off)")


def describe() -> str:
    """One-line summary of the effective logging destination, for diagnostics."""
    if settings.log_to_files(True):
        return f"stdout + {os.path.join(settings.LOG_DIR, '*.log')} (rotating)"
    return "stdout only"
