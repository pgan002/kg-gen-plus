"""Tests for logging configuration: rotation, role scoping, and graceful fallback.

The fallback cases are the point of this module. Log files created by a process
running under one uid are not appendable by another (the container runs as uid
1000, a developer account usually does not), and the previous behaviour was for
`dictConfig` to raise and take startup down with it.
"""

from __future__ import annotations

import logging
import os
import stat

import pytest
import yaml

from app import logging_setup, settings


@pytest.fixture(autouse=True)
def restore_logging():
    """Undo whatever a test did to the root logger."""
    yield
    logging.shutdown()
    for handler in list(logging.getLogger().handlers):
        logging.getLogger().removeHandler(handler)
    logging.config = logging.config  # keep module reference for clarity


@pytest.fixture
def log_dir(tmp_path, monkeypatch):
    directory = tmp_path / "logs"
    monkeypatch.setattr(settings, "LOG_DIR", str(directory))
    monkeypatch.delenv("KGGEN_LOG_TO_FILES", raising=False)
    return directory


def _file_handlers() -> list[logging.Handler]:
    return [
        h for h in logging.getLogger().handlers if isinstance(h, logging.FileHandler)
    ]


def test_files_are_rotating_handlers_with_configured_limits(log_dir, monkeypatch):
    monkeypatch.setattr(settings, "LOG_MAX_BYTES", 4096)
    monkeypatch.setattr(settings, "LOG_BACKUP_COUNT", 2)

    logging_setup.configure_logging(role="api", files_by_default=True)

    handlers = _file_handlers()
    assert handlers, "expected at least one file handler"
    for handler in handlers:
        assert isinstance(handler, logging.handlers.RotatingFileHandler)
        assert handler.maxBytes == 4096
        assert handler.backupCount == 2


def test_rotation_bounds_total_size_on_disk(log_dir, monkeypatch):
    monkeypatch.setattr(settings, "LOG_MAX_BYTES", 2048)
    monkeypatch.setattr(settings, "LOG_BACKUP_COUNT", 1)

    logging_setup.configure_logging(role="api", files_by_default=True)
    log = logging.getLogger("kg_gen_app")
    for i in range(400):
        log.info("padding %d %s", i, "x" * 100)

    written = sorted(p.name for p in log_dir.glob("main-api.log*"))
    # main + exactly one backup, never a third.
    assert written == ["main-api.log", "main-api.log.1"], written
    total = sum(p.stat().st_size for p in log_dir.glob("main-api.log*"))
    assert total <= 2048 * 2 * 1.1, f"{total} exceeds the rotation ceiling"


def test_role_scopes_filenames_so_processes_do_not_share_a_file(log_dir):
    logging_setup.configure_logging(role="api", files_by_default=True)
    logging.getLogger("kg_gen_app").info("from api")
    logging.shutdown()

    logging_setup.configure_logging(role="worker", files_by_default=True)
    logging.getLogger("kg_gen_app").info("from worker")

    names = {p.name for p in log_dir.glob("*.log")}
    assert "main-api.log" in names
    assert "main-worker.log" in names


# caplog cannot be used below: dictConfig installs its own root handlers and
# removes pytest's, so records emitted after configure_logging never reach
# caplog. Asserting on stdout also checks what an operator actually sees.
@pytest.mark.skipif(os.geteuid() == 0, reason="root can write anywhere")
def test_unwritable_directory_falls_back_to_stdout(tmp_path, monkeypatch, capsys):
    blocked = tmp_path / "blocked"
    blocked.mkdir()
    blocked.chmod(stat.S_IRUSR | stat.S_IXUSR)  # r-x: cannot create files
    monkeypatch.setattr(settings, "LOG_DIR", str(blocked))
    monkeypatch.delenv("KGGEN_LOG_TO_FILES", raising=False)

    try:
        logging_setup.configure_logging(role="api", files_by_default=True)
        assert not _file_handlers()
        # Still usable, and no per-record traceback.
        logging.getLogger("kg_gen_app").info("survived")
        out = capsys.readouterr().out
        assert "File logging disabled" in out
        assert "survived" in out
        assert "Traceback" not in out
    finally:
        blocked.chmod(stat.S_IRWXU)


@pytest.mark.skipif(os.geteuid() == 0, reason="root can append to anything")
def test_unappendable_existing_file_falls_back_instead_of_raising(log_dir, capsys):
    """The original bug: a leftover file owned by another uid broke startup."""
    log_dir.mkdir(parents=True, exist_ok=True)
    squatter = log_dir / "main-api.log"
    squatter.touch()
    squatter.chmod(0)  # stands in for "owned by a different uid"

    try:
        logging_setup.configure_logging(role="api", files_by_default=True)
        assert not _file_handlers()
        logging.getLogger("kg_gen_app").info("survived")
        out = capsys.readouterr().out
        assert "cannot append to" in out
        # The old failure mode was a PermissionError traceback per record.
        assert "Traceback" not in out
        assert "survived" in out
    finally:
        squatter.chmod(stat.S_IRWXU)


def test_files_disabled_writes_nothing(log_dir):
    logging_setup.configure_logging(role="worker", files_by_default=False)
    logging.getLogger("kg_gen_app").info("stdout only")

    assert not _file_handlers()
    assert not log_dir.exists(), "directory should not even be created"


def test_env_var_overrides_the_per_process_default(log_dir, monkeypatch):
    monkeypatch.setenv("KGGEN_LOG_TO_FILES", "1")
    logging_setup.configure_logging(role="worker", files_by_default=False)
    assert _file_handlers()


def test_yaml_declares_bare_filenames_not_paths():
    """logging.yaml must not hard-code a directory; the dir comes from settings."""
    with open(logging_setup._CONFIG_PATH) as f:
        config = yaml.safe_load(f.read())

    for name, handler in config["handlers"].items():
        if str(handler.get("class", "")).endswith("FileHandler"):
            filename = handler["filename"]
            assert "/" not in filename, f"{name} should not embed a path: {filename}"
