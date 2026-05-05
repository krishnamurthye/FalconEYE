"""Unit tests for parallel file and document indexing in IndexCodebaseHandler.

The handler used to iterate files one at a time; the parallel-indexing
patch wraps both loops in asyncio.gather gated by a Semaphore whose size
is read from FALCONEYE_INDEX_CONCURRENCY (default 8).

These tests build a handler via __new__ to bypass the heavy DI graph in
__init__, mock the collaborators that handle() calls, and replace
_process_file / _process_document with trackers that record concurrency.
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from falconeye.application.commands.index_codebase import (
    IndexCodebaseCommand,
    IndexCodebaseHandler,
)
from falconeye.domain.value_objects.project_metadata import ProjectType


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_handler() -> IndexCodebaseHandler:
    """Build a handler with stubbed collaborators — enough for handle() to
    reach the parallel-dispatch loop without a real DI graph."""
    handler = IndexCodebaseHandler.__new__(IndexCodebaseHandler)
    handler.logger = logging.getLogger("test.index_parallel")

    handler.project_identifier = MagicMock()
    handler.project_identifier.identify_project.return_value = (
        "proj-123",
        "test_proj",
        ProjectType.NON_GIT,
        None,
    )

    handler.language_detector = MagicMock()
    handler.language_detector.detect_language.return_value = "python"
    handler.language_detector.detect_all_languages.return_value = ["python"]

    handler.index_registry = MagicMock()
    handler.index_registry.get_project.return_value = None  # first-time path

    handler.vector_store = AsyncMock()
    handler.metadata_repo = AsyncMock()
    handler.llm_service = AsyncMock()
    handler.ast_analyzer = MagicMock()
    handler.checksum_service = MagicMock()

    return handler


def _seed_files(tmp_path: Path, n: int, prefix: str = "f") -> list[Path]:
    files: list[Path] = []
    for i in range(n):
        p = tmp_path / f"{prefix}{i}.py"
        p.write_text(f"# {prefix} {i}\n")
        files.append(p)
    return files


# ---------------------------------------------------------------------------
# Files are dispatched in parallel
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.asyncio
async def test_files_processed_concurrently(monkeypatch, tmp_path):
    """Multiple files should be in flight at the same time, not serial."""
    monkeypatch.setenv("FALCONEYE_INDEX_CONCURRENCY", "8")

    handler = _make_handler()
    files = _seed_files(tmp_path, 5)
    handler._discover_files = MagicMock(return_value=files)
    handler._discover_documents = MagicMock(return_value=[])

    in_flight = 0
    peak = 0

    async def fake_process(*args, **kwargs):
        nonlocal in_flight, peak
        in_flight += 1
        peak = max(peak, in_flight)
        await asyncio.sleep(0.05)
        in_flight -= 1
        return None

    handler._process_file = fake_process

    cmd = IndexCodebaseCommand(
        codebase_path=tmp_path,
        language="python",
        force_reindex=True,
        include_documents=False,
    )
    await handler.handle(cmd)

    assert peak >= 2, f"Expected concurrent execution, peak in-flight = {peak}"


# ---------------------------------------------------------------------------
# Concurrency cap honours the env var
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.asyncio
async def test_concurrency_capped_via_env_var(monkeypatch, tmp_path):
    """FALCONEYE_INDEX_CONCURRENCY=N caps the number of in-flight tasks."""
    monkeypatch.setenv("FALCONEYE_INDEX_CONCURRENCY", "2")

    handler = _make_handler()
    files = _seed_files(tmp_path, 8)
    handler._discover_files = MagicMock(return_value=files)
    handler._discover_documents = MagicMock(return_value=[])

    in_flight = 0
    peak = 0

    async def fake_process(*args, **kwargs):
        nonlocal in_flight, peak
        in_flight += 1
        peak = max(peak, in_flight)
        await asyncio.sleep(0.05)
        in_flight -= 1
        return None

    handler._process_file = fake_process

    cmd = IndexCodebaseCommand(
        codebase_path=tmp_path,
        language="python",
        force_reindex=True,
        include_documents=False,
    )
    await handler.handle(cmd)

    assert peak == 2, (
        f"Expected concurrency capped at 2 with 8 files, but peak = {peak}"
    )


# ---------------------------------------------------------------------------
# One bad file does not abort the others
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.asyncio
async def test_failed_file_does_not_abort_others(monkeypatch, tmp_path):
    """If one file's _process_file returns None (the existing failure path),
    asyncio.gather must still wait for the other files to finish."""
    monkeypatch.setenv("FALCONEYE_INDEX_CONCURRENCY", "8")

    handler = _make_handler()
    files = _seed_files(tmp_path, 4)
    handler._discover_files = MagicMock(return_value=files)
    handler._discover_documents = MagicMock(return_value=[])

    completed: list[Path] = []

    async def fake_process(file_path, *args, **kwargs):
        if file_path.name == "f1.py":
            return None  # mimics _process_file's exception-swallow path
        completed.append(file_path)
        return MagicMock(chunk_count=3)

    handler._process_file = fake_process

    cmd = IndexCodebaseCommand(
        codebase_path=tmp_path,
        language="python",
        force_reindex=True,
        include_documents=False,
    )
    await handler.handle(cmd)

    assert len(completed) == 3
    assert all(fp.name != "f1.py" for fp in completed)


# ---------------------------------------------------------------------------
# Document loop is parallelised the same way
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.asyncio
async def test_documents_processed_concurrently(monkeypatch, tmp_path):
    """The document loop should also dispatch concurrently when enabled."""
    monkeypatch.setenv("FALCONEYE_INDEX_CONCURRENCY", "8")

    handler = _make_handler()
    handler._discover_files = MagicMock(return_value=[])
    docs = _seed_files(tmp_path, 4, prefix="doc")
    handler._discover_documents = MagicMock(return_value=docs)

    in_flight = 0
    peak = 0

    async def fake_process_doc(*args, **kwargs):
        nonlocal in_flight, peak
        in_flight += 1
        peak = max(peak, in_flight)
        await asyncio.sleep(0.05)
        in_flight -= 1

    handler._process_document = fake_process_doc

    cmd = IndexCodebaseCommand(
        codebase_path=tmp_path,
        language="python",
        force_reindex=True,
        include_documents=True,
    )
    await handler.handle(cmd)

    assert peak >= 2, f"Expected document loop to be parallel, peak = {peak}"
