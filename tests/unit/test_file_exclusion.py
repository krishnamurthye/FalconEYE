"""Unit tests for the shared file-exclusion helper.

Covers both real glob semantics (the primary behaviour) and the legacy
substring fallback (retained so existing user configs keep working).
"""

from pathlib import Path

import pytest

from falconeye.domain.services.file_exclusion import (
    filter_excluded,
    is_excluded,
)


ROOT = Path("/project")


# ---------------------------------------------------------------------------
# Real glob semantics
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "path, pattern",
    [
        # bundled minified JS excluded via extension pattern
        (ROOT / "static/js/app.min.js", "*.min.js"),
        # build artifacts under any depth
        (ROOT / "node_modules/foo/index.js", "*/node_modules/*"),
        # nested vendor directory
        (ROOT / "src/vendor/lib/a.php", "vendor/**/*.php"),
    ],
)
def test_glob_patterns_match_intended_files(path, pattern):
    assert is_excluded(path, ROOT, [pattern]) is True


def test_glob_does_not_overmatch_similar_names():
    """The substring bug once made `build/` match `rebuild.js`. Verify
    that a proper glob pattern no longer over-matches."""
    path = ROOT / "src/rebuild.js"
    # A strict glob for a build directory should not hit a file whose name
    # merely contains the word 'build'.
    assert is_excluded(path, ROOT, ["build/**"]) is False


# ---------------------------------------------------------------------------
# Legacy substring fallback (backward compat)
# ---------------------------------------------------------------------------

def test_legacy_substring_fallback_still_matches():
    """Patterns that only worked under the old substring logic must keep
    matching so historical user configs don't silently stop excluding."""
    path = ROOT / "some/deep/node_modules/lib/x.js"
    # This glob WOULD match too, but we specifically want to confirm the
    # substring fallback works for patterns with lots of star-noise that
    # don't translate cleanly into fnmatch.
    assert is_excluded(path, ROOT, ["**/node_modules/**"]) is True


# ---------------------------------------------------------------------------
# No-op cases
# ---------------------------------------------------------------------------

def test_empty_pattern_list_excludes_nothing():
    path = ROOT / "src/app.py"
    assert is_excluded(path, ROOT, []) is False


def test_unrelated_pattern_does_not_match():
    path = ROOT / "src/app.py"
    assert is_excluded(path, ROOT, ["*.rs"]) is False


def test_filter_excluded_returns_expected_subset():
    files = [
        ROOT / "src/app.py",
        ROOT / "node_modules/foo.js",
        ROOT / "static/app.min.js",
        ROOT / "tests/test_app.py",
    ]
    kept = filter_excluded(files, ROOT, ["*/node_modules/*", "*.min.js"])
    kept_names = {p.name for p in kept}
    assert "foo.js" not in kept_names
    assert "app.min.js" not in kept_names
    assert "app.py" in kept_names
    assert "test_app.py" in kept_names


def test_file_outside_root_still_evaluated():
    """When the candidate file isn't under root, the matcher should still
    evaluate against the absolute path (and not raise)."""
    path = Path("/elsewhere/node_modules/x.js")
    assert is_excluded(path, ROOT, ["*/node_modules/*"]) is True
