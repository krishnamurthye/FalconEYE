"""Shared file-exclusion helpers.

Centralizes glob-pattern matching for file discovery so index, review, and
scan paths all use the same exclusion semantics. Historically each call site
did ``pattern.replace("**", "").replace("*", "")`` and substring-matched the
result, which over-matched (e.g. ``build/`` matched ``rebuild.js``) and
under-matched real glob patterns. This helper fixes that without removing
the legacy behaviour: real glob patterns are tried first, then the legacy
substring match is used as a fallback so configs that relied on it keep
working.
"""

from fnmatch import fnmatch
from pathlib import Path, PurePosixPath
from typing import Iterable, List


def _iter_tail_subpaths(path_str: str) -> Iterable[str]:
    """Yield progressively shorter tail-suffixes of a POSIX-style path.

    For ``src/vendor/lib/a.php`` yields:
        ``src/vendor/lib/a.php``, ``vendor/lib/a.php``,
        ``lib/a.php``, ``a.php``

    This lets gitignore-style patterns anchored at a middle segment
    (e.g. ``vendor/**/*.php``) match a file that happens to live under
    a deeper prefix (``src/vendor/...``), matching the intuitive behaviour
    users expect when configuring an exclusion list.
    """
    # Normalize separators so fnmatch sees POSIX paths regardless of OS.
    normalized = path_str.replace("\\", "/")
    parts = normalized.split("/")
    # Skip the empty leading segment that comes from absolute paths ("/a/b").
    if parts and parts[0] == "":
        parts = parts[1:]
    for i in range(len(parts)):
        yield "/".join(parts[i:])


def _pattern_matches(pattern: str, relative_path: str, absolute_path: str) -> bool:
    """Return True when ``pattern`` matches either the relative or absolute path.

    Matching order:
    1. fnmatch on the relative path, then every tail-subpath of it
       (handles ``*.min.js``, ``*/node_modules/*``, ``vendor/**/*.php``
       matching a file under ``src/vendor/...``).
    2. Same for the absolute path.
    3. Legacy substring fallback (strip ``*``/``**`` and check both paths).

    This belt-and-braces approach means the stricter glob semantics are used
    when patterns are well-formed, while loosely-written patterns that only
    worked under the old substring logic continue to match.
    """
    # Real glob semantics, anchored at every intermediate directory.
    for sub in _iter_tail_subpaths(relative_path):
        if fnmatch(sub, pattern):
            return True
    for sub in _iter_tail_subpaths(absolute_path):
        if fnmatch(sub, pattern):
            return True

    # Legacy substring fallback (preserves historical behaviour for patterns
    # like "*/node_modules/*" that users may have configured assuming
    # substring semantics).
    legacy_fragment = pattern.replace("**", "").replace("*", "")
    if legacy_fragment and (
        legacy_fragment in relative_path or legacy_fragment in absolute_path
    ):
        return True

    return False


def is_excluded(
    file_path: Path,
    root_path: Path,
    excluded_patterns: Iterable[str],
) -> bool:
    """Return True if ``file_path`` should be excluded under any pattern.

    Args:
        file_path: Absolute or root-relative path to the candidate file.
        root_path: Project root the ``relative_to`` check is anchored at.
        excluded_patterns: Iterable of glob / substring exclusion patterns.
    """
    try:
        relative_path = str(file_path.relative_to(root_path))
    except ValueError:
        # file_path isn't under root_path; compare against the raw path
        relative_path = str(file_path)
    absolute_path = str(file_path)

    return any(
        _pattern_matches(p, relative_path, absolute_path) for p in excluded_patterns
    )


def filter_excluded(
    file_paths: Iterable[Path],
    root_path: Path,
    excluded_patterns: Iterable[str],
) -> List[Path]:
    """Return the subset of ``file_paths`` that pass all exclusion patterns.

    Materializes ``excluded_patterns`` into a tuple once so the iterable can
    be re-used per-file without exhausting a one-shot iterator.
    """
    patterns = tuple(excluded_patterns)
    return [
        fp for fp in file_paths
        if not is_excluded(fp, root_path, patterns)
    ]
