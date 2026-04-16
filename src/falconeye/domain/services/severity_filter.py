"""Severity-threshold helpers shared by --severity and --fail-on.

Extracted from the CLI command layer so they can be unit-tested without
transitively importing infrastructure adapters (chromadb, ollama, etc.).
"""

from typing import List, Optional

from ..models.security import Severity, SecurityFinding


# Severity ordering: higher value = more severe. Used by both the
# --severity (minimum-to-report) filter and the --fail-on (CI gate) check.
SEVERITY_RANK: dict[Severity, int] = {
    Severity.INFO: 0,
    Severity.LOW: 1,
    Severity.MEDIUM: 2,
    Severity.HIGH: 3,
    Severity.CRITICAL: 4,
}


def parse_severity_threshold(value: Optional[str]) -> Optional[Severity]:
    """Parse a severity threshold string into a Severity enum.

    Accepts case-insensitive strings: critical, high, medium, low, info.
    Returns None if ``value`` is None or empty.

    Raises:
        ValueError: If ``value`` is not a recognized severity.
    """
    if not value:
        return None
    normalized = value.strip().lower()
    try:
        return Severity(normalized)
    except ValueError as e:
        valid = ", ".join(s.value for s in Severity)
        raise ValueError(
            f"Invalid severity '{value}'. Must be one of: {valid}"
        ) from e


def filter_findings_by_min_severity(
    findings: List[SecurityFinding],
    min_severity: Optional[Severity],
) -> List[SecurityFinding]:
    """Return findings at or above the minimum severity.

    If ``min_severity`` is None, returns the input list unchanged.
    """
    if min_severity is None:
        return findings
    threshold = SEVERITY_RANK[min_severity]
    return [f for f in findings if SEVERITY_RANK.get(f.severity, 0) >= threshold]
