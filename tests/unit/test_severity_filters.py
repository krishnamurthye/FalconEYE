"""Unit tests for the --severity and --fail-on threshold helpers."""

import pytest

from falconeye.domain.services.severity_filter import (
    SEVERITY_RANK as _SEVERITY_RANK,
    filter_findings_by_min_severity as _filter_findings_by_min_severity,
    parse_severity_threshold as _parse_severity_threshold,
)
from falconeye.domain.models.security import Severity


# ---------------------------------------------------------------------------
# _parse_severity_threshold
# ---------------------------------------------------------------------------

def test_parse_none_returns_none():
    assert _parse_severity_threshold(None) is None


def test_parse_empty_string_returns_none():
    assert _parse_severity_threshold("") is None


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("critical", Severity.CRITICAL),
        ("HIGH", Severity.HIGH),
        ("  medium  ", Severity.MEDIUM),
        ("low", Severity.LOW),
        ("info", Severity.INFO),
    ],
)
def test_parse_accepts_case_and_whitespace(raw, expected):
    assert _parse_severity_threshold(raw) is expected


def test_parse_rejects_unknown_value():
    with pytest.raises(ValueError) as exc:
        _parse_severity_threshold("catastrophic")
    # Error message should list all valid options so the user knows what to type.
    msg = str(exc.value)
    for s in ("critical", "high", "medium", "low", "info"):
        assert s in msg


# ---------------------------------------------------------------------------
# _SEVERITY_RANK ordering
# ---------------------------------------------------------------------------

def test_severity_rank_is_strictly_monotonic():
    order = [
        Severity.INFO, Severity.LOW, Severity.MEDIUM,
        Severity.HIGH, Severity.CRITICAL,
    ]
    ranks = [_SEVERITY_RANK[s] for s in order]
    assert ranks == sorted(ranks)
    assert len(set(ranks)) == len(ranks)


# ---------------------------------------------------------------------------
# _filter_findings_by_min_severity
# ---------------------------------------------------------------------------

def _make_finding(severity):
    # Lazy-built so we only import the domain type inside the test helper.
    from falconeye.domain.models.security import (
        FindingConfidence, SecurityFinding,
    )
    return SecurityFinding.create(
        issue="x",
        reasoning="y",
        mitigation="z",
        severity=severity,
        confidence=FindingConfidence.MEDIUM,
        file_path="f.py",
        code_snippet="",
    )


def test_filter_none_threshold_returns_input_unchanged():
    findings = [_make_finding(s) for s in Severity]
    assert _filter_findings_by_min_severity(findings, None) is findings


def test_filter_high_keeps_high_and_critical_only():
    findings = [_make_finding(s) for s in Severity]
    kept = _filter_findings_by_min_severity(findings, Severity.HIGH)
    kept_severities = {f.severity for f in kept}
    assert kept_severities == {Severity.HIGH, Severity.CRITICAL}


def test_filter_info_keeps_everything():
    findings = [_make_finding(s) for s in Severity]
    kept = _filter_findings_by_min_severity(findings, Severity.INFO)
    assert len(kept) == len(findings)


def test_filter_critical_keeps_only_critical():
    findings = [_make_finding(s) for s in Severity]
    kept = _filter_findings_by_min_severity(findings, Severity.CRITICAL)
    assert [f.severity for f in kept] == [Severity.CRITICAL]


def test_filter_empty_list_returns_empty_list():
    assert _filter_findings_by_min_severity([], Severity.HIGH) == []
