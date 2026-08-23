"""Guard: CLAUDE.md's endpoint table must match the app's real routes.

The table is maintained by hand, so it silently drifts every time a route
is added — which is how it came to be missing 34 endpoints. This test
fails the moment the two disagree, in either direction:

* a route in the code but not the table (the usual drift), and
* a row in the table for a route that no longer exists (a stale promise,
  worse than an omission because it reads as authoritative).

When it fails, fix the table — the code is the source of truth.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

CLAUDE_MD = Path(__file__).resolve().parent.parent / "CLAUDE.md"

ROW_RE = re.compile(r"\|\s*(GET|POST|PUT|PATCH|DELETE)\s*\|\s*(/api[^\s|]*)\s*\|")


def normalize(path: str) -> str:
    """Collapse path params so {id} and {job_id} compare equal."""
    return re.sub(r"\{[^}]+\}", "{}", path.rstrip("/"))


def documented_routes() -> set[tuple[str, str]]:
    rows: set[tuple[str, str]] = set()
    for line in CLAUDE_MD.read_text(encoding="utf-8").splitlines():
        m = ROW_RE.match(line)
        if m:
            rows.add((m.group(1), normalize(m.group(2))))
    return rows


def real_routes() -> set[tuple[str, str]]:
    from backend.main import app

    rows: set[tuple[str, str]] = set()
    for path, ops in app.openapi()["paths"].items():
        if not path.startswith("/api"):
            continue
        for method in ops:
            upper = method.upper()
            if upper in ("HEAD", "OPTIONS"):
                continue
            rows.add((upper, normalize(path)))
    return rows


def test_no_undocumented_endpoints():
    missing = sorted(real_routes() - documented_routes())
    assert not missing, (
        "Endpoints exist in code but are absent from the CLAUDE.md endpoint "
        "table:\n" + "\n".join(f"  {m} {p}" for m, p in missing)
    )


def test_no_phantom_endpoints():
    phantom = sorted(documented_routes() - real_routes())
    assert not phantom, (
        "CLAUDE.md documents endpoints that do not exist:\n"
        + "\n".join(f"  {m} {p}" for m, p in phantom)
    )


def test_table_has_no_duplicate_rows():
    counts: dict[tuple[str, str], int] = {}
    for line in CLAUDE_MD.read_text(encoding="utf-8").splitlines():
        m = ROW_RE.match(line)
        if m:
            key = (m.group(1), normalize(m.group(2)))
            counts[key] = counts.get(key, 0) + 1
    dupes = sorted(k for k, v in counts.items() if v > 1)
    assert not dupes, (
        "Duplicate rows in the endpoint table:\n"
        + "\n".join(f"  {m} {p}" for m, p in dupes)
    )
