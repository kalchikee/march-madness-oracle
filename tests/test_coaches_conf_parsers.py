"""Coaches + conferences parser tests (offline fixtures)."""
from __future__ import annotations

from madness.ingest.coaches import parse_coaches_page
from madness.ingest.conferences import parse_standings

COACHES_FIXTURE = """
<html><body>
<table id="coaches">
  <thead><tr><th>Sch</th><th>Coach</th></tr></thead>
  <tbody>
    <tr>
      <td data-stat="school_name">Duke</td>
      <td data-stat="coach">Jon Scheyer</td>
    </tr>
    <tr>
      <td data-stat="school_name">Kansas</td>
      <td data-stat="coach">Bill Self</td>
    </tr>
  </tbody>
</table>
</body></html>
"""

STANDINGS_FIXTURE = """
<html><body>
<table>
  <caption>Big 12 Conference</caption>
  <tbody>
    <tr><td data-stat="school_name">Kansas</td></tr>
    <tr><td data-stat="school_name">Houston</td></tr>
  </tbody>
</table>
<table>
  <caption>Big East Conference</caption>
  <tbody>
    <tr><td data-stat="school_name">UConn</td></tr>
  </tbody>
</table>
</body></html>
"""


def test_coaches_parser_basic():
    df = parse_coaches_page(COACHES_FIXTURE, season=2024)
    assert len(df) == 2
    assert set(df["team"]) == {"Duke", "Kansas"}
    row = df[df["team"] == "Duke"].iloc[0]
    assert row["coach_name"] == "Jon Scheyer"
    assert row["season"] == 2024


def test_standings_parser_captures_conference_names():
    df = parse_standings(STANDINGS_FIXTURE, season=2024)
    assert len(df) == 3
    conferences = set(df["conference"])
    assert "Big 12 Conference" in conferences
    assert "Big East Conference" in conferences
