"""Sports-Reference schedule parser — unit tests against fixture HTML.

These run offline against a representative snippet so CI is not
dependent on the upstream site being reachable.
"""
from __future__ import annotations

from madness.ingest.sports_reference import parse_schedule_page

SCHEDULE_FIXTURE = """
<html><body>
<table id="schedule">
  <thead><tr><th>Date</th><th>W</th><th>W Pts</th><th>Loc</th><th>L</th><th>L Pts</th><th>OT</th></tr></thead>
  <tbody>
    <tr>
      <th data-stat="date_game">2023-11-06</th>
      <td data-stat="winner_school_name">Kansas</td>
      <td data-stat="winner_pts">95</td>
      <td data-stat="game_location"></td>
      <td data-stat="loser_school_name">NC Central</td>
      <td data-stat="loser_pts">68</td>
      <td data-stat="overtimes"></td>
    </tr>
    <tr>
      <th data-stat="date_game">2023-11-14</th>
      <td data-stat="winner_school_name">Duke</td>
      <td data-stat="winner_pts">74</td>
      <td data-stat="game_location">N</td>
      <td data-stat="loser_school_name">Michigan State</td>
      <td data-stat="loser_pts">65</td>
      <td data-stat="overtimes"></td>
    </tr>
    <tr>
      <th data-stat="date_game">2023-12-02</th>
      <td data-stat="winner_school_name">North Carolina</td>
      <td data-stat="winner_pts">77</td>
      <td data-stat="game_location">@</td>
      <td data-stat="loser_school_name">Kentucky</td>
      <td data-stat="loser_pts">73</td>
      <td data-stat="overtimes">OT</td>
    </tr>
  </tbody>
</table>
</body></html>
"""


def test_parse_schedule_three_games():
    games = parse_schedule_page(SCHEDULE_FIXTURE, season=2024)
    assert len(games) == 3


def test_parse_schedule_home_winner():
    games = parse_schedule_page(SCHEDULE_FIXTURE, season=2024)
    g = games[0]
    assert g.team_winner == "Kansas"
    assert g.team_loser == "NC Central"
    assert g.score_winner == 95
    assert g.score_loser == 68
    assert g.site_neutral is False
    assert g.site_home == "Kansas"  # blank location = winner hosted


def test_parse_schedule_neutral():
    games = parse_schedule_page(SCHEDULE_FIXTURE, season=2024)
    g = games[1]
    assert g.site_neutral is True
    assert g.site_home == ""


def test_parse_schedule_at_away():
    games = parse_schedule_page(SCHEDULE_FIXTURE, season=2024)
    g = games[2]
    assert g.site_neutral is False
    assert g.site_home == "Kentucky"  # @ means winner was away, loser hosted
    assert g.overtime is True


def test_parse_schedule_empty_html_returns_empty_list():
    assert parse_schedule_page("<html></html>", season=2024) == []
