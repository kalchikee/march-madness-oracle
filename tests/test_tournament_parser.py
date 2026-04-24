"""Tournament-page parser test against a minimal fixture."""
from __future__ import annotations

from madness.ingest.sports_reference import parse_tournament_page

FIXTURE = """
<html><body>
<div id="brackets">
  <div id="east" class="current">
    <div id="bracket">
      <div class="round">
        <div>
          <div class="winner">
            <span>1</span>
            <a href="/cbb/schools/connecticut/men/2024.html">UConn</a>
            <a href="/cbb/boxscores/2024-03-22-14-connecticut.html">91</a>
          </div>
          <div>
            <span>16</span>
            <a href="/cbb/schools/stetson/men/2024.html">Stetson</a>
            <a href="/cbb/boxscores/2024-03-22-14-connecticut.html">52</a>
          </div>
          <span><a href="/cbb/boxscores/2024-03-22-14-connecticut.html">at Brooklyn, NY</a></span>
        </div>
      </div>
    </div>
  </div>
  <div id="national">
    <div id="bracket">
      <div class="round">
        <div>
          <div class="winner">
            <span>1</span>
            <a href="/cbb/schools/connecticut/men/2024.html">UConn</a>
            <a href="/cbb/boxscores/2024-04-08-21-connecticut.html">75</a>
          </div>
          <div>
            <span>1</span>
            <a href="/cbb/schools/purdue/men/2024.html">Purdue</a>
            <a href="/cbb/boxscores/2024-04-08-21-connecticut.html">60</a>
          </div>
          <span><a href="/cbb/boxscores/2024-04-08-21-connecticut.html">at Glendale, AZ</a></span>
        </div>
      </div>
    </div>
  </div>
</div>
</body></html>
"""


def test_parses_r64_game():
    games = parse_tournament_page(FIXTURE, season=2024)
    r64 = [g for g in games if g.round_name == "Round of 64"]
    assert len(r64) == 1
    g = r64[0]
    assert g.team_winner == "UConn"
    assert g.team_loser == "Stetson"
    assert g.seed_winner == 1
    assert g.seed_loser == 16
    assert g.score_winner == 91
    assert g.score_loser == 52
    assert g.region == "east"
    assert g.date == "2024-03-22"


def test_parses_final_four_as_championship_with_no_region():
    games = parse_tournament_page(FIXTURE, season=2024)
    nat = [g for g in games if g.region is None]
    assert len(nat) == 1
    g = nat[0]
    # In the fixture "national" has only one round div, so it's indexed as "Final Four"
    assert g.round_name == "Final Four"
    assert g.team_winner == "UConn"
    assert g.score_winner == 75


def test_handles_empty_html():
    assert parse_tournament_page("<html></html>", season=2024) == []
