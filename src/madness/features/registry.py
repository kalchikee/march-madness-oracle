"""Feature registry — tracks availability per season.

Every feature declares `available_since` so the feature builder can
emit a sparse table and the model layer can filter to feature-eligible
time windows. This is how we avoid leakage-by-imputation.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from madness.config import (
    ADVANCED_STATS_FROM,
    KENPOM_AVAILABLE_FROM,
    MASSEY_AVAILABLE_FROM,
    TORVIK_AVAILABLE_FROM,
    TOURNAMENT_EXPANSION_64,
    TOURNAMENT_EXPANSION_68,
)


@dataclass(frozen=True)
class FeatureSpec:
    name: str
    available_since: int
    group: str
    description: str


REGISTRY: dict[str, FeatureSpec] = {}


def register(spec: FeatureSpec) -> FeatureSpec:
    REGISTRY[spec.name] = spec
    return spec


# --- Team-season ---
register(FeatureSpec("wins", 1975, "team_season", "Regular-season wins"))
register(FeatureSpec("losses", 1975, "team_season", "Regular-season losses"))
register(FeatureSpec("win_pct", 1975, "team_season", "Regular-season win pct"))
register(FeatureSpec("point_diff_per_game", 1975, "team_season", "Avg margin"))
register(FeatureSpec("srs", 1975, "team_season", "Simple Rating System"))
register(FeatureSpec("sos", 1975, "team_season", "Strength of schedule"))
register(FeatureSpec("efg_pct", ADVANCED_STATS_FROM, "four_factors", "Effective FG%"))
register(FeatureSpec("tov_pct", ADVANCED_STATS_FROM, "four_factors", "Turnover rate"))
register(FeatureSpec("orb_pct", ADVANCED_STATS_FROM, "four_factors", "Offensive rebound rate"))
register(FeatureSpec("ft_rate", ADVANCED_STATS_FROM, "four_factors", "FT attempts / FG attempts"))
register(FeatureSpec("adj_oe", KENPOM_AVAILABLE_FROM, "kenpom", "KenPom adjusted offense"))
register(FeatureSpec("adj_de", KENPOM_AVAILABLE_FROM, "kenpom", "KenPom adjusted defense"))
register(FeatureSpec("adj_tempo", KENPOM_AVAILABLE_FROM, "kenpom", "KenPom adjusted tempo"))
register(FeatureSpec("luck", KENPOM_AVAILABLE_FROM, "kenpom", "KenPom luck"))
register(FeatureSpec("torvik_barthag", TORVIK_AVAILABLE_FROM, "torvik", "Torvik Barthag"))
register(FeatureSpec("massey_rating", MASSEY_AVAILABLE_FROM, "massey", "Massey composite"))

# --- Momentum / late-season ---
register(FeatureSpec("last10_wins", 1975, "momentum", "Wins in last 10 games before tournament"))
register(FeatureSpec("last10_point_diff", 1975, "momentum", "Avg margin last 10 games"))
register(FeatureSpec("longest_win_streak", 1975, "momentum", "Longest win streak of season"))
register(FeatureSpec("conf_tourney_result", 1975, "momentum", "Conf tourney round reached"))
register(FeatureSpec("late_weighted_margin", 1975, "momentum", "Time-decay weighted margin (novel)"))
register(FeatureSpec("late_weighted_winpct", 1975, "momentum", "Time-decay weighted win pct (novel)"))
register(FeatureSpec("feb_mar_games", 1975, "momentum", "Games played Feb 1 onwards"))
register(FeatureSpec("feb_mar_winpct", 1975, "momentum", "Feb-Mar-only win pct (novel)"))
register(FeatureSpec("late_sos", 1975, "momentum", "Late-season SoS (opp-strength decay weighted, novel)"))

# --- Site splits (novel) ---
register(FeatureSpec("home_winpct", 1975, "site_splits", "Home games win pct"))
register(FeatureSpec("away_winpct", 1975, "site_splits", "Road games win pct"))
register(FeatureSpec("neutral_winpct", 1975, "site_splits", "Neutral-site win pct (most relevant to tournament)"))
register(FeatureSpec("home_margin", 1975, "site_splits", "Home avg margin"))
register(FeatureSpec("away_margin", 1975, "site_splits", "Road avg margin"))
register(FeatureSpec("neutral_margin", 1975, "site_splits", "Neutral avg margin"))

# --- Rest / fatigue (novel) ---
register(FeatureSpec("days_rest", 1975, "rest", "Days since last played"))
register(FeatureSpec("games_in_last_7", 1975, "rest", "Conf-tournament-fatigue proxy"))

# --- Matchup (pairwise) ---
register(FeatureSpec("seed_diff", TOURNAMENT_EXPANSION_64, "matchup", "Seed B minus Seed A"))
register(FeatureSpec("tempo_match", KENPOM_AVAILABLE_FROM, "matchup", "Tempo compatibility index"))
register(FeatureSpec("three_pt_style_clash", ADVANCED_STATS_FROM, "matchup", "Team A 3PT-rate vs Team B 3PT-def"))
register(FeatureSpec("common_opp_margin", 1975, "matchup", "Net margin vs shared opponents"))
register(FeatureSpec("travel_miles", TOURNAMENT_EXPANSION_64, "matchup", "Miles from campus to pod city"))
register(FeatureSpec("rest_days_diff", TOURNAMENT_EXPANSION_64, "matchup", "Rest day difference"))
register(FeatureSpec("tz_shift_abs", TOURNAMENT_EXPANSION_64, "matchup", "Timezone crossings"))

# --- Tournament context ---
register(FeatureSpec("round", TOURNAMENT_EXPANSION_64, "context", "Round index 1..6"))
register(FeatureSpec("seed_matchup_upset_rate", TOURNAMENT_EXPANSION_64, "context", "Historical upset rate for this seed pairing"))
register(FeatureSpec("coach_tourney_wins", 1975, "coach", "Career tournament wins of head coach"))
register(FeatureSpec("coach_final_fours", 1975, "coach", "Career Final Four appearances"))
register(FeatureSpec("coach_tenure_years", 1975, "coach", "Years as head coach at this school"))
register(FeatureSpec("roster_continuity", 2008, "roster", "Minutes returning from prior season"))


def available_for_season(season: int, group: str | None = None) -> list[FeatureSpec]:
    return [
        s for s in REGISTRY.values()
        if s.available_since <= season and (group is None or s.group == group)
    ]


def groups() -> set[str]:
    return {s.group for s in REGISTRY.values()}


__all__ = ["FeatureSpec", "REGISTRY", "register", "available_for_season", "groups"]

# --- reference constants used downstream ---
# Round indices — canonical values the model always sees
ROUND_INDEX = {
    "First Four": 0,
    "Round of 64": 1,
    "Round of 32": 2,
    "Sweet Sixteen": 3,
    "Elite Eight": 4,
    "Final Four": 5,
    "Championship": 6,
}

# Tournament expansion markers exported for convenience
EXPAND_64 = TOURNAMENT_EXPANSION_64
EXPAND_68 = TOURNAMENT_EXPANSION_68
