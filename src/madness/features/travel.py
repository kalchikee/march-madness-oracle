"""Travel, distance, and circadian-disruption features.

Novel-signal hypothesis: In the first two rounds, teams playing closer
to home win ~3-5 pp more than pure talent differential predicts. Timezone
shifts ≥2 hours also measurably hurt performance on afternoon games.
"""
from __future__ import annotations

import math

import pandas as pd


EARTH_R_MI = 3958.8


def haversine_miles(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    lat1r, lat2r = math.radians(lat1), math.radians(lat2)
    dlat = lat2r - lat1r
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1r) * math.cos(lat2r) * math.sin(dlon / 2) ** 2
    return 2 * EARTH_R_MI * math.asin(math.sqrt(a))


def build_travel_features(
    games: pd.DataFrame,
    campus_locations: pd.DataFrame,
    venue_locations: pd.DataFrame,
) -> pd.DataFrame:
    """Compute travel miles and timezone shift per team per game.

    games: season, team_a, team_b, venue_id
    campus_locations: team, lat, lon, tz_offset
    venue_locations: venue_id, lat, lon, tz_offset
    """
    if games.empty:
        return games

    g = games.merge(venue_locations, on="venue_id", how="left", suffixes=("", "_venue"))
    for side in ("a", "b"):
        c = campus_locations.rename(columns={
            "team": f"team_{side}", "lat": f"{side}_lat",
            "lon": f"{side}_lon", "tz_offset": f"{side}_tz",
        })
        g = g.merge(c, on=f"team_{side}", how="left")

    g["a_travel_miles"] = g.apply(
        lambda r: haversine_miles(r["a_lat"], r["a_lon"], r["lat"], r["lon"])
        if pd.notna(r.get("a_lat")) and pd.notna(r.get("lat")) else None,
        axis=1,
    )
    g["b_travel_miles"] = g.apply(
        lambda r: haversine_miles(r["b_lat"], r["b_lon"], r["lat"], r["lon"])
        if pd.notna(r.get("b_lat")) and pd.notna(r.get("lat")) else None,
        axis=1,
    )
    g["a_tz_shift"] = (g["tz_offset"] - g["a_tz"]).abs()
    g["b_tz_shift"] = (g["tz_offset"] - g["b_tz"]).abs()
    g["travel_miles_diff"] = g["a_travel_miles"] - g["b_travel_miles"]
    g["tz_shift_diff"] = g["a_tz_shift"] - g["b_tz_shift"]
    return g
