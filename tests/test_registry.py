from madness.features.registry import REGISTRY, available_for_season, groups


def test_registry_has_core_features():
    assert "win_pct" in REGISTRY
    assert "seed_diff" in REGISTRY
    assert "round" in REGISTRY


def test_availability_by_year():
    pre_kenpom = available_for_season(1990, group="kenpom")
    assert pre_kenpom == []
    post_kenpom = available_for_season(2010, group="kenpom")
    assert len(post_kenpom) > 0


def test_groups_nonempty():
    assert {"team_season", "matchup", "context"}.issubset(groups())
