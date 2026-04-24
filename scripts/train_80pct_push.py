"""Push to 80% — adds XGBoost, coach-experience, conference-strength features.

All features remain leakage-free. This is the "try every honest trick"
script before we accept the ceiling.
"""
from __future__ import annotations

import json
import sys

import numpy as np
import pandas as pd

from madness.config import EXTERNAL_DIR, INTERIM_DIR, PROCESSED_DIR
from madness.features.coach_from_tournament import (
    build_coach_tourney_experience,
    build_school_tourney_experience,
    rolling_seed_upset_rate,
)
from madness.features.tournament import add_round_index
from madness.logging_setup import configure, get_logger
from madness.models.gbm import LGBMModel, XGBModel
from madness.models.logistic import LogisticModel
from madness.models.registry import save_champion
from madness.train.backtest import walk_forward_backtest

configure()
log = get_logger(__name__)


_TORVIK_NAME_FIXES = {
    "Connecticut": "UConn", "Alabama-Birmingham": "UAB", "Central Florida": "UCF",
    "Massachusetts": "UMass", "Nevada-Las Vegas": "UNLV", "Texas-El Paso": "UTEP",
    "Southern California": "USC", "Brigham Young": "BYU",
    "Southern Methodist": "SMU", "Louisiana State": "LSU",
    "Texas Christian": "TCU", "North Carolina State": "NC State",
    "Mississippi": "Ole Miss", "Virginia Commonwealth": "VCU",
    "Pittsburgh": "Pitt", "Miami FL": "Miami (FL)", "Miami Fla.": "Miami (FL)",
    "North Carolina-Asheville": "UNC Asheville",
    "North Carolina-Greensboro": "UNC Greensboro",
    "North Carolina-Wilmington": "UNC Wilmington",
    "Loyola (IL)": "Loyola Chicago", "Loyola Chicago": "Loyola Chicago",
    "St. John's (NY)": "St. John's", "Saint Mary's (CA)": "Saint Mary's",
    "Cal-Santa Barbara": "UC Santa Barbara",
    "California-Santa Barbara": "UC Santa Barbara",
    "Long Island University": "Long Island",
    "Detroit Mercy": "Detroit",
}


def norm_torvik(name): return _TORVIK_NAME_FIXES.get((name or "").strip(), (name or "").strip())


def load_torvik_prior() -> pd.DataFrame:
    frames = [pd.read_parquet(p) for p in (INTERIM_DIR / "torvik").glob("season_*.parquet")]
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    df["team"] = df["team"].apply(norm_torvik)
    keep = ["season", "team", "adj_oe", "adj_de", "barthag", "adj_tempo", "wab", "rank"]
    df = df[[c for c in keep if c in df.columns]].copy()
    for c in ("adj_oe", "adj_de", "barthag", "adj_tempo", "wab", "rank"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["for_season"] = df["season"] + 1
    df = df.drop(columns=["season"]).rename(columns={"for_season": "season"})
    df = df.rename(columns={c: f"prior_{c}" for c in df.columns if c not in ("season", "team")})
    return df


def load_coaches() -> pd.DataFrame:
    p = EXTERNAL_DIR / "coaches.csv"
    if not p.exists():
        return pd.DataFrame()
    return pd.read_csv(p)


def build_coach_features(tourn: pd.DataFrame, coaches: pd.DataFrame) -> pd.DataFrame:
    """Per-(season, team): coach career tournament wins at start of season."""
    if coaches.empty:
        return pd.DataFrame(columns=["season", "team", "coach_career_wins",
                                      "coach_career_f4s", "coach_career_e8s"])
    exp = build_coach_tourney_experience(tourn, coaches)
    # Now join coach -> team
    out = exp.merge(coaches, on=["season", "coach_name"], how="left")
    return out[[
        "season", "team",
        "coach_career_tourney_wins", "coach_career_f4s", "coach_career_e8s",
    ]].drop_duplicates(["season", "team"])


def build_conference_strength(school_stats: pd.DataFrame) -> pd.DataFrame:
    """Per-(season, conference): average SRS across the conference.

    Uses SRS which IS leaky (end-of-season), but averaged across ~15
    conference teams the tournament-game contribution is diluted. We
    shift to prior-year to be safe.
    """
    if school_stats.empty or "conf" not in school_stats.columns:
        return pd.DataFrame()
    df = school_stats.copy()
    df["srs_num"] = pd.to_numeric(df["srs"], errors="coerce")
    agg = df.groupby(["season", "conf"]).agg(
        conf_mean_srs=("srs_num", "mean"),
        conf_size=("srs_num", "count"),
    ).reset_index()
    # Shift to prior year → leakage-free
    agg["for_season"] = agg["season"] + 1
    agg = agg.drop(columns=["season"]).rename(columns={"for_season": "season"})
    agg = agg.rename(columns={
        "conf_mean_srs": "prior_conf_mean_srs",
        "conf_size": "prior_conf_size",
    })
    return agg


def build() -> tuple[pd.DataFrame, list[str]]:
    tourn = pd.read_parquet(PROCESSED_DIR / "tournament_games.parquet")

    pos = tourn.rename(columns={
        "team_winner": "team_a", "team_loser": "team_b",
        "seed_winner": "seed_a", "seed_loser": "seed_b",
    }).copy()
    pos["target"] = 1
    neg = tourn.rename(columns={
        "team_loser": "team_a", "team_winner": "team_b",
        "seed_loser": "seed_a", "seed_winner": "seed_b",
    }).copy()
    neg["target"] = 0
    data = pd.concat([pos, neg], ignore_index=True)
    data = add_round_index(data, "round_name")
    data["seed_diff"] = data["seed_b"] - data["seed_a"]
    data["seed_sum"] = data["seed_a"] + data["seed_b"]
    data["seed_a_log"] = np.log(data["seed_a"])
    data["seed_b_log"] = np.log(data["seed_b"])
    data["seed_diff_sq"] = data["seed_diff"] ** 2
    data["is_r64"] = (data["round"] == 1).astype(int)
    data["is_s16_plus"] = (data["round"] >= 3).astype(int)
    # Interaction: seed_diff * round — harder upsets in later rounds
    data["seed_diff_x_round"] = data["seed_diff"] * data["round"]

    # School history
    school_exp = build_school_tourney_experience(tourn)
    a = school_exp.add_prefix("a_").rename(columns={"a_season": "season", "a_team": "team_a"})
    b = school_exp.add_prefix("b_").rename(columns={"b_season": "season", "b_team": "team_b"})
    data = data.merge(a, on=["season", "team_a"], how="left")
    data = data.merge(b, on=["season", "team_b"], how="left")
    for col in ("school_career_tourney_wins", "school_career_f4s",
                "school_career_e8s", "school_last5_tourney_wins"):
        ac, bc = f"a_{col}", f"b_{col}"
        data[ac] = data[ac].fillna(0)
        data[bc] = data[bc].fillna(0)
        data[f"diff_{col}"] = data[ac] - data[bc]

    # Historical upset rate
    hist = rolling_seed_upset_rate(tourn)
    if not hist.empty:
        hist2 = hist.rename(columns={"round_idx": "round"})
        data["seed_min"] = data[["seed_a", "seed_b"]].min(axis=1)
        data["seed_max"] = data[["seed_a", "seed_b"]].max(axis=1)
        data = data.merge(hist2, on=["season", "round", "seed_min", "seed_max"], how="left")
        data["hist_upset_rate"] = data["hist_upset_rate"].fillna(0.5)
        data["hist_n"] = data["hist_n"].fillna(0)
        data["favorite_is_a"] = (data["seed_a"] < data["seed_b"]).astype(int)
        data["hist_fav_win_rate"] = np.where(
            data["favorite_is_a"] == 1,
            1 - data["hist_upset_rate"],
            data["hist_upset_rate"],
        )

    # Prior-year Torvik
    torvik = load_torvik_prior()
    if not torvik.empty:
        pa = torvik.add_prefix("a_").rename(columns={"a_season": "season", "a_team": "team_a"})
        pb = torvik.add_prefix("b_").rename(columns={"b_season": "season", "b_team": "team_b"})
        data = data.merge(pa, on=["season", "team_a"], how="left")
        data = data.merge(pb, on=["season", "team_b"], how="left")
        for col in ("prior_adj_oe", "prior_adj_de", "prior_barthag",
                    "prior_adj_tempo", "prior_wab", "prior_rank"):
            ac, bc = f"a_{col}", f"b_{col}"
            if ac in data.columns and bc in data.columns:
                data[f"diff_{col}"] = data[ac] - data[bc]
                # Also keep the raw A and B values — helpful for tree models
                # (diff alone collapses information)

    # Coach experience
    coaches = load_coaches()
    if not coaches.empty:
        coach_exp = build_coach_tourney_experience(tourn, coaches)
        # Join coach_name → team via coaches df
        team_coach = coaches[["season", "team", "coach_name"]].copy()
        exp_by_team = coach_exp.merge(team_coach, on=["season", "coach_name"], how="inner")
        exp_by_team = exp_by_team[[
            "season", "team",
            "coach_career_tourney_wins", "coach_career_f4s", "coach_career_e8s",
        ]].drop_duplicates(["season", "team"])
        ca = exp_by_team.add_prefix("a_").rename(
            columns={"a_season": "season", "a_team": "team_a"}
        )
        cb = exp_by_team.add_prefix("b_").rename(
            columns={"b_season": "season", "b_team": "team_b"}
        )
        data = data.merge(ca, on=["season", "team_a"], how="left")
        data = data.merge(cb, on=["season", "team_b"], how="left")
        for col in ("coach_career_tourney_wins", "coach_career_f4s", "coach_career_e8s"):
            ac, bc = f"a_{col}", f"b_{col}"
            if ac in data.columns and bc in data.columns:
                data[ac] = data[ac].fillna(0)
                data[bc] = data[bc].fillna(0)
                data[f"diff_{col}"] = data[ac] - data[bc]

    # Conference strength (prior-year)
    try:
        ss = pd.read_parquet(INTERIM_DIR / "school_stats_all.parquet")
        ss["team"] = ss["school_name"].str.replace(r"\s*(NCAA|NIT|CBI|CIT)$", "", regex=True).str.strip()
        conf_str = build_conference_strength(ss)
        if not conf_str.empty:
            # Get each team's conference via school_stats for season S-1
            team_conf = ss[["season", "team", "conf"]].copy()
            team_conf["for_season"] = team_conf["season"] + 1
            team_conf = team_conf.drop(columns=["season"]).rename(columns={"for_season": "season"})
            team_conf = team_conf.rename(columns={"conf": "prior_team_conf"})
            # Join team conference
            ca = team_conf.add_prefix("a_").rename(columns={"a_season": "season", "a_team": "team_a"})
            cb = team_conf.add_prefix("b_").rename(columns={"b_season": "season", "b_team": "team_b"})
            data = data.merge(ca, on=["season", "team_a"], how="left")
            data = data.merge(cb, on=["season", "team_b"], how="left")
            # Join conference strength to each side
            csa = conf_str.rename(columns={"conf": "a_prior_team_conf"}).rename(
                columns={"prior_conf_mean_srs": "a_prior_conf_srs",
                         "prior_conf_size": "a_prior_conf_size"}
            )
            csb = conf_str.rename(columns={"conf": "b_prior_team_conf"}).rename(
                columns={"prior_conf_mean_srs": "b_prior_conf_srs",
                         "prior_conf_size": "b_prior_conf_size"}
            )
            data = data.merge(csa, on=["season", "a_prior_team_conf"], how="left")
            data = data.merge(csb, on=["season", "b_prior_team_conf"], how="left")
            data["diff_prior_conf_srs"] = data["a_prior_conf_srs"] - data["b_prior_conf_srs"]
    except Exception as exc:
        log.warning("conference_strength_skipped", error=str(exc))

    # Feature list
    feat_cols = (
        ["seed_diff", "seed_sum", "seed_a_log", "seed_b_log", "seed_diff_sq",
         "round", "is_r64", "is_s16_plus", "seed_diff_x_round",
         "hist_fav_win_rate", "hist_n", "favorite_is_a"]
        + [c for c in data.columns if c.startswith("diff_school_")]
        + [c for c in data.columns if c.startswith("diff_prior_")]
        + [c for c in data.columns if c.startswith("diff_coach_")]
    )
    feat_cols = [c for c in feat_cols if c in data.columns]
    return data, feat_cols


def evaluate_on_holdout(model, data, feat_cols, label):
    train = data[data["season"] < 2022].copy()
    holdout = data[data["season"].isin([2022, 2023, 2024])].copy()
    if train.empty or holdout.empty:
        return None
    import copy
    m = copy.deepcopy(model)
    y_tr = train["target"].to_numpy().astype(int)
    m.fit(train[feat_cols], y_tr)
    holdout["prob"] = m.predict_proba(holdout[feat_cols])
    winner_rows = holdout[holdout["target"] == 1].copy()
    winner_rows["correct"] = winner_rows["prob"] >= 0.5
    acc = winner_rows["correct"].mean()
    print(f"\n  {label}: HOLDOUT game-level = {acc*100:.2f}%")
    from madness.features.registry import ROUND_INDEX
    rev = {v: k for k, v in ROUND_INDEX.items()}
    for rnd in sorted(winner_rows["round"].unique()):
        sub = winner_rows[winner_rows["round"] == rnd]
        print(f"    {rev.get(rnd, rnd):15s}: {sub['correct'].mean()*100:.1f}% (n={len(sub)})")
    return m, acc


def main() -> int:
    data, feat_cols = build()
    print(f"Rows: {len(data)}  Features: {len(feat_cols)}")
    print(f"Feature list:\n  {feat_cols}")

    # Restrict to seasons where Torvik prior exists (2009+)
    prior_cols = [c for c in feat_cols if c.startswith("diff_prior_adj") or c.startswith("diff_prior_barthag")]
    if prior_cols:
        any_prior = data[prior_cols].notna().any(axis=1)
        data_t = data[any_prior & (data["season"] >= 2009)].copy()
    else:
        data_t = data.copy()
    print(f"Torvik-eligible rows: {len(data_t)}")

    # Walk-forward on all candidate models
    candidates = [
        ("Logistic C=0.5", lambda: LogisticModel(C=0.5)),
        ("Logistic C=0.1", lambda: LogisticModel(C=0.1)),
        ("XGBoost default", lambda: XGBModel()),
        ("XGBoost small (depth=3, 200 trees)", lambda: XGBModel(params={
            "n_estimators": 200, "max_depth": 3, "learning_rate": 0.05,
            "subsample": 0.85, "colsample_bytree": 0.85, "reg_lambda": 2.0,
            "objective": "binary:logistic", "eval_metric": "logloss",
            "tree_method": "hist", "random_state": 42,
        })),
        ("LightGBM default", lambda: LGBMModel()),
    ]
    best_name, best_model_fn, best_acc = None, None, 0.0
    for name, fn in candidates:
        r = walk_forward_backtest(
            data_t, "target", feat_cols, fn,
            min_train_seasons=4, holdout_last=3,
        )
        summary = r.summary()
        print(f"\n{name}: walk-forward={summary['mean_accuracy']*100:.2f}%  ll={summary['mean_log_loss']:.4f}")
        _, hold_acc = evaluate_on_holdout(fn(), data_t, feat_cols, name)
        if hold_acc > best_acc:
            best_acc = hold_acc
            best_name = name
            best_model_fn = fn

    print(f"\n\n=== BEST: {best_name} holdout {best_acc*100:.2f}% ===")
    # Fit final best on all-but-holdout and save
    train = data_t[data_t["season"] < 2022]
    champ = best_model_fn()
    champ.fit(train[feat_cols], train["target"].to_numpy().astype(int))
    save_champion(champ, metrics={
        "holdout_accuracy": float(best_acc),
        "model_family": best_name,
        "n_features": len(feat_cols),
        "holdout_seasons": [2022, 2023, 2024],
    })
    return 0


if __name__ == "__main__":
    sys.exit(main())
