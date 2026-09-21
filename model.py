"""
Phase 2 step 3 — train real models on the historical data and settle the
direct-vs-two-stage question that 3 gameweeks could not answer.

VALIDATION IS CHRONOLOGICAL, NEVER RANDOM. Train on 2022-23..2024-25, test on
2025-26. Random k-fold on time-series data leaks the future into the past and
produces beautiful scores from a model that would lose you points.

NO LEAKAGE IN FEATURES. Every feature is built from a player's PREVIOUS
fixtures only, via groupby().shift(1) before any rolling window. If a feature
could not have been known before the deadline, it must not be in the model.

Three things are compared on identical test rows:
    xP          FPL's own expected points, already in the data (the bar: 4.428)
    direct      one model predicting points
    two_stage   P(60+ mins) x E[points | played], as separate models

Usage:
    python model.py --train
"""

import sqlite3
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, HistGradientBoostingClassifier
from sklearn.metrics import mean_absolute_error

warnings.filterwarnings("ignore")

DB_PATH = Path(__file__).parent / "fpl.db"
TRAIN_SEASONS = ["2022-23", "2023-24", "2024-25"]
TEST_SEASON = "2025-26"

# Columns we roll into recent-form features.
ROLL_COLS = ["total_points", "minutes", "xg", "xa", "xgi", "ict", "bps", "threat", "creativity"]
WINDOWS = [3, 5, 10]


def load(conn):
    df = pd.read_sql_query(
        # `team` is here for opponent pairing (see opponent.py). It is not a
        # model feature and build_features never selects it.
        "SELECT season, gw, element, fixture, name, position, team, was_home,"
        " minutes, total_points, xp, xg, xa, xgi, xgc, ict, bps, threat,"
        " creativity, influence, value, selected"
        " FROM history ORDER BY season, element, gw, fixture",
        conn,
    )
    # Managers were added as a 'player' type in recent seasons; they score on a
    # different basis entirely and would just be noise here.
    df = df[df["position"].isin(["GKP", "DEF", "MID", "FWD"])].copy()
    return df


def build_features(df):
    """
    All features come from a player's PRIOR fixtures.

    The shift(1) inside each group is what enforces that: row N sees rows
    1..N-1 and nothing else. Rolling without the shift would include the
    current fixture's own result, which is the classic silent leak.
    """
    df = df.sort_values(["season", "element", "gw", "fixture"]).copy()
    g = df.groupby(["season", "element"], sort=False)

    df["played_60"] = (df["minutes"] >= 60).astype(int)

    for col in ROLL_COLS + ["played_60"]:
        prior = g[col].shift(1)
        for w in WINDOWS:
            df[f"{col}_m{w}"] = (
                prior.groupby([df["season"], df["element"]], sort=False)
                     .rolling(w, min_periods=1).mean()
                     .reset_index(level=[0, 1], drop=True)
            )
        df[f"{col}_last"] = prior

    # how many fixtures this player already has this season (experience/rotation signal)
    df["appearance_no"] = g.cumcount()

    df["is_gkp"] = (df["position"] == "GKP").astype(int)
    df["is_def"] = (df["position"] == "DEF").astype(int)
    df["is_mid"] = (df["position"] == "MID").astype(int)
    df["is_fwd"] = (df["position"] == "FWD").astype(int)

    feats = (
        [f"{c}_m{w}" for c in ROLL_COLS + ["played_60"] for w in WINDOWS]
        + [f"{c}_last" for c in ROLL_COLS + ["played_60"]]
        + ["was_home", "value", "appearance_no", "is_gkp", "is_def", "is_mid", "is_fwd"]
    )
    return df, feats


def report(name, y_true, y_pred, extra=""):
    mae = mean_absolute_error(y_true, y_pred)
    hi = y_true >= 5
    mae_hi = mean_absolute_error(y_true[hi], y_pred[hi]) if hi.sum() else float("nan")
    print(f"  {name:<12}{mae:>9.3f}{mae_hi:>15.3f}   {extra}")
    return mae_hi


def rank_eval(df_test, preds, k=20):
    """
    Mean ACTUAL points of the top-k picks per gameweek — the decision-relevant
    metric. MAE tells you about average error; this tells you whether the
    players the model likes actually deliver.
    """
    tmp = df_test[["gw", "total_points"]].copy()
    tmp["pred"] = preds
    out = []
    for _, grp in tmp.groupby("gw"):
        top = grp.nlargest(k, "pred")
        out.append(top["total_points"].mean())
    return float(np.mean(out))


def train(conn):
    df = load(conn)
    print(f"  Loaded {len(df):,} player-fixture rows")

    df, feats = build_features(df)

    train_df = df[df["season"].isin(TRAIN_SEASONS)].dropna(subset=["total_points"])
    test_df = df[df["season"] == TEST_SEASON].dropna(subset=["total_points"])
    print(f"  Train: {len(train_df):,} rows ({', '.join(TRAIN_SEASONS)})")
    print(f"  Test:  {len(test_df):,} rows ({TEST_SEASON})   <- never seen in training\n")

    Xtr, ytr = train_df[feats], train_df["total_points"].values
    Xte, yte = test_df[feats], test_df["total_points"].values

    print(f"  {'model':<12}{'MAE':>9}{'HIGH-RET MAE':>15}")
    print("  " + "-" * 40)

    # --- benchmark: FPL's own expected points -----------------------------
    has_xp = test_df["xp"].notna().values
    report("xP (FPL)", yte[has_xp], test_df["xp"].values[has_xp], "<- the bar")

    # --- direct model -----------------------------------------------------
    direct = HistGradientBoostingRegressor(
        max_iter=400, learning_rate=0.06, max_depth=6,
        min_samples_leaf=40, l2_regularization=1.0, random_state=42,
    )
    direct.fit(Xtr, ytr)
    p_direct = direct.predict(Xte)
    report("direct", yte, p_direct)

    # --- two-stage --------------------------------------------------------
    # Stage 1: will they play 60+ minutes?
    clf = HistGradientBoostingClassifier(
        max_iter=300, learning_rate=0.06, max_depth=6,
        min_samples_leaf=40, random_state=42,
    )
    clf.fit(Xtr, train_df["played_60"].values)
    p60 = clf.predict_proba(Xte)[:, 1]

    # Stage 2: given they played 60+, how much do they score?
    # Fitting only on played rows is the point — the 60% of rows with zero
    # minutes otherwise drag every estimate toward zero.
    played = train_df["played_60"] == 1
    reg = HistGradientBoostingRegressor(
        max_iter=400, learning_rate=0.06, max_depth=6,
        min_samples_leaf=40, l2_regularization=1.0, random_state=42,
    )
    reg.fit(Xtr[played], ytr[played.values])
    cond = reg.predict(Xte)
    p_two = p60 * cond
    report("two_stage", yte, p_two)

    # --- ranking comparison ----------------------------------------------
    print(f"\n  Ranking test — mean ACTUAL points of top-20 picks per gameweek:")
    xp_series = test_df["xp"].fillna(0).values
    for nm, pr in (("xP (FPL)", xp_series), ("direct", p_direct), ("two_stage", p_two)):
        print(f"    {nm:<12}{rank_eval(test_df, pr):>7.2f} pts/pick")

    # --- what the model actually leans on ---------------------------------
    from sklearn.inspection import permutation_importance
    sample = test_df.sample(min(4000, len(test_df)), random_state=0)
    imp = permutation_importance(
        direct, sample[feats], sample["total_points"].values,
        n_repeats=3, random_state=0, scoring="neg_mean_absolute_error",
    )
    order = np.argsort(imp.importances_mean)[::-1][:10]
    print("\n  Top 10 features (permutation importance, direct model):")
    for i in order:
        print(f"    {feats[i]:<22}{imp.importances_mean[i]:>7.4f}")


def main():
    conn = sqlite3.connect(DB_PATH)
    try:
        if "--train" in sys.argv:
            train(conn)
        else:
            print(__doc__)
    finally:
        conn.close()


if __name__ == "__main__":
    main()
