"""
Re-judging two rejected or deferred changes against the metric that turned out
to matter.

    python exp_reevaluate.py

WHY RUN THIS AT ALL
===================
Two training-data changes are sitting unshipped, for different reasons.

  1. OPPONENT DIFFICULTY was built, measured and REJECTED (see opponent.py).
     The rejection was made on the RANKING metric: mean actual points of the
     top-20 picks per gameweek, which went 4.39 -> 4.27. But the same
     experiment recorded high-return MAE improving, 5.305 -> 5.169, and that
     was treated as a consolation.

     Then GW4 was scored live, the first graded gameweek this project has
     ever had, and it said the champion's overall MAE is the best of three
     models (1.126) while its HIGH-RETURN MAE is the worst by a distance
     (5.353 against form_fdr's 4.572).

     So the model's measured weakness in the real world is precisely the
     thing opponent difficulty improved, and the metric it was rejected on
     may simply have been the wrong one to optimise. That is worth re-running
     rather than assuming either way.

  2. THE GOALKEEPER FIX has never been evaluated alongside anything else.
     `history` labels keepers GK, the live API labels them GKP, and model.py
     filters on GKP, so 12,500 rows and 11% of the archive never reach
     training while 71 keepers a week are predicted anyway.

Both are changes to what the model learns from, so they are tested together
as a 2x2 rather than one after the other. Two changes evaluated separately
can each look neutral and still interact.

WHAT THIS DOES NOT DO
=====================
It does not promote anything. It prints a table and a recommendation, and a
human decides. Shipping a model mid-season also silently breaks the accuracy
scoreboard, because GW4 was scored against the current champion and a
replacement would be graded under the same name.
"""
import sqlite3
import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import (HistGradientBoostingClassifier,
                              HistGradientBoostingRegressor)
from sklearn.metrics import mean_absolute_error

from model import TEST_SEASON, TRAIN_SEASONS, build_features, load
from opponent import build_opponent_features

SEED = 42
BOOTSTRAP = 2000
COMMON = dict(max_iter=400, learning_rate=0.06, max_depth=6,
              min_samples_leaf=40, random_state=SEED)

# The live API's spelling. `history` uses "GK" for the same thing, which is
# the whole bug.
LIVE_POSITIONS = ["GKP", "DEF", "MID", "FWD"]
HISTORY_GK = "GK"


def prepare(conn, fix_goalkeepers):
    """Load history, with or without the keeper fix applied.

    NOT model.load(). Two reasons, and the first one nearly invalidated this
    whole experiment:

      1. model.load() ITSELF filters position to the live spelling before
         returning, so calling it and then renaming GK to GKP renames nothing:
         the rows are already gone. The first version of this script did
         exactly that, ran clean, and would have reported the goalkeeper fix
         as having no effect. It had no effect because it was never applied.

      2. It does not select `team` or the columns opponent.py pairs fixtures
         on, so the opponent arm could not be built from it at all.
    """
    df = pd.read_sql_query(
        "SELECT season, gw, element, fixture, name, position, team,"
        " opponent_team, was_home, minutes, total_points, xp, xg, xa, xgi,"
        " xgc, ict, bps, threat, creativity, influence, value, selected"
        " FROM history ORDER BY season, element, gw, fixture",
        conn,
    )

    # Managers score on a different basis entirely and are noise here. Note
    # this drops them WITHOUT touching keepers, which is the distinction
    # model.load() fails to make.
    df = df[df["position"].isin(LIVE_POSITIONS + [HISTORY_GK])].copy()

    if fix_goalkeepers:
        df.loc[df["position"] == HISTORY_GK, "position"] = "GKP"

    return df


def fit_predict(train_df, test_df, feats):
    """The champion's own two-stage recipe, unchanged."""
    Xtr, ytr = train_df[feats], train_df["total_points"].values
    Xte = test_df[feats]

    clf = HistGradientBoostingClassifier(**COMMON).fit(
        Xtr, train_df["played_60"].values)
    played = train_df["played_60"] == 1
    reg = HistGradientBoostingRegressor(l2_regularization=1.0, **COMMON).fit(
        Xtr[played], ytr[played.values])

    return clf.predict_proba(Xte)[:, 1] * reg.predict(Xte)


def rank_by_gameweek(test_df, preds, k=20):
    """Top-k actual points, per gameweek. Returned per gameweek, not averaged,
    so the bootstrap has something to resample."""
    tmp = test_df[["gw", "total_points"]].copy()
    tmp["pred"] = preds
    return np.array([grp.nlargest(k, "pred")["total_points"].mean()
                     for _, grp in tmp.groupby("gw")])


def paired_bootstrap(a, b, n=BOOTSTRAP, seed=SEED):
    """P(b beats a) and a 95% interval on the difference, resampling
    gameweeks. Paired because both models saw the same gameweeks: comparing
    unpaired throws away the thing that makes the comparison sensitive."""
    rng = np.random.default_rng(seed)
    idx = np.arange(len(a))
    diffs = np.array([
        (b[s] - a[s]).mean()
        for s in (rng.choice(idx, len(idx), replace=True) for _ in range(n))
    ])
    return diffs.mean(), np.percentile(diffs, [2.5, 97.5]), (diffs > 0).mean()


def run_arm(conn, name, fix_goalkeepers, use_opponent):
    """IDENTICAL TEST ROWS FOR EVERY ARM. Only the TRAINING set changes.

    This is the trap FINDINGS_2026-09-11.md documents, and the first version
    of this script walked straight into it and produced the very numbers that
    document calls superseded, 0.954 -> 0.912.

    The cause: letting the keeper fix change the TEST set as well as the
    training set. Keeper rows have MAE around 0.58, far below outfield, so
    adding 3,427 of them to the test set drags the mean down whatever the
    model has learned. It measures composition and reports it as skill. The
    ranking metric has the mirror flaw, since the top-20 is chosen from the
    candidate pool and the champion's pool contained no keepers at all.

    So keepers are ALWAYS present in the frame and always in the test set,
    which is also what production actually does: it predicts 71 keepers a
    week from a model that has never seen one. Only training changes.
    """
    df = prepare(conn, fix_goalkeepers=True)   # always present in the frame

    if use_opponent:
        df, extra = build_opponent_features(df)
    else:
        extra = []

    df = df[df["position"].isin(LIVE_POSITIONS)].copy()
    df, feats = build_features(df)
    feats = list(feats) + [f for f in extra if f in df.columns]

    train_df = df[df["season"].isin(TRAIN_SEASONS)].dropna(subset=["total_points"])
    test_df = df[df["season"] == TEST_SEASON].dropna(subset=["total_points"])

    # THE ONLY THING THIS FLAG CHANGES: whether keepers are learned from.
    if not fix_goalkeepers:
        train_df = train_df[train_df["position"] != "GKP"]

    preds = fit_predict(train_df, test_df, feats)
    y = test_df["total_points"].values
    high = y >= 5

    return {
        "name": name,
        "n_train": len(train_df),
        "n_test": len(test_df),
        "mae": mean_absolute_error(y, preds),
        "mae_high": mean_absolute_error(y[high], preds[high]),
        "rank": rank_by_gameweek(test_df, preds),
        "abs_err": np.abs(y - preds),
        "abs_err_high": np.abs(y[high] - preds[high]),
    }


def main():
    conn = sqlite3.connect("fpl.db")

    arms = [
        ("champion",            False, False),
        ("+ keeper fix",        True,  False),
        ("+ opponent",          False, True),
        ("+ both",              True,  True),
    ]

    results = []
    for name, gk, opp in arms:
        print(f"  fitting {name} ...", flush=True)
        results.append(run_arm(conn, name, gk, opp))

    base = results[0]

    print()
    print("  " + "=" * 74)
    print("  %-14s %9s %14s %14s %9s" %
          ("arm", "MAE", "high-ret MAE", "rank pts/pick", "train n"))
    print("  " + "-" * 74)
    for r in results:
        print("  %-14s %9.3f %14.3f %14.2f %9s" %
              (r["name"], r["mae"], r["mae_high"], r["rank"].mean(),
               f"{r['n_train']:,}"))
    print("  " + "=" * 74)

    print()
    print("  Against the champion, resampling gameweeks (2000 draws):")
    print()
    for r in results[1:]:
        d, ci, p = paired_bootstrap(base["rank"], r["rank"])
        print("  %-14s ranking  %+.3f  95%% CI [%+.3f, %+.3f]  P(better) = %.2f"
              % (r["name"], d, ci[0], ci[1], p))

    print()
    print("  High-return MAE is the live weakness, so it gets its own test.")
    print("  LOWER is better here, so P(better) is P(error goes down):")
    print()
    for r in results[1:]:
        assert len(r["abs_err_high"]) == len(base["abs_err_high"]), (
            "test sets differ, so the pairing would compare unrelated rows")
        d, ci, p = paired_bootstrap(r["abs_err_high"], base["abs_err_high"])
        print("  %-14s hi-MAE   %+.3f  95%% CI [%+.3f, %+.3f]  P(better) = %.2f"
              % (r["name"], -d, -ci[1], -ci[0], p))

    print()
    print("  Nothing here is promoted. This prints evidence; a human ships.")


if __name__ == "__main__":
    sys.exit(main())
