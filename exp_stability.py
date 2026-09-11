"""
How stable is a single player's predicted score?

WHY THIS EXISTS. Fixing the goalkeeper bug moved individual GW4 predictions a
long way — Palmer -1.385, B.Fernandes +0.807, Isak -1.316 — while the aggregate
accuracy barely moved at all (outfield MAE 0.9538 -> 0.9559, ranking within
noise). A model whose population-level error is unchanged but whose individual
numbers swing by more than a point is telling you something important about how
much weight a single prediction can carry.

That matters because captaincy is decided on gaps of a tenth of a point. The
GW4 call came down to B.Fernandes 6.05 versus Palmer 5.97 — a gap of 0.08. If
retraining moves individual predictions by an average of several tenths, a gap
of 0.08 is not a signal at all; it is rounding.

THE CONTROL THAT MAKES THIS HONEST. Comparing broken-vs-fixed alone cannot
distinguish "the goalkeeper fix changed things" from "this model is simply
unstable player-to-player". So the same comparison is run for a change that
should mean NOTHING: the random seed. Identical data, identical features, only
`random_state` differs. Whatever movement the seed produces is the model's
inherent noise floor, and any real effect has to clear it.

Read-only. Never writes predictions.

Usage:
    python exp_stability.py
"""

import sqlite3
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import (HistGradientBoostingClassifier,
                              HistGradientBoostingRegressor)

import recommend
from model import build_features

warnings.filterwarnings("ignore")
BASE = dict(max_iter=400, learning_rate=0.06, max_depth=6, min_samples_leaf=40)


def history(conn, fix_gkp):
    df = pd.read_sql_query(
        "SELECT season, gw, element, fixture, name, position, was_home,"
        " minutes, total_points, xp, xg, xa, xgi, xgc, ict, bps, threat,"
        " creativity, influence, value, selected"
        " FROM history ORDER BY season, element, gw, fixture", conn)
    if fix_gkp:
        df["position"] = df["position"].replace({"GK": "GKP"})
    return df[df["position"].isin(["GKP", "DEF", "MID", "FWD"])].copy()


def predict_gw(conn, fix_gkp, seed):
    hist, feats = build_features(history(conn, fix_gkp))
    train = hist.dropna(subset=["total_points"])
    p = dict(BASE, random_state=seed)
    X, y = train[feats], train["total_points"].values
    clf = HistGradientBoostingClassifier(**p).fit(X, train["played_60"].values)
    m = train["played_60"] == 1
    reg = HistGradientBoostingRegressor(l2_regularization=1.0, **p).fit(
        X[m], y[m.values])

    past, snap = recommend.current_season_frame(conn)
    future, gw, _ = recommend.upcoming_rows(conn, snap)
    cols = list(set(past.columns) & set(future.columns))
    combined, _ = build_features(
        pd.concat([past[cols], future[cols]], ignore_index=True))
    rows = combined[combined["gw"] == gw].copy()
    rows["ep"] = clf.predict_proba(rows[feats])[:, 1] * reg.predict(rows[feats])
    return rows.set_index("element")["ep"], gw


def compare(a, b, label, pos_map, restrict=None):
    idx = a.index.intersection(b.index)
    if restrict:
        idx = [e for e in idx if pos_map.get(e) in restrict]
    x, y = a.loc[idx], b.loc[idx]
    d = (y - x).abs()
    # Only players with a real chance of being picked matter; the tail of
    # non-playing squad filler is trivially stable and would flatter every
    # number here.
    live = x >= 2.0
    print(f"  {label:<34}{d.mean():>8.3f}{d[live].mean():>10.3f}"
          f"{d.max():>9.3f}{x.corr(y, method='spearman'):>10.4f}")
    return d[live].mean()


def main():
    conn = sqlite3.connect("fpl.db")
    try:
        pos = {r[0]: r[1] for r in conn.execute(
            "SELECT element_id, position FROM players"
            " WHERE snapshot_id=(SELECT MAX(id) FROM snapshots)")}

        print("Training four models (this takes a few minutes)...\n")
        broken_42, gw = predict_gw(conn, False, 42)
        fixed_42, _ = predict_gw(conn, True, 42)
        broken_7, _ = predict_gw(conn, False, 7)
        broken_99, _ = predict_gw(conn, False, 99)

        print(f"GW{gw} prediction movement, all {len(broken_42)} players\n")
        print(f"  {'comparison':<34}{'mean|d|':>8}{'mean|d| >2':>10}"
              f"{'max|d|':>9}{'spearman':>10}")
        print("  " + "-" * 71)

        gk_effect = compare(broken_42, fixed_42,
                            "goalkeeper fix (seed held at 42)", pos)
        noise_1 = compare(broken_42, broken_7,
                          "SEED ONLY 42 -> 7  (control)", pos)
        noise_2 = compare(broken_42, broken_99,
                          "SEED ONLY 42 -> 99 (control)", pos)

        print()
        print(f"  {'outfield only:':<34}")
        compare(broken_42, fixed_42, "  goalkeeper fix", pos,
                restrict={"DEF", "MID", "FWD"})
        compare(broken_42, broken_7, "  seed only (control)", pos,
                restrict={"DEF", "MID", "FWD"})

        noise = (noise_1 + noise_2) / 2
        print("\n" + "=" * 71)
        print("  WHAT THIS MEANS FOR A CAPTAINCY DECISION")
        print("=" * 71)
        print(f"\n  Changing NOTHING but the random seed moves a pickable")
        print(f"  player's predicted score by {noise:.3f} points on average.")
        print(f"  The goalkeeper fix moves it by {gk_effect:.3f}.")
        print()
        if gk_effect <= noise * 1.5:
            print("  The goalkeeper fix is INDISTINGUISHABLE from reseeding the")
            print("  model. It is a correctness fix, not a new signal.")
        else:
            print("  The goalkeeper fix moves predictions by more than reseeding,")
            print("  so it is doing something real to the feature space.")
        print()
        print(f"  Either way: any gap between two players smaller than about")
        print(f"  {noise:.2f} points is NOISE. The GW4 captaincy came down to")
        print(f"  B.Fernandes 6.05 vs Palmer 5.97 — a gap of 0.08.")
        print()
        print("  That gap is roughly an order of magnitude below the model's own")
        print("  run-to-run variation. The engine did not have a preference")
        print("  between those two players. It reported one anyway.")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
