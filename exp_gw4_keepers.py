"""
Did the goalkeeper bug change Percival's GW4 transfer?

He switched Horníček -> Pickford on 2026-09-11 because the engine rated
Pickford 3.38 against Horníček 3.09. Both of those numbers came from a model
that had never been trained on a single goalkeeper.

This re-runs the live prediction path with the bug fixed and compares, for the
two keepers actually involved plus the rest of the field. It answers one
question: was the decision he acted on a real signal or an artefact?

STRICTLY READ-ONLY. It never touches the `predictions` table. Those rows were
written before the deadline and re-running recommend.py would overwrite them,
destroying the honest record and the engine-vs-gut comparison in compare.py.

Usage:
    python exp_gw4_keepers.py
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
COMMON = dict(max_iter=400, learning_rate=0.06, max_depth=6,
              min_samples_leaf=40, random_state=42)


def history(conn, fix_gkp):
    df = pd.read_sql_query(
        "SELECT season, gw, element, fixture, name, position, was_home,"
        " minutes, total_points, xp, xg, xa, xgi, xgc, ict, bps, threat,"
        " creativity, influence, value, selected"
        " FROM history ORDER BY season, element, gw, fixture", conn)
    if fix_gkp:
        df["position"] = df["position"].replace({"GK": "GKP"})
    return df[df["position"].isin(["GKP", "DEF", "MID", "FWD"])].copy()


def predict_gw(conn, fix_gkp):
    """recommend.run()'s prediction path, without any of its writes."""
    hist, feats = build_features(history(conn, fix_gkp))
    train = hist.dropna(subset=["total_points"])

    X, y = train[feats], train["total_points"].values
    clf = HistGradientBoostingClassifier(**COMMON).fit(X, train["played_60"].values)
    m = train["played_60"] == 1
    reg = HistGradientBoostingRegressor(l2_regularization=1.0, **COMMON).fit(
        X[m], y[m.values])

    past, snap = recommend.current_season_frame(conn)
    future, gw, _ = recommend.upcoming_rows(conn, snap)
    cols = list(set(past.columns) & set(future.columns))
    combined, _ = build_features(
        pd.concat([past[cols], future[cols]], ignore_index=True))

    rows = combined[combined["gw"] == gw].copy()
    rows["ep"] = clf.predict_proba(rows[feats])[:, 1] * reg.predict(rows[feats])

    info = future.set_index("element")[["position", "status", "chance_next_round"]]
    rows = rows.join(info, on="element", rsuffix="_i")
    unavailable = rows["status"].isin(["i", "s", "u"]) | (
        rows["chance_next_round"].fillna(100) == 0)
    rows.loc[unavailable, "ep"] = 0.0
    return rows.set_index("element")["ep"], rows, gw


def main():
    conn = sqlite3.connect("fpl.db")
    try:
        print("Training both models on the full archive...\n")
        broken, rb, gw = predict_gw(conn, fix_gkp=False)
        fixed, rf, _ = predict_gw(conn, fix_gkp=True)

        squad = pd.read_sql_query(
            "SELECT s.name, s.element_id, s.started FROM my_squad s"
            " WHERE s.gameweek = (SELECT MAX(gameweek) FROM my_squad)", conn)
        pos = {r[0]: r[1] for r in conn.execute(
            "SELECT element_id, position FROM players"
            " WHERE snapshot_id=(SELECT MAX(id) FROM snapshots)")}

        print(f"GW{gw} — your squad, both models\n")
        print(f"  {'player':<14}{'pos':<5}{'broken':>9}{'fixed':>9}{'delta':>9}   role")
        print("  " + "-" * 58)
        recs = []
        for r in squad.itertuples():
            b = float(broken.get(r.element_id, np.nan))
            f = float(fixed.get(r.element_id, np.nan))
            recs.append((r.name, pos.get(r.element_id), b, f, f - b, r.started))
        for n, p, b, f, d, st in sorted(recs, key=lambda x: -x[3]):
            role = "XI" if st else "bench"
            print(f"  {n[:13]:<14}{str(p):<5}{b:>9.3f}{f:>9.3f}{d:>+9.3f}   {role}")

        print("\n" + "=" * 60)
        print("  THE DECISION: Horníček -> Pickford")
        print("=" * 60)
        keepers = [(n, b, f) for n, p, b, f, d, st in recs if p == "GKP"]
        for n, b, f in sorted(keepers, key=lambda x: -x[2]):
            print(f"    {n[:13]:<14} broken {b:.3f}   fixed {f:.3f}")
        if len(keepers) == 2:
            (n1, b1, f1), (n2, b2, f2) = sorted(keepers, key=lambda x: -x[1])
            print(f"\n    broken model preferred: {n1}  by {b1-b2:+.3f}")
            g1, g2 = sorted(keepers, key=lambda x: -x[2])[:2]
            print(f"    fixed  model prefers:   {g1[0]}  by {g1[2]-g2[2]:+.3f}")
            same = n1 == g1[0]
            print(f"\n    SAME CHOICE: {'YES — the transfer stands' if same else 'NO — the bug changed the call'}")

        # How much did keeper predictions move in general?
        gk_ids = [e for e, p in pos.items() if p == "GKP"]
        b = broken.reindex(gk_ids).dropna()
        f = fixed.reindex(gk_ids).dropna()
        common = b.index.intersection(f.index)
        print(f"\n  Across all {len(common)} goalkeepers in GW{gw}:")
        print(f"    mean prediction  broken {b.loc[common].mean():.3f}"
              f"   fixed {f.loc[common].mean():.3f}"
              f"   ({f.loc[common].mean()-b.loc[common].mean():+.3f})")
        print(f"    mean |change|    {np.abs(f.loc[common]-b.loc[common]).mean():.3f}")
        rho = b.loc[common].corr(f.loc[common], method="spearman")
        print(f"    rank correlation {rho:.4f}   "
              f"({'ordering essentially unchanged' if rho > 0.95 else 'ORDERING MOVED'})")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
