"""
Phase 2 step 5 — run the champion model on the CURRENT season and produce an
actual recommendation: who to own, and who to captain.

This is where the project stops being analysis and starts being a decision
support system.

HOW IT WORKS
  1. Train the two-stage champion on all four historical seasons.
  2. Rebuild current-season rows from player_gw (our own collected results)
     in the same shape as the historical table, so features line up exactly.
  3. Append one "prediction row" per player for the upcoming gameweek.
  4. Predict, then log every prediction to the `predictions` table BEFORE the
     deadline so it can be scored honestly afterwards.

CAPTAINCY: chosen by highest expected points, not by ceiling. The season-long
simulation in captain.py showed ceiling strategies lose, because captain points
double linearly — maximising E[2X] is just maximising E[X].

Usage:
    python recommend.py --run
"""

import sqlite3
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, HistGradientBoostingClassifier

from model import load as load_history, build_features
import challenger

warnings.filterwarnings("ignore")
DB_PATH = Path(__file__).parent / "fpl.db"
CURRENT_SEASON = "2026-27"
MODEL_NAME = "two_stage_ml"


def current_season_frame(conn):
    """
    Reshape our own collected results into the same columns build_features()
    expects, so current-season features are computed identically to training.
    """
    snap = conn.execute("SELECT MAX(snapshot_id) FROM players").fetchone()[0]

    played = pd.read_sql_query(
        """SELECT g.gameweek AS gw, g.element_id AS element,
                  g.total_points, g.minutes, g.bps, g.influence, g.creativity,
                  g.threat, g.ict, g.xg, g.xa, g.xgi, g.xgc,
                  p.web_name AS name, p.position, p.price, p.team_id
           FROM player_gw g
           JOIN players p ON p.element_id = g.element_id AND p.snapshot_id = ?""",
        conn, params=(snap,),
    )
    played["season"] = CURRENT_SEASON
    played["fixture"] = played["gw"]          # placeholder; only ordering matters
    played["was_home"] = 0
    played["xp"] = np.nan
    played["value"] = played["price"] * 10    # history stores tenths of a million
    played["selected"] = np.nan
    return played, snap


def upcoming_rows(conn, snap):
    """One row per player for the gameweek we are about to predict."""
    gw, deadline = conn.execute(
        "SELECT next_gw, next_deadline FROM snapshots WHERE id=?", (snap,)
    ).fetchone()

    home = {}
    for th, ta in conn.execute(
        "SELECT team_h, team_a FROM fixtures WHERE snapshot_id=? AND event=?", (snap, gw)
    ):
        home[th] = 1
        home[ta] = 0

    players = pd.read_sql_query(
        """SELECT element_id AS element, web_name AS name, position, price,
                  team_id, selected_by_percent, chance_next_round, status, news
           FROM players WHERE snapshot_id = ?""",
        conn, params=(snap,),
    )
    players["season"] = CURRENT_SEASON
    players["gw"] = gw
    players["fixture"] = gw
    players["was_home"] = players["team_id"].map(home)
    players["value"] = players["price"] * 10
    players["selected"] = players["selected_by_percent"]
    # Outcome columns are unknown — that is the point.
    for c in ["total_points", "minutes", "bps", "influence", "creativity",
              "threat", "ict", "xg", "xa", "xgi", "xgc", "xp"]:
        players[c] = np.nan
    return players, gw, deadline


def run(conn):
    # ---------- train on history ----------
    hist = load_history(conn)
    # The challenger needs history BEFORE build_features, because it adds its
    # own opponent columns first.
    hist_raw = hist.copy()
    hist, feats = build_features(hist)
    train = hist.dropna(subset=["total_points"])
    print(f"  Training on {len(train):,} historical rows (4 seasons)")

    Xtr, ytr = train[feats], train["total_points"].values
    common = dict(max_iter=400, learning_rate=0.06, max_depth=6,
                  min_samples_leaf=40, random_state=42)

    clf = HistGradientBoostingClassifier(**common).fit(Xtr, train["played_60"].values)
    played_mask = train["played_60"] == 1
    reg = HistGradientBoostingRegressor(l2_regularization=1.0, **common).fit(
        Xtr[played_mask], ytr[played_mask.values]
    )

    # ---------- build current-season features ----------
    past, snap = current_season_frame(conn)
    future, gw, deadline = upcoming_rows(conn, snap)

    cols = list(set(past.columns) & set(future.columns))
    combined = pd.concat([past[cols], future[cols]], ignore_index=True)
    combined, _ = build_features(combined)

    pred_rows = combined[combined["gw"] == gw].copy()
    X = pred_rows[feats]

    p60 = clf.predict_proba(X)[:, 1]
    cond = reg.predict(X)
    pred_rows["p60"] = p60
    # Kept, not discarded. cond is E[points | they play], and it is the
    # quantity correct bench ordering needs: the right bench order is by cond
    # alone, NOT by the product below. See bench.py for the proof.
    pred_rows["cond"] = cond
    pred_rows["ep"] = p60 * cond

    # bring back availability info for display
    info = future.set_index("element")[
        ["chance_next_round", "status", "news", "selected_by_percent", "price", "position"]
    ]
    pred_rows = pred_rows.join(info, on="element", rsuffix="_i")

    # A player flagged unavailable should never be recommended, whatever the
    # model says — the model only sees form, not this morning's team news.
    unavailable = pred_rows["status"].isin(["i", "s", "u"]) | (
        pred_rows["chance_next_round"].fillna(100) == 0
    )
    pred_rows.loc[unavailable, "ep"] = 0.0

    # ---------- log predictions BEFORE the deadline ----------
    now = datetime.now(timezone.utc)
    if deadline:
        dl = datetime.fromisoformat(deadline.replace("Z", "+00:00"))
        if now > dl:
            print(f"  Deadline for GW{gw} has passed — not logging (would be dishonest).")
        else:
            # Columns named explicitly. It was positional VALUES (?,?,?,?,?),
            # which silently breaks the moment the table gains a column, and
            # the table just gained two.
            conn.executemany(
                "INSERT OR REPLACE INTO predictions "
                "(gameweek, element_id, model, predicted, made_at, p60, cond) "
                "VALUES (?,?,?,?,?,?,?)",
                [(gw, int(r.element), MODEL_NAME, float(r.ep), now.isoformat(),
                  float(r.p60), float(r.cond))
                 for r in pred_rows.itertuples()],
            )
            conn.commit()
            hrs = (dl - now).total_seconds() / 3600
            print(f"  Logged {len(pred_rows)} predictions for GW{gw} "
                  f"({hrs:.1f}h before deadline)\n")

            # The challenger, logged AFTER the champion and never instead
            # of it. Scored by the same machinery and used for nothing:
            # see challenger.py. Failures there are swallowed, because a
            # research model must not be able to cost a real gameweek.
            challenger.log(conn, hist_raw, combined, gw, now.isoformat())

    # ---------- the recommendation ----------
    ranked = pred_rows.sort_values("ep", ascending=False)

    print(f"  TOP 15 BY EXPECTED POINTS — GW{gw}")
    print(f"  {'player':<18}{'pos':<5}{'£m':>6}{'own%':>7}{'P(60+)':>8}{'xPts':>7}")
    print("  " + "-" * 51)
    for r in ranked.head(15).itertuples():
        own = r.selected_by_percent if pd.notna(r.selected_by_percent) else 0
        print(f"  {str(r.name_i if hasattr(r,'name_i') else r.name)[:17]:<18}"
              f"{str(r.position_i)[:3]:<5}{r.price:>6.1f}{own:>7.1f}{r.p60:>8.2f}{r.ep:>7.2f}")

    cap = ranked.iloc[0]
    print(f"\n  CAPTAIN: {cap.name_i if hasattr(cap,'name_i') else cap['name']}"
          f"  —  {cap.ep:.2f} xPts  (doubles to {2*cap.ep:.2f})")
    print(f"  chosen by highest expected points, per the captain.py simulation")

    flagged = pred_rows[pred_rows["news"].notna() & (pred_rows["selected_by_percent"] > 5)]
    if len(flagged):
        print(f"\n  AVAILABILITY WARNINGS on widely-owned players:")
        for r in flagged.sort_values("selected_by_percent", ascending=False).head(6).itertuples():
            ch = "n/a" if pd.isna(r.chance_next_round) else f"{int(r.chance_next_round)}%"
            print(f"    {str(r.name_i if hasattr(r,'name_i') else r.name)[:18]:<20}"
                  f"{r.selected_by_percent:>5.1f}% owned  chance {ch:>4}  {str(r.news)[:44]}")


def main():
    conn = sqlite3.connect(DB_PATH)
    try:
        if "--run" in sys.argv:
            run(conn)
        else:
            print(__doc__)
    finally:
        conn.close()


if __name__ == "__main__":
    main()
