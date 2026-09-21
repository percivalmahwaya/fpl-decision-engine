"""
The challenger: the champion, plus opponent difficulty.

WHAT A CHALLENGER IS HERE
=========================
A second model that makes a prediction for every gameweek, logged under its
own name, scored by exactly the same machinery, and used for NOTHING. It does
not pick the captain, it does not drive a transfer, it does not appear in a
recommendation. It exists to be graded alongside the champion so that a
promotion, if it ever happens, is decided by live results rather than by a
backtest and an argument.

WHY THIS ONE
============
Opponent difficulty was built, measured and rejected in `opponent.py` on the
RANKING metric, 4.39 -> 4.27, before a single gameweek had ever been scored
live. The same experiment recorded high-return MAE improving and filed it as
a consolation.

Two live gameweeks have now been graded and both say the same thing:

    GW4   two_stage_ml   MAE 1.126 (best)   high-return 5.353 (WORST)
    GW5   two_stage_ml   MAE 1.048 (best)   high-return 5.116 (WORST)

The champion wins on the average and loses on the half that decides a
gameweek, twice running. Re-measured on identical test rows
(FINDINGS_2026-09-19.md), adding opponent difficulty moves high-return MAE
by -0.214 with a 95% interval of [-0.228, -0.200] and P(better) = 1.00,
while costing about 0.08 on ranking with an interval that crosses zero.

That is a real trade-off and not a free win, which is exactly the situation
a challenger is for. Six live gameweeks will settle it; two backtest metrics
disagreeing will not.

WHAT IT DOES NOT DO
===================
It never writes to `MODEL_NAME`. If this file has a bug, the worst case is a
junk row in the scoreboard under `two_stage_opp` and a champion that carries
on exactly as before. Any failure here is caught and logged rather than
allowed to take the pipeline down with it, because a challenger is a
research artefact and the champion is what actually has to ship.
"""
import pandas as pd

from model import build_features
from opponent import build_opponent_features

CHALLENGER_NAME = "two_stage_opp"


def opponents_by_gameweek(conn):
    """Who each team faces, per gameweek, from the latest fixture snapshot.

    Needed because recommend.py's current-season frame sets `fixture` to the
    gameweek number as an ordering placeholder, so the fixture-pairing trick
    used for history would pair every player in a gameweek with every other.

    Read from the LATEST snapshot only. `fixtures` holds one row per fixture
    per snapshot, so reading them all reports each team playing dozens of
    times a week.
    """
    snapshot = conn.execute("SELECT MAX(snapshot_id) FROM fixtures").fetchone()[0]
    if snapshot is None:
        return pd.DataFrame(columns=["gw", "team_id", "opponent"])

    return pd.read_sql_query(
        """SELECT event AS gw, team_h AS team_id, team_a AS opponent
             FROM fixtures WHERE snapshot_id = ? AND event IS NOT NULL
           UNION ALL
           SELECT event, team_a, team_h
             FROM fixtures WHERE snapshot_id = ? AND event IS NOT NULL""",
        conn, params=(snapshot, snapshot))


def attach_current_season_opponent(df, conn):
    """Give current-season rows a real opponent, keyed by team id.

    The identifier differs from history, which uses team NAMES, and that is
    fine: opponent strength is grouped by (season, opponent), so the two
    namespaces can never collide across the season boundary.
    """
    fixtures = opponents_by_gameweek(conn)
    if fixtures.empty or "team_id" not in df.columns:
        return df

    merged = df.merge(fixtures, on=["gw", "team_id"], how="left")
    # A team with two fixtures in one gameweek would duplicate its players.
    # Keep the first; a double gameweek is a fixture-count question, not an
    # opponent-strength one.
    return merged.drop_duplicates(subset=["gw", "element"], keep="first")


def fit_and_predict(history, combined, gameweek, conn):
    """Train the challenger and predict the upcoming gameweek.

    Returns (element_id, ep, p60, cond) tuples, or None if anything at all
    goes wrong. A challenger must never be able to break the champion's run.
    """
    from sklearn.ensemble import (HistGradientBoostingClassifier,
                                  HistGradientBoostingRegressor)

    # --- history: opponent comes from fixture pairing ---------------------
    hist, extra = build_opponent_features(history)
    hist, feats = build_features(hist)
    feats = list(feats) + [f for f in extra if f in hist.columns]

    train = hist.dropna(subset=["total_points"])
    if train.empty:
        return None

    common = dict(max_iter=400, learning_rate=0.06, max_depth=6,
                  min_samples_leaf=40, random_state=42)
    clf = HistGradientBoostingClassifier(**common).fit(
        train[feats], train["played_60"].values)
    played = train["played_60"] == 1
    reg = HistGradientBoostingRegressor(l2_regularization=1.0, **common).fit(
        train[feats][played], train["total_points"].values[played.values])

    # --- current season: opponent comes from the fixture list -------------
    current = attach_current_season_opponent(combined, conn)
    current, _ = build_opponent_features(current)
    current, _ = build_features(current)

    rows = current[current["gw"] == gameweek].copy()
    if rows.empty:
        return None

    for column in feats:
        if column not in rows.columns:
            rows[column] = float("nan")

    X = rows[feats]
    p60 = clf.predict_proba(X)[:, 1]
    cond = reg.predict(X)

    return [(int(e), float(a * b), float(a), float(b))
            for e, a, b in zip(rows["element"], p60, cond)]


def log(conn, history, combined, gameweek, made_at):
    """Train, predict and log the challenger. Never raises.

    The champion has already been logged by the time this runs. Anything that
    fails here is printed and swallowed, because a research model must not be
    able to cost a real gameweek.
    """
    try:
        predictions = fit_and_predict(history, combined, gameweek, conn)
    except Exception as exc:                       # noqa: BLE001
        print(f"  challenger {CHALLENGER_NAME} failed, champion unaffected: "
              f"{type(exc).__name__}: {exc}")
        return 0

    if not predictions:
        print(f"  challenger {CHALLENGER_NAME}: nothing to log")
        return 0

    conn.executemany(
        "INSERT OR REPLACE INTO predictions "
        "(gameweek, element_id, model, predicted, made_at, p60, cond) "
        "VALUES (?,?,?,?,?,?,?)",
        [(gameweek, element, CHALLENGER_NAME, ep, made_at, p60, cond)
         for element, ep, p60, cond in predictions],
    )
    conn.commit()
    print(f"  Logged {len(predictions)} {CHALLENGER_NAME} predictions "
          f"for GW{gameweek} (challenger, not used for anything)")
    return len(predictions)
