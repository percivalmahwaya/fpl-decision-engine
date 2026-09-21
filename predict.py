"""
Phase 1b — baseline predictors and the prediction scoreboard.

THE POINT OF THIS FILE: you cannot tell whether a model helps unless you write
predictions down BEFORE the deadline and score them afterwards. Everything
built later (real ML, the optimiser, the minutes model) gets judged against
what this file records. Build the scoreboard first, then try to beat it.

Two deliberately simple baselines are included so the harness has something to
compare from day one:

  naive_form   mean points over the player's last N gameweeks. No adjustments.
  form_fdr     the same, scaled by fixture difficulty and availability.

If a fancy model later can't beat 'form_fdr', the fancy model isn't earning
its complexity.

Usage:
    python predict.py --backfill   # pull actual per-GW results for finished GWs
    python predict.py --predict    # generate + log predictions for the next GW
    python predict.py --score      # score logged predictions against actuals
    python predict.py --status     # what's in the log so far
"""

import json
import sqlite3
import sys
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from migrate import migrate

BASE = "https://fantasy.premierleague.com/api"
DB_PATH = Path(__file__).parent / "fpl.db"
UA = "fpl-research/0.1 (personal research project)"

LOOKBACK = 3  # gameweeks of history the baselines average over

# Fixture Difficulty Rating -> multiplier. FDR 1 is an easy fixture, 5 is hard.
FDR_MULTIPLIER = {1: 1.15, 2: 1.08, 3: 1.00, 4: 0.92, 5: 0.85}


def fetch(endpoint):
    req = urllib.request.Request(f"{BASE}/{endpoint}", headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read().decode("utf-8"))


def create_schema(conn):
    conn.executescript(
        """
        -- Actual per-gameweek results. One row per player per gameweek.
        -- This is the training data spine: real points AND real minutes.
        --
        -- IMPORTANT: this must carry the SAME columns the trained model needs
        -- (xg/xa/xgi/ict/bps/threat/creativity), otherwise features cannot be
        -- built for the current season and the model cannot run live. The
        -- /event/{gw}/live/ endpoint returns all of them, so there is no reason
        -- to store less.
        CREATE TABLE IF NOT EXISTS player_gw (
            gameweek     INTEGER NOT NULL,
            element_id   INTEGER NOT NULL,
            total_points INTEGER,
            minutes      INTEGER,
            goals        INTEGER,
            assists      INTEGER,
            clean_sheet  INTEGER,
            bonus        INTEGER,
            starts       INTEGER,
            bps          INTEGER,
            influence    REAL,
            creativity   REAL,
            threat       REAL,
            ict          REAL,
            xg           REAL,
            xa           REAL,
            xgi          REAL,
            xgc          REAL,
            PRIMARY KEY (gameweek, element_id)
        );

        -- Every prediction ever made, written BEFORE the deadline.
        -- Never update these rows after the fact; that is the whole point.
        CREATE TABLE IF NOT EXISTS predictions (
            gameweek    INTEGER NOT NULL,
            element_id  INTEGER NOT NULL,
            model       TEXT NOT NULL,
            predicted   REAL,
            made_at     TEXT,
            -- The two halves of the two-stage prediction, kept separately.
            -- `predicted` is p60 * cond and was for a long time the only thing
            -- stored, which quietly made correct bench ordering impossible:
            -- the right order is by cond alone, NOT by the product. See
            -- bench.py for why. NULL on rows logged before 2026-09-19.
            p60         REAL,
            cond        REAL,
            PRIMARY KEY (gameweek, element_id, model)
        );

        CREATE INDEX IF NOT EXISTS idx_pgw_element ON player_gw(element_id);
        """
    )

    # Existing databases do not get new columns from CREATE TABLE
    # IF NOT EXISTS. See migrate.py: omitting this killed three
    # scheduled runs on 2026-09-19.
    migrate(conn, verbose=True)



def latest_snapshot(conn):
    row = conn.execute("SELECT MAX(snapshot_id) FROM players").fetchone()
    return row[0] if row else None


# ---------------------------------------------------------------- backfill

def backfill(conn):
    """Pull actual results for every finished gameweek — one request each."""
    boot = fetch("bootstrap-static/")
    finished = [e["id"] for e in boot["events"] if e.get("finished")]
    if not finished:
        print("  No finished gameweeks yet.")
        return

    def f(v):
        """Several of these arrive as strings; keep them numeric."""
        try:
            return float(v)
        except (TypeError, ValueError):
            return None

    for gw in finished:
        live = fetch(f"event/{gw}/live/")
        rows = []
        for el in live.get("elements", []):
            s = el["stats"]
            rows.append((
                gw, el["id"], s.get("total_points"), s.get("minutes"),
                s.get("goals_scored"), s.get("assists"),
                s.get("clean_sheets"), s.get("bonus"),
                s.get("starts"), s.get("bps"),
                f(s.get("influence")), f(s.get("creativity")), f(s.get("threat")),
                f(s.get("ict_index")),
                f(s.get("expected_goals")), f(s.get("expected_assists")),
                f(s.get("expected_goal_involvements")), f(s.get("expected_goals_conceded")),
            ))
        conn.executemany(
            "INSERT OR REPLACE INTO player_gw VALUES (" + ",".join("?" * 18) + ")", rows
        )
        played = sum(1 for r in rows if r[3])  # minutes > 0
        print(f"  GW{gw}: stored {len(rows)} players ({played} actually played)")
    conn.commit()


# ---------------------------------------------------------------- predict

def next_gameweek(conn):
    """The gameweek we should be predicting, from the most recent snapshot."""
    snap = latest_snapshot(conn)
    row = conn.execute(
        "SELECT next_gw, next_deadline FROM snapshots WHERE id=?", (snap,)
    ).fetchone()
    return (row[0], row[1]) if row else (None, None)


def team_difficulty(conn, gw):
    """Map team_id -> FDR for that team in the given gameweek."""
    snap = latest_snapshot(conn)
    diff = {}
    for th, ta, dh, da in conn.execute(
        "SELECT team_h, team_a, difficulty_h, difficulty_a FROM fixtures"
        " WHERE snapshot_id=? AND event=?", (snap, gw)
    ):
        # A team can appear twice in a double gameweek; keep the easier fixture.
        if dh is not None:
            diff[th] = min(diff.get(th, 9), dh)
        if da is not None:
            diff[ta] = min(diff.get(ta, 9), da)
    return diff


def recent_mean(conn, element_id, upto_gw, n=LOOKBACK):
    """Mean points over the player's last n gameweeks before upto_gw."""
    rows = conn.execute(
        "SELECT total_points FROM player_gw WHERE element_id=? AND gameweek<?"
        " ORDER BY gameweek DESC LIMIT ?", (element_id, upto_gw, n)
    ).fetchall()
    if not rows:
        return None
    return sum(r[0] or 0 for r in rows) / len(rows)


def predict(conn):
    gw, deadline = next_gameweek(conn)
    if gw is None:
        print("  No upcoming gameweek found. Run fpl_collect.py first.")
        return

    now = datetime.now(timezone.utc)
    if deadline:
        dl = datetime.fromisoformat(deadline.replace("Z", "+00:00"))
        if now > dl:
            print(f"  WARNING: GW{gw} deadline ({deadline}) has already passed.")
            print("  Predictions logged now would not be honest. Aborting.")
            return
        hrs = (dl - now).total_seconds() / 3600
        print(f"  Predicting GW{gw} — deadline in {hrs:.1f} hours ({deadline})")

    snap = latest_snapshot(conn)
    diff = team_difficulty(conn, gw)

    players = conn.execute(
        "SELECT element_id, web_name, team_id, position, chance_next_round, status"
        " FROM players WHERE snapshot_id=?", (snap,)
    ).fetchall()

    made_at = now.isoformat()
    rows_naive, rows_fdr = [], []
    for eid, name, team, pos, chance, status in players:
        base = recent_mean(conn, eid, gw)
        if base is None:
            continue  # no history yet (new signing) — baseline says nothing

        rows_naive.append((gw, eid, "naive_form", round(base, 3), made_at))

        # availability: explicit percentage if given, else infer from status
        if chance is not None:
            avail = chance / 100.0
        elif status in ("i", "s", "u"):   # injured / suspended / unavailable
            avail = 0.0
        else:
            avail = 1.0

        mult = FDR_MULTIPLIER.get(diff.get(team), 1.0) if team in diff else 0.0
        rows_fdr.append((gw, eid, "form_fdr", round(base * mult * avail, 3), made_at))

    conn.executemany(
        "INSERT OR REPLACE INTO predictions VALUES (?,?,?,?,?)", rows_naive + rows_fdr
    )
    conn.commit()
    print(f"  Logged {len(rows_naive)} naive_form and {len(rows_fdr)} form_fdr predictions.")

    print(f"\n  Top 10 by form_fdr for GW{gw}:")
    for name, pred, pos, price in conn.execute(
        """SELECT p.web_name, pr.predicted, p.position, p.price
           FROM predictions pr JOIN players p
             ON p.element_id = pr.element_id AND p.snapshot_id = ?
           WHERE pr.gameweek=? AND pr.model='form_fdr'
           ORDER BY pr.predicted DESC LIMIT 10""",
        (snap, gw),
    ):
        print(f"    {name:<18} {pos}  {price:>5.1f}m   {pred:>5.2f} pts")


# ---------------------------------------------------------------- score

def score(conn):
    gws = [
        r[0] for r in conn.execute(
            "SELECT DISTINCT gameweek FROM predictions WHERE gameweek IN"
            " (SELECT DISTINCT gameweek FROM player_gw) ORDER BY gameweek"
        )
    ]
    if not gws:
        print("  Nothing to score yet — no gameweek has both predictions and results.")
        print("  (Predictions must be made before a deadline, then scored after it.)")
        return

    print(f"  {'GW':<4}{'Model':<13}{'MAE':>7}{'MAE high-ret':>14}{'n':>7}")
    for gw in gws:
        for model in ("naive_form", "form_fdr"):
            rows = conn.execute(
                """SELECT pr.predicted, a.total_points
                   FROM predictions pr JOIN player_gw a
                     ON a.element_id = pr.element_id AND a.gameweek = pr.gameweek
                   WHERE pr.gameweek=? AND pr.model=?""",
                (gw, model),
            ).fetchall()
            if not rows:
                continue
            errs = [abs(p - (a or 0)) for p, a in rows]
            mae = sum(errs) / len(errs)
            # High-return players are what actually decide rank, so track them apart.
            hi = [abs(p - a) for p, a in rows if (a or 0) >= 5]
            mae_hi = sum(hi) / len(hi) if hi else float("nan")
            print(f"  {gw:<4}{model:<13}{mae:>7.2f}{mae_hi:>14.2f}{len(rows):>7}")


def p_sixty(conn, element_id, upto_gw, n=LOOKBACK):
    """Stage 1: P(player gets 60+ minutes), from recent starts. None = no history."""
    rows = conn.execute(
        "SELECT minutes FROM player_gw WHERE element_id=? AND gameweek<?"
        " ORDER BY gameweek DESC LIMIT ?", (element_id, upto_gw, n)
    ).fetchall()
    if not rows:
        return None
    return sum(1 for (m,) in rows if m and m >= 60) / len(rows)


def mean_when_played(conn, element_id, upto_gw, n=5):
    """
    Stage 2: mean points in gameweeks the player actually played 60+ minutes.

    Fitting on played-only rows is the whole point: the 961 zero-minute rows
    drag a plain average toward zero and hide how good a player is when fit.
    """
    rows = conn.execute(
        "SELECT total_points FROM player_gw WHERE element_id=? AND gameweek<?"
        " AND minutes >= 60 ORDER BY gameweek DESC LIMIT ?",
        (element_id, upto_gw, n),
    ).fetchall()
    if not rows:
        return None
    return sum(r[0] or 0 for r in rows) / len(rows)


def two_stage(conn, element_id, upto_gw):
    """E[points] = P(60+ mins) x E[points | played 60+]."""
    p = p_sixty(conn, element_id, upto_gw)
    if p is None:
        return None
    if p == 0:
        return 0.0          # not starting recently -> expect nothing
    ceiling = mean_when_played(conn, element_id, upto_gw)
    if ceiling is None:
        return 0.0
    return p * ceiling


def backtest(conn):
    """
    Replay naive_form over past gameweeks using ONLY data that existed before
    each one, and score it. This gives the number any future model must beat.

    Deliberately limited to naive_form: it depends solely on prior gameweek
    points. The form_fdr variant needs fixture difficulty and availability as
    they stood before the deadline, and our snapshots only start at GW3 — using
    today's values would leak the future and flatter the result.
    """
    gws = [r[0] for r in conn.execute(
        "SELECT DISTINCT gameweek FROM player_gw ORDER BY gameweek")]
    testable = [g for g in gws if g > min(gws)]  # need at least one prior GW
    if not testable:
        print("  Not enough history to backtest yet.")
        return

    models = {
        "naive_form": recent_mean,
        "two_stage": two_stage,
    }

    print("  Honest backtest — each model sees only gameweeks before the one scored.")
    print("  HIGH-RET MAE is the headline number: aggregate MAE is dominated by")
    print("  correctly predicting that non-players score nothing.\n")
    print(f"  {'GW':<5}{'model':<13}{'MAE':>7}{'HIGH-RET MAE':>14}{'n':>7}")

    totals = {m: [] for m in models}
    for gw in testable:
        for name, fn in models.items():
            rows = []
            for eid, actual in conn.execute(
                "SELECT element_id, total_points FROM player_gw WHERE gameweek=?", (gw,)
            ):
                pred = fn(conn, eid, gw)
                if pred is not None:
                    rows.append((pred, actual or 0))
            if not rows:
                continue
            mae = sum(abs(p - a) for p, a in rows) / len(rows)
            hi = [abs(p - a) for p, a in rows if a >= 5]
            mae_hi = sum(hi) / len(hi) if hi else float("nan")
            totals[name].extend(hi)
            print(f"  {gw:<5}{name:<13}{mae:>7.2f}{mae_hi:>14.2f}{len(rows):>7}")
        print()

    print("  Overall high-return MAE (all backtested gameweeks pooled):")
    ranked = sorted(
        ((sum(v) / len(v), k) for k, v in totals.items() if v)
    )
    for i, (score_, name) in enumerate(ranked):
        mark = "  <- best" if i == 0 else ""
        print(f"    {name:<13}{score_:>7.2f}{mark}")
    if len(ranked) == 2:
        gain = ranked[1][0] - ranked[0][0]
        pct = 100 * gain / ranked[1][0]
        print(f"\n    {ranked[0][1]} improves on {ranked[1][1]} by {gain:.2f} pts ({pct:.1f}%)")


def rank_test(conn):
    """
    Evaluate models the way FPL actually uses them: as a ranking.

    MAE is the wrong headline metric here. You never "use" a prediction of 4.2
    points — you pick the best ~15 players. What matters is whether the players
    a model ranks highest actually score. Worse still, MAE measured on the
    high-return subset is biased: that subset is conditioned on players who
    DID play, which systematically punishes any model that discounts for the
    risk of not playing.

    So: take each model's top-K, and look at what those players really scored.
    """
    gws = [r[0] for r in conn.execute(
        "SELECT DISTINCT gameweek FROM player_gw ORDER BY gameweek")]
    testable = [g for g in gws if g > min(gws)]
    if not testable:
        print("  Not enough history.")
        return

    models = {"naive_form": recent_mean, "two_stage": two_stage}
    K = 20

    print(f"  Ranking test — mean ACTUAL points of each model's top {K} picks.")
    print("  This is the question that matters: do its favourites deliver?\n")
    print(f"  {'GW':<5}{'model':<13}{'mean actual':>13}{'hauls(5+)':>11}{'blanks(0)':>11}")

    pooled = {m: [] for m in models}
    for gw in testable:
        actuals = dict(conn.execute(
            "SELECT element_id, total_points FROM player_gw WHERE gameweek=?", (gw,)
        ))
        for name, fn in models.items():
            scored = []
            for eid in actuals:
                pred = fn(conn, eid, gw)
                if pred is not None:
                    scored.append((pred, eid))
            if not scored:
                continue
            scored.sort(reverse=True)
            top = [actuals[eid] or 0 for _, eid in scored[:K]]
            pooled[name].extend(top)
            hauls = sum(1 for a in top if a >= 5)
            blanks = sum(1 for a in top if a == 0)
            print(f"  {gw:<5}{name:<13}{sum(top)/len(top):>13.2f}{hauls:>11}{blanks:>11}")
        print()

    print(f"  Pooled across gameweeks — mean actual points of top {K}:")
    ranked = sorted(((sum(v) / len(v), k) for k, v in pooled.items() if v), reverse=True)
    for i, (val, name) in enumerate(ranked):
        blanks = sum(1 for a in pooled[name] if a == 0)
        n = len(pooled[name])
        mark = "  <- best" if i == 0 else ""
        print(f"    {name:<13}{val:>7.2f} pts/pick   blanks {blanks}/{n}{mark}")


def status(conn):
    n_pred = conn.execute("SELECT COUNT(*) FROM predictions").fetchone()[0]
    n_act = conn.execute("SELECT COUNT(*) FROM player_gw").fetchone()[0]
    gws_act = conn.execute("SELECT COUNT(DISTINCT gameweek) FROM player_gw").fetchone()[0]
    print(f"  Actual results stored: {n_act} rows across {gws_act} gameweek(s)")
    print(f"  Predictions logged:    {n_pred} rows")
    for gw, model, n, made in conn.execute(
        "SELECT gameweek, model, COUNT(*), MIN(made_at) FROM predictions"
        " GROUP BY gameweek, model ORDER BY gameweek, model"
    ):
        print(f"    GW{gw}  {model:<12} {n:>4} predictions, logged {made[:19]}")


def main():
    conn = sqlite3.connect(DB_PATH)
    try:
        create_schema(conn)
        if "--backfill" in sys.argv:
            backfill(conn)
        elif "--predict" in sys.argv:
            predict(conn)
        elif "--score" in sys.argv:
            score(conn)
        elif "--backtest" in sys.argv:
            backtest(conn)
        elif "--rank" in sys.argv:
            rank_test(conn)
        elif "--status" in sys.argv:
            status(conn)
        else:
            print(__doc__)
    finally:
        conn.close()


if __name__ == "__main__":
    main()
