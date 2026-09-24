"""
Turn the database into small JSON files the web app can read.

WHY THIS EXISTS — the split that makes free hosting work
========================================================
Streamlit Community Cloud cannot run this project. Its filesystem is ephemeral,
so `fpl.db` is destroyed on every reboot; it has no scheduler, so nothing runs
unless a browser is open; and it sleeps after 12 quiet hours. A decision support
system that only thinks while you are looking at it is useless.

So the work is split:

    GitHub Actions  = the clock.  Collects, predicts, alerts, emails.
    Streamlit Cloud = the window. Renders what Actions already decided.

This script is the seam between them. It reads the 23 MB database and writes a
few hundred KB of JSON, which is committed to the repo. The web app then needs
no database, no scikit-learn and no model training — it reads JSON and draws.
That is what keeps the app inside 1 GB of RAM with a cold start measured in
seconds rather than minutes.

DESIGN RULE: this script computes nothing that the pipeline already computed.
It reads the `predictions` table that recommend.py wrote before the deadline,
rather than re-running the model — re-running it here would produce numbers
that disagree with the ones that were logged, and the logged ones are the
honest record.

Usage:
    python publish.py              # write site/data/*.json
    python publish.py --print      # write, and show a summary
"""

import json
import os
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path

import alert

ROOT = Path(__file__).parent
DB_PATH = ROOT / "fpl.db"
OUT_DIR = ROOT / "site" / "data"

MODEL_NAME = "two_stage_ml"

# Email when a squad/watchlist player changes badly, or when the deadline is
# close enough that you still have time to act but might forget to look.
#
# 30, NOT 26, AND THE DIFFERENCE IS NOT ARBITRARY. At 26 the Friday-morning run
# before the GW4 deadline measured 26.4 hours remaining and sent nothing — it
# missed the window by 24 minutes, purely because GitHub had delivered the
# trigger four hours late. The margin has to be wider than the scheduler's
# jitter, or a late run silently becomes a missed alert on exactly the morning
# it matters most.
DEADLINE_WARN_HOURS = 30


def utcnow():
    return datetime.now(timezone.utc)


def parse_iso(s):
    """FPL deadlines come back as '...Z', which fromisoformat rejects."""
    if not s:
        return None
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def write(name, payload):
    """
    Write one JSON file.

    encoding is pinned to UTF-8 deliberately: the default on Windows is cp1252,
    which cannot represent the accented names in this data (Horníček, Rúben,
    João Pedro) and raises UnicodeEncodeError mid-write, leaving a truncated
    file behind. ensure_ascii=False keeps them readable in the committed diff.
    """
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / name
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=1, ensure_ascii=False, default=str)
    return path


# --------------------------------------------------------------- sections


def build_meta(conn):
    snap, taken_at, cur_gw, next_gw, deadline = conn.execute(
        "SELECT id, taken_at, current_gw, next_gw, next_deadline"
        " FROM snapshots ORDER BY id DESC LIMIT 1"
    ).fetchone()

    dl = parse_iso(deadline)
    hours = (dl - utcnow()).total_seconds() / 3600 if dl else None

    counts = {}
    for table in ("history", "player_gw", "predictions", "players",
                  "snapshots", "my_squad", "watchlist"):
        counts[table] = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]

    return {
        "generated_at": utcnow().isoformat(),
        # Which database produced these files. After `git pull`, a local
        # checkout holds JSON published by the cloud alongside a local fpl.db
        # that is a different database entirely — the consistency checks in
        # qa_deploy.py must be able to tell "published elsewhere" apart from
        # "genuinely inconsistent", or they cry wolf on every pull.
        "published_by": "github-actions" if os.environ.get("GITHUB_ACTIONS") else "local",
        "snapshot_id": snap,
        "snapshot_taken_at": taken_at,
        "current_gw": cur_gw,
        "next_gw": next_gw,
        "deadline": deadline,
        "hours_to_deadline": round(hours, 2) if hours is not None else None,
        "deadline_passed": bool(dl and utcnow() > dl),
        "row_counts": counts,
    }


def build_squad(conn, next_gw):
    """
    The squad currently owned, with the model's expected points attached.

    Reads `my_squad` at its highest gameweek. That table is the single source
    of truth for ownership; when it fell out of date the alerter spent a week
    warning about players that had already been sold.
    """
    rows = conn.execute(
        """SELECT s.name, s.element_id, s.is_captain, s.is_vice, s.started,
                  p.position, p.price, p.selected_by_percent, p.status,
                  p.chance_next_round, p.news, t.short_name,
                  (SELECT predicted FROM predictions
                    WHERE element_id = s.element_id AND gameweek = ?
                      AND model = ?) AS ep
           FROM my_squad s
           LEFT JOIN players p
             ON p.element_id = s.element_id
            AND p.snapshot_id = (SELECT MAX(id) FROM snapshots)
           LEFT JOIN teams t
             ON t.team_id = p.team_id
            AND t.snapshot_id = p.snapshot_id
           WHERE s.gameweek = (SELECT MAX(gameweek) FROM my_squad)
           ORDER BY CASE p.position
                      WHEN 'GKP' THEN 1 WHEN 'DEF' THEN 2
                      WHEN 'MID' THEN 3 ELSE 4 END,
                    ep DESC""",
        (next_gw, MODEL_NAME),
    ).fetchall()

    squad = []
    for (name, eid, cap, vice, started, pos, price, owned, status,
         chance, news, club, ep) in rows:
        squad.append({
            "name": name, "element_id": eid, "position": pos,
            "club": club, "price": price, "owned": owned,
            "status": status, "chance": chance, "news": news,
            "is_captain": bool(cap), "is_vice": bool(vice),
            "started": started,
            "ep": round(ep, 2) if ep is not None else None,
            # A flagged player is a flagged player whatever the model thinks.
            "flagged": bool(news) or (status not in (None, "a")),
        })

    gw = conn.execute("SELECT MAX(gameweek) FROM my_squad").fetchone()[0]
    value = sum(p["price"] for p in squad if p["price"] is not None)
    return {
        "gameweek": gw,
        "squad": squad,
        "squad_value": round(value, 1),
        "count": len(squad),
    }



def build_bench(conn, next_gw, squad_rows):
    """The bench decision for the upcoming gameweek.

    Needs BOTH halves of the two-stage prediction, because the correct bench
    order is by E[points | played] and not by expected points. See bench.py.

    p60 and cond were only persisted from 2026-09-19, so any gameweek whose
    predictions were logged before that has them NULL. Rather than silently
    ordering by the wrong number, this falls back and SAYS the ordering is
    provisional: a wrong order presented confidently is worse than an
    approximate one that admits it.
    """
    from bench import (Player, bench_boost_advice, build_plan,
                       find_double_gameweeks)

    rows = conn.execute(
        """SELECT s.element_id, s.name, p.position, p.price, p.status,
                  p.chance_next_round,
                  pr.predicted, pr.p60, pr.cond
             FROM my_squad s
             LEFT JOIN players p
               ON p.element_id = s.element_id
              AND p.snapshot_id = (SELECT MAX(id) FROM snapshots)
             LEFT JOIN predictions pr
               ON pr.element_id = s.element_id
              AND pr.gameweek = ? AND pr.model = ?
            WHERE s.gameweek = (SELECT MAX(gameweek) FROM my_squad)""",
        (next_gw, MODEL_NAME),
    ).fetchall()

    # NO PREDICTIONS YET is a different state from PROVISIONAL ORDERING, and
    # conflating them was the first version's bug: with no predictions logged
    # for the upcoming gameweek, every value fell back to zero and the panel
    # rendered a confident bench order and a "hold the chip" verdict built on
    # nothing at all. Predictions are only logged in the run before a
    # deadline, so this is the NORMAL state for most of the week.
    with_prediction = sum(1 for r in rows if r[6] is not None)
    if with_prediction == 0:
        return {"available": False,
                "reason": f"no predictions logged for GW{next_gw} yet. They "
                          "are written in the run before the deadline, so "
                          "this fills in once the gameweek is close."}

    players, provisional = [], False
    for eid, name, pos, price, status, chance, predicted, p60, cond in rows:
        if not pos:
            continue
        if predicted is None:
            # A squad member with no prediction at all: treat as a zero rather
            # than dropping them, or the squad stops being fifteen and the
            # whole plan is refused.
            predicted = 0.0
        if cond is None or p60 is None:
            # Fall back: treat the stored expected points as if the player
            # always appears. The ORDER this produces is the naive one, which
            # is why it is flagged rather than shown as final.
            provisional = True
            p60, cond = 1.0, float(predicted or 0.0)
        players.append(Player(
            element_id=eid, name=name, position=pos,
            p_play=float(p60), points_if_played=float(cond),
            price=float(price or 0.0),
            flagged=bool(status and status not in ("a",))
                    or (chance is not None and chance < 100),
        ))

    if len(players) != 15:
        return {"available": False,
                "reason": f"squad has {len(players)} players with a known "
                          "position, not 15"}

    plan = build_plan(players)
    notes = list(plan.notes)
    if provisional:
        notes.insert(0,
            "ORDER IS PROVISIONAL. These predictions were logged before the "
            "engine started storing both halves of the two-stage model, so "
            "the ordering here uses expected points, which is the wrong "
            "number for a bench. It corrects itself at the next deadline.")

    # History of what Bench Boost would have been worth, for scale.
    recent = []
    for gw in range(max(1, next_gw - 5), next_gw):
        row = conn.execute(
            """SELECT SUM(pr.predicted) FROM my_squad s
                 JOIN predictions pr
                   ON pr.element_id = s.element_id AND pr.gameweek = s.gameweek
                  AND pr.model = ?
                WHERE s.gameweek = ? AND s.started = 0""",
            (MODEL_NAME, gw),
        ).fetchone()
        if row and row[0]:
            recent.append((gw, round(float(row[0]), 2)))

    advice = bench_boost_advice(
        gameweek=next_gw,
        value_now=plan.bench_boost_value,
        recent=recent,
        doubles=find_double_gameweeks(conn, next_gw),
    )

    def render(player):
        return {"name": player.name, "element_id": player.element_id,
                "position": player.position,
                "ep": round(player.expected_points, 2),
                "p_play": round(player.p_play, 3),
                "if_played": round(player.points_if_played, 2),
                "flagged": player.flagged}

    return {
        "available": True,
        "gameweek": next_gw,
        "provisional": provisional,
        "formation": plan.formation,
        "starting_xi": [render(x) for x in plan.starting_xi],
        "bench": [render(b) for b in plan.bench],
        # None, not 0.0, when the ordering is provisional. The fallback sets
        # every p_play to 1.0, so nothing can blank and the autosub figure is
        # structurally zero: an artefact of the assumption rather than a
        # finding about the bench. Publishing it as 0.00 would read as "your
        # bench is worthless as cover", which is not what was measured.
        "autosub_value": None if provisional else round(plan.autosub_value, 2),
        "notes": notes,
        "bench_boost": {
            "value_now": round(advice.value_now, 2),
            "verdict": advice.verdict,
            "recent": advice.recent,
            "double_gameweeks": advice.double_gameweeks,
            "reasoning": advice.reasoning,
        },
    }



def build_transfers(conn, next_gw):
    """Who to buy, who to drop, within the money actually in the bank.

    The bank is READ, not assumed. It was a command-line flag defaulting to
    zero, which silently restricted every suggestion to an exact-price swap.
    """
    from transfers import Candidate, suggest, summarise

    snap = conn.execute("SELECT MAX(id) FROM snapshots").fetchone()[0]

    def fetch(extra, args):
        return conn.execute(f"""
            SELECT p.element_id, p.web_name, p.position, t.short_name, p.price,
                   pr.predicted, p.penalties_order, p.freekicks_order,
                   p.status, p.chance_next_round, p.selected_by_percent
              FROM players p
              JOIN teams t ON t.team_id = p.team_id
                          AND t.snapshot_id = p.snapshot_id
              LEFT JOIN predictions pr ON pr.element_id = p.element_id
                   AND pr.gameweek = ? AND pr.model = ?
             WHERE p.snapshot_id = ? {extra}""",
            (next_gw, MODEL_NAME, snap, *args)).fetchall()

    def build(raw):
        return [Candidate(
            element_id=e, name=n, position=pos, club=club or "?",
            price=price or 0.0, expected_points=float(ep or 0.0),
            penalties_order=pen, freekicks_order=fk, status=st or "a",
            chance=ch, owned_by_percent=float(own or 0))
            for e, n, pos, club, price, ep, pen, fk, st, ch, own in raw]

    owned = [r[0] for r in conn.execute(
        "SELECT element_id FROM my_squad "
        "WHERE gameweek = (SELECT MAX(gameweek) FROM my_squad)")]
    if not owned:
        return {"available": False, "reason": "no squad recorded yet"}

    market = build(fetch("AND pr.predicted IS NOT NULL", ()))
    if not market:
        return {"available": False,
                "reason": f"no predictions logged for GW{next_gw} yet, so "
                          "there is nothing to compare against. They are "
                          "written in the run before the deadline."}

    squad = build(fetch(
        "AND p.element_id IN (%s)" % ",".join("?" * len(owned)), owned))

    row = conn.execute("SELECT bank FROM my_gameweeks "
                       "ORDER BY gameweek DESC LIMIT 1").fetchone()
    bank = (row[0] or 0) / 10.0 if row else 0.0

    moves = suggest(squad, market, bank=bank, free_transfers=1, limit=5)
    out = summarise(moves, bank, 1)
    out["available"] = True
    out["gameweek"] = next_gw
    return out


def build_recommendations(conn, next_gw, limit=40):
    """Top players by the model's expected points for the upcoming gameweek."""
    rows = conn.execute(
        """SELECT pr.element_id, pr.predicted, pr.made_at,
                  p.web_name, p.position, p.price, p.selected_by_percent,
                  p.status, p.chance_next_round, p.news, t.short_name,
                  EXISTS(SELECT 1 FROM my_squad m
                          WHERE m.element_id = pr.element_id
                            AND m.gameweek = (SELECT MAX(gameweek) FROM my_squad))
           FROM predictions pr
           JOIN players p
             ON p.element_id = pr.element_id
            AND p.snapshot_id = (SELECT MAX(id) FROM snapshots)
           LEFT JOIN teams t
             ON t.team_id = p.team_id AND t.snapshot_id = p.snapshot_id
           WHERE pr.gameweek = ? AND pr.model = ?
           ORDER BY pr.predicted DESC
           LIMIT ?""",
        (next_gw, MODEL_NAME, limit),
    ).fetchall()

    picks = [{
        "element_id": eid, "ep": round(ep, 2), "made_at": made_at,
        "name": name, "position": pos, "club": club, "price": price,
        "owned": owned, "status": status, "chance": chance, "news": news,
        "owned_by_me": bool(mine),
    } for (eid, ep, made_at, name, pos, price, owned,
           status, chance, news, club, mine) in rows]

    return {"gameweek": next_gw, "model": MODEL_NAME, "picks": picks}


def build_captain(conn, next_gw):
    """
    The captaincy call: highest expected points among players actually owned.

    Not highest ceiling. captain.py simulated a full season and ceiling
    strategies lost, because captain points double linearly — maximising E[2X]
    is exactly maximising E[X].
    """
    rows = conn.execute(
        """SELECT s.name, s.element_id, p.position, p.news, p.status,
                  (SELECT predicted FROM predictions
                    WHERE element_id = s.element_id AND gameweek = ?
                      AND model = ?) AS ep
           FROM my_squad s
           LEFT JOIN players p
             ON p.element_id = s.element_id
            AND p.snapshot_id = (SELECT MAX(id) FROM snapshots)
           WHERE s.gameweek = (SELECT MAX(gameweek) FROM my_squad)""",
        (next_gw, MODEL_NAME),
    ).fetchall()

    ranked = sorted(
        [{"name": n, "element_id": e, "position": pos, "ep": round(ep, 2),
          "flagged": bool(news) or (st not in (None, "a"))}
         for (n, e, pos, news, st, ep) in rows if ep is not None],
        key=lambda r: -r["ep"],
    )
    recorded = conn.execute(
        "SELECT name FROM my_squad WHERE gameweek = (SELECT MAX(gameweek) FROM my_squad)"
        " AND is_captain = 1"
    ).fetchone()

    return {
        "gameweek": next_gw,
        "model_pick": ranked[0] if ranked else None,
        "your_pick": recorded[0] if recorded else None,
        "agrees": bool(ranked and recorded and ranked[0]["name"] == recorded[0]),
        "ranked": ranked[:8],
    }



# When the champion model changed, and what changed about it.
#
# The accuracy scoreboard compares models BY NAME across gameweeks. If
# `two_stage_ml` quietly starts meaning something else, the line on that
# chart splices two different models together and reads as a trend. That is
# not a chart, it is a mistake with axes.
#
# So every change to what the champion learns from is recorded here and shown
# beside the scoreboard. Keep it honest: add a row, do not edit one.
MODEL_CHANGES = [
    {
        "gameweek": 6,
        "model": "two_stage_ml",
        "change": "Goalkeepers entered training for the first time.",
        "detail": (
            "The archive spelled goalkeeper GK and the rest of the system "
            "spelled it GKP, so 12,500 rows, 11% of the archive, had never "
            "reached training, and the is_gkp feature was a constant zero "
            "while 71 keepers a week were predicted anyway. Scores before "
            "GW6 come from a model that had never seen a goalkeeper, so they "
            "are not strictly comparable with those after it."
        ),
    },
]

def build_accuracy(conn):
    """
    Every model's error on every gameweek that has both predictions and
    results. This is the part that keeps the system honest: predictions are
    written before a deadline and never touched afterwards, so the scoring
    below cannot be flattered after the fact.
    """
    models = [r[0] for r in conn.execute(
        "SELECT DISTINCT model FROM predictions ORDER BY model")]
    gws = [r[0] for r in conn.execute(
        "SELECT DISTINCT gameweek FROM predictions"
        " WHERE gameweek IN (SELECT DISTINCT gameweek FROM player_gw)"
        " ORDER BY gameweek")]

    scored = []
    for gw in gws:
        for m in models:
            rows = conn.execute(
                """SELECT pr.predicted, a.total_points
                   FROM predictions pr
                   JOIN player_gw a
                     ON a.element_id = pr.element_id AND a.gameweek = pr.gameweek
                   WHERE pr.gameweek = ? AND pr.model = ?""",
                (gw, m),
            ).fetchall()
            if not rows:
                continue
            errs = [abs(p - (a or 0)) for p, a in rows]
            hi = [abs(p - (a or 0)) for p, a in rows if (a or 0) >= 5]
            scored.append({
                "gameweek": gw, "model": m, "n": len(rows),
                "mae": round(sum(errs) / len(errs), 3),
                "mae_high_return": round(sum(hi) / len(hi), 3) if hi else None,
            })

    return {
        # WHICH OF THESE MODELS IS THE ONE THAT ACTUALLY DECIDES ANYTHING.
        #
        # Published rather than assumed by each front end. There are now four
        # models on this scoreboard and only one of them picks a captain; the
        # others are baselines and a registered challenger. Both front ends
        # draw the champion in the kit colour and the rest behind it, and
        # before this key existed each of them worked out which was which by
        # hardcoding the name, which is two copies of a fact that changes the
        # day a challenger is promoted.
        "champion": MODEL_NAME,
        "model_changes": MODEL_CHANGES,"models": models, "scored": scored,
            "pending": [g for g in
                        [r[0] for r in conn.execute(
                            "SELECT DISTINCT gameweek FROM predictions ORDER BY gameweek")]
                        if g not in gws]}


def build_season(conn):
    """Percival's own results — the benchmark the model has to beat."""
    rows = conn.execute(
        "SELECT gameweek, total_points, transfers, formation, decided_by"
        " FROM my_gameweeks ORDER BY gameweek"
    ).fetchall()

    gws = []
    for gw, pts, tr, form, how in rows:
        cap = conn.execute(
            "SELECT name, points FROM my_squad WHERE gameweek=? AND is_captain=1", (gw,)
        ).fetchone()
        gws.append({
            "gameweek": gw, "points": pts, "transfers": tr,
            "formation": form, "decided_by": how,
            "captain": cap[0] if cap else None,
            "captain_points": cap[1] if cap else None,
            "played": pts is not None,
        })

    played = [g for g in gws if g["played"]]
    return {
        "gameweeks": gws,
        "total_points": sum(g["points"] for g in played),
        "played": len(played),
        "average": round(sum(g["points"] for g in played) / len(played), 1) if played else None,
    }


def build_alerts(conn, hours=24):
    data = alert.collect(conn, hours=hours)
    if data is None:
        return {"available": False, "alerts": [], "price_moves": [], "urgent": []}
    data["available"] = True
    data["window_hours"] = hours
    # Price moves are long and low-value on a phone; the biggest movers suffice.
    data["price_moves"] = data["price_moves"][:12]
    return data


def build_notify(meta, alerts, squad, captain, transfers=None, bench=None):
    """
    Decide whether this run should email, and what it should say.

    The workflow reads `should_email` and skips the send step when it is false,
    so a quiet day costs nothing and does not train you to ignore the inbox.
    """
    reasons, lines = [], []

    urgent = [a for a in alerts.get("urgent", [])]
    if urgent:
        reasons.append(f"{len(urgent)} squad/watchlist change(s)")
        lines.append("AVAILABILITY — players you own or watch:")
        for a in urgent:
            lines.append(f"  [{a['tag']}] {a['severity']}  {a['name']} ({a['position']})"
                         f"  {a['message']}")
        lines.append("")

    hrs = meta.get("hours_to_deadline")
    if hrs is not None and 0 < hrs <= DEADLINE_WARN_HOURS:
        reasons.append(f"GW{meta['next_gw']} deadline in {hrs:.0f}h")
        lines.append(f"DEADLINE — GW{meta['next_gw']} in {hrs:.1f} hours "
                     f"({meta['deadline']}).")
        lines.append("")

    flagged = [p for p in squad["squad"] if p["flagged"]]
    if flagged:
        lines.append("FLAGGED IN YOUR SQUAD:")
        for p in flagged:
            ch = "n/a" if p["chance"] is None else f"{p['chance']}%"
            lines.append(f"  {p['name']} ({p['position']}) chance {ch} — {p['news']}")
        lines.append("")

    # TRANSFERS. Only emailed when a move clears the noise floor AND is free.
    # A suggestion that costs a four point hit it does not repay is arithmetic
    # worth showing on the page, not worth putting in somebody's inbox, and an
    # email that arrives every single week saying "consider this marginal
    # swap" is an email that stops being read.
    if transfers and transfers.get("available") and transfers.get("has_recommendation"):
        top = next((m for m in transfers["moves"] if m["free"] and m["worth_it"]), None)
        if top:
            reasons.append("transfer worth making")
            lines.append("TRANSFER — %s out, %s in." % (top["out"], top["in"]))
            lines.append("  %+.2f expected points, %.1fm, %.1fm left in the bank."
                         % (top["net"], top["cost"], top["bank_after"]))
            if top["penalties"]:
                lines.append("  %s is his club's first-choice penalty taker."
                             % top["in"])
            for reason in top["reasons"][:2]:
                lines.append("  " + reason)
            lines.append("")

    # BENCH BOOST. Said only when the answer is "play it", because a chip you
    # are holding needs no weekly reminder that you are still holding it.
    if bench and bench.get("available"):
        bb = bench.get("bench_boost", {})
        if bb.get("verdict") == "play it":
            reasons.append("bench boost")
            lines.append("BENCH BOOST — worth %.1f points this week, the best "
                         "of your season." % bb["value_now"])
            lines.append("")

    if captain and captain.get("model_pick"):
        mp = captain["model_pick"]
        lines.append(f"CAPTAIN — model says {mp['name']} ({mp['ep']:.2f} xPts).")
        if captain.get("your_pick") and not captain["agrees"]:
            lines.append(f"  You have {captain['your_pick']} recorded. Model disagrees.")
        lines.append("")

    lines.append(f"Snapshot #{meta['snapshot_id']} at {meta['snapshot_taken_at']}")

    subject = "FPL: " + "; ".join(reasons) if reasons else "FPL: nothing urgent"
    return {
        "should_email": bool(reasons),
        "reasons": reasons,
        "subject": subject[:180],
        "body": "\n".join(lines),
    }


# ------------------------------------------------------------------- main


def photo_caption(stem):
    """Filename to caption: `01_katowice-2024.jpg` becomes "katowice 2024".

    A leading `NN_` orders the files without the digits showing, so adding a
    photograph needs no code at all. This rule also exists in app.py, which
    can glob the directory because Streamlit has a filesystem. qa_deploy.py
    asserts the two agree on the files actually present, because a caption
    that differs between the two front ends is the sort of thing nobody
    notices and everybody trips over later.
    """
    if stem[:2].isdigit():
        stem = stem.split("_", 1)[-1]
    return stem.replace("-", " ").replace("_", " ").strip()


def build_photos():
    """The photographs, listed for a host with no directory listing.

    The Streamlit app globs the folder at request time. A static page cannot:
    it is served by a plain file host that will happily 404 anything not asked
    for by name and offers no way to enumerate. So the list is published like
    every other piece of data here, at the same moment and by the same job.
    """
    folder = ROOT / "site" / "assets" / "photos"
    if not folder.exists():
        return []
    shots = sorted(folder.glob("*.jpg")) + sorted(folder.glob("*.webp"))
    return [{"file": p.name, "caption": photo_caption(p.stem)} for p in shots]


def run(conn, verbose=False):
    meta = build_meta(conn)
    next_gw = meta["next_gw"]

    squad = build_squad(conn, next_gw)
    bench_plan = build_bench(conn, next_gw, squad)
    transfer_plan = build_transfers(conn, next_gw)
    recs = build_recommendations(conn, next_gw)
    captain = build_captain(conn, next_gw)
    accuracy = build_accuracy(conn)
    season = build_season(conn)
    alerts = build_alerts(conn)
    notify = build_notify(meta, alerts, squad, captain,
                          transfers=transfer_plan, bench=bench_plan)

    written = []
    for name, payload in [
        ("meta.json", meta),
        ("squad.json", squad),
        ("bench.json", bench_plan),
        ("transfers.json", transfer_plan),
        ("recommendations.json", recs),
        ("captain.json", captain),
        ("accuracy.json", accuracy),
        ("season.json", season),
        ("alerts.json", alerts),
        ("notify.json", notify),
        ("photos.json", build_photos()),
    ]:
        written.append(write(name, payload))

    total = sum(p.stat().st_size for p in written)
    print(f"  Wrote {len(written)} files to {OUT_DIR.relative_to(ROOT)} "
          f"({total/1024:.0f} KB total)")
    if verbose:
        for p in written:
            print(f"    {p.name:<24}{p.stat().st_size/1024:>7.1f} KB")
        print(f"\n  GW{next_gw}  deadline {meta['deadline']}  "
              f"({meta['hours_to_deadline']}h)")
        print(f"  squad {squad['count']} players, value {squad['squad_value']}m")
        print(f"  {len(recs['picks'])} recommendations, "
              f"{len(alerts.get('alerts', []))} alerts, "
              f"{len(alerts.get('urgent', []))} urgent")
        print(f"  email: {notify['should_email']}  ({notify['subject']})")
    return written


def main():
    if not DB_PATH.exists():
        print(f"  No database at {DB_PATH}. Run fpl_collect.py first.")
        sys.exit(1)
    conn = sqlite3.connect(DB_PATH)
    try:
        run(conn, verbose="--print" in sys.argv)
    finally:
        conn.close()


if __name__ == "__main__":
    main()
