"""
QA suite for the deployment layer.

qa.py checks the modelling. This checks everything the modelling gets published
THROUGH — the JSON contract between the pipeline and the web app, the workflow
that produces it, and the repository hygiene that keeps the free hosting inside
its limits.

WHY A SEPARATE SUITE: the failures here are different in kind. A model bug
produces a wrong number. A deployment bug produces a page that is confidently
out of date, an email that never arrives, or a squad shown to the public that
was sold three gameweeks ago — which is exactly what this project had before
these checks existed. Nothing crashes; it just quietly stops being true.

Checks are grouped:
    CONTRACT    the published JSON is complete, well-formed and internally sane
    CONSISTENT  what was published matches what is in the database
    HONESTY     predictions were logged before deadlines, not after
    APP         the front end stays inside Streamlit's free-tier constraints
    WORKFLOW    the schedule, permissions and ordering are correct
    HYGIENE     nothing large or secret is committed

Every check below is written so that it CAN fail. A check that compares a value
to itself passes forever and tests nothing — this suite already caught two of
those in qa.py, so each assertion here is anchored to an independent source.

Usage:
    python qa_deploy.py
"""

import ast
import json
import os
import re
import sqlite3
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path

ROOT = Path(__file__).parent
DB_PATH = ROOT / "fpl.db"
DATA = ROOT / "site" / "data"
WORKFLOW = ROOT / ".github" / "workflows" / "fpl.yml"

PASS, FAIL, WARN = "PASS", "FAIL", "WARN"
results = []

EXPECTED_FILES = ["meta", "squad", "recommendations", "captain",
                  "accuracy", "season", "alerts", "notify", "bench",
                  "transfers", "photos"]

SQUAD_SHAPE = {"GKP": 2, "DEF": 5, "MID": 5, "FWD": 3}
SEVERITIES = {"CRITICAL", "WARNING", "INFO", "GOOD"}
TAGS = {"SQUAD", "WATCH", "OWNED", "-"}

# Streamlit Community Cloud caps memory at 1 GB. These must never appear in
# the app's requirements or imports.
BANNED_IN_APP = ["scikit-learn", "sklearn", "scipy", "sqlite3", "torch", "tensorflow"]


def check(name, status, detail=""):
    results.append((name, status, detail))
    icon = {PASS: "[ok]  ", FAIL: "[FAIL]", WARN: "[warn]"}[status]
    print(f"  {icon} {name}")
    if detail:
        print(f"         {detail}")


def ok(name, condition, detail=""):
    check(name, PASS if condition else FAIL, detail)
    return condition


def load(name):
    path = DATA / f"{name}.json"
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def parse_iso(s):
    if not s:
        return None
    try:
        return datetime.fromisoformat(str(s).replace("Z", "+00:00"))
    except ValueError:
        return None


# ------------------------------------------------------------- CONTRACT

def qa_contract():
    print("\nPUBLISHED DATA CONTRACT")

    missing = [n for n in EXPECTED_FILES if not (DATA / f"{n}.json").exists()]
    if not ok("all expected files published", not missing,
              f"missing: {missing}" if missing else f"{len(EXPECTED_FILES)} files"):
        return None

    # Valid UTF-8 JSON. Windows defaults to cp1252, which cannot encode the
    # accented names in this data and truncates the file mid-write.
    bad = []
    for n in EXPECTED_FILES:
        try:
            with open(DATA / f"{n}.json", encoding="utf-8") as fh:
                json.load(fh)
        except (json.JSONDecodeError, UnicodeDecodeError) as e:
            bad.append(f"{n}: {e}")
    ok("every file is valid UTF-8 JSON", not bad, "; ".join(bad))

    meta = load("meta")
    required = ["generated_at", "snapshot_id", "next_gw", "deadline",
                "hours_to_deadline", "row_counts"]
    absent = [k for k in required if k not in meta]
    ok("meta has all required keys", not absent, f"missing: {absent}" if absent else "")

    gen = parse_iso(meta.get("generated_at"))
    ok("generated_at parses as a datetime", gen is not None, str(meta.get("generated_at")))
    if gen:
        ok("generated_at is not in the future",
           gen <= datetime.now(timezone.utc) + timedelta(minutes=5),
           f"generated {gen.isoformat()}")

    dl = parse_iso(meta.get("deadline"))
    ok("deadline parses as a datetime", dl is not None, str(meta.get("deadline")))

    # hours_to_deadline must agree with the two timestamps it was derived from.
    # Anchored to deadline and generated_at independently, so a hardcoded or
    # stale value fails rather than passing trivially.
    if dl and gen and meta.get("hours_to_deadline") is not None:
        expected = (dl - gen).total_seconds() / 3600
        drift = abs(expected - meta["hours_to_deadline"])
        ok("hours_to_deadline matches deadline minus generated_at", drift < 0.05,
           f"stated {meta['hours_to_deadline']:.2f}h, derived {expected:.2f}h")

    total_kb = sum((DATA / f"{n}.json").stat().st_size for n in EXPECTED_FILES) / 1024
    # These files are committed twice a day forever. At 2 MB a commit that is
    # ~1.4 GB of git history per year, which defeats the point of the split.
    check("published payload is small enough to commit twice daily",
          PASS if total_kb < 512 else (WARN if total_kb < 2048 else FAIL),
          f"{total_kb:.0f} KB across {len(EXPECTED_FILES)} files")

    return meta


def qa_squad():
    print("\nSQUAD")
    squad = load("squad")
    if not ok("squad.json has players", squad and squad.get("squad")):
        return None
    players = squad["squad"]

    ok("squad has exactly 15 players", len(players) == 15, f"{len(players)} players")

    shape = {}
    for p in players:
        shape[p["position"]] = shape.get(p["position"], 0) + 1
    ok("squad is a legal FPL shape (2/5/5/3)", shape == SQUAD_SHAPE, str(shape))

    ids = [p["element_id"] for p in players]
    ok("every squad player has an element_id", all(i is not None for i in ids),
       f"{sum(1 for i in ids if i is None)} unmatched")
    ok("no duplicate players in squad", len(set(ids)) == len(ids),
       f"{len(ids) - len(set(ids))} duplicates")

    caps = [p for p in players if p["is_captain"]]
    vices = [p for p in players if p["is_vice"]]
    ok("exactly one captain", len(caps) == 1, f"{len(caps)} captains")
    ok("at most one vice-captain", len(vices) <= 1, f"{len(vices)} vices")
    if caps and vices:
        ok("captain and vice are different players",
           caps[0]["element_id"] != vices[0]["element_id"])

    value = squad.get("squad_value")
    # A squad outside this band means prices failed to join, not that a
    # remarkable team was assembled.
    ok("squad value is plausible", value is not None and 75 <= value <= 125,
       f"£{value}m")

    priced = [p for p in players if p["price"] is not None]
    ok("every player joined to a price", len(priced) == len(players),
       f"{len(players) - len(priced)} missing prices")

    stated = round(sum(p["price"] for p in priced), 1)
    ok("squad_value equals the sum of its prices", abs(stated - value) < 0.05,
       f"stated {value}, summed {stated}")

    return squad


def qa_recommendations(meta):
    print("\nRECOMMENDATIONS")
    recs = load("recommendations")
    if not ok("recommendations published", recs and recs.get("picks")):
        return
    picks = recs["picks"]

    eps = [p["ep"] for p in picks]
    ok("sorted by expected points, descending",
       all(eps[i] >= eps[i + 1] for i in range(len(eps) - 1)))
    ok("no negative expected points", all(e >= 0 for e in eps),
       f"min {min(eps):.2f}")
    ok("expected points are within a sane range", max(eps) < 25,
       f"max {max(eps):.2f}")

    ok("recommendations are for the upcoming gameweek",
       recs.get("gameweek") == meta.get("next_gw"),
       f"recs GW{recs.get('gameweek')} vs meta GW{meta.get('next_gw')}")

    # A player the FPL API says is injured, suspended or unavailable must never
    # be recommended, whatever the model thinks — the model sees form, not this
    # morning's team news.
    leaked = [p["name"] for p in picks
              if p.get("status") in ("i", "s", "u") and p["ep"] > 0]
    ok("no unavailable player carries a positive score", not leaked,
       f"{leaked[:5]}" if leaked else "")

    zero_chance = [p["name"] for p in picks
                   if p.get("chance") == 0 and p["ep"] > 0]
    ok("no 0%-chance player carries a positive score", not zero_chance,
       f"{zero_chance[:5]}" if zero_chance else "")

    ids = [p["element_id"] for p in picks]
    ok("no duplicate players in recommendations", len(set(ids)) == len(ids))


def qa_captain(squad):
    print("\nCAPTAIN")
    cap = load("captain")
    if not ok("captain.json published", cap is not None):
        return
    if not cap.get("model_pick"):
        check("model pick present", WARN, "no prediction available yet")
        return

    ranked = cap.get("ranked", [])
    ok("model pick is the highest-scoring ranked player",
       ranked and cap["model_pick"]["element_id"] == ranked[0]["element_id"],
       f"pick {cap['model_pick']['name']}, top of list "
       f"{ranked[0]['name'] if ranked else 'none'}")

    ok("ranked list is sorted descending",
       all(ranked[i]["ep"] >= ranked[i + 1]["ep"] for i in range(len(ranked) - 1)))

    if squad:
        squad_ids = {p["element_id"] for p in squad["squad"]}
        ok("model pick is a player you actually own",
           cap["model_pick"]["element_id"] in squad_ids,
           cap["model_pick"]["name"])
        ok("every ranked captain candidate is in the squad",
           all(r["element_id"] in squad_ids for r in ranked))

        recorded = [p["name"] for p in squad["squad"] if p["is_captain"]]
        if recorded:
            ok("your_pick matches the captain flagged in the squad",
               cap.get("your_pick") == recorded[0],
               f"captain.json says {cap.get('your_pick')}, squad says {recorded[0]}")
            # `agrees` must be derived, not asserted.
            ok("`agrees` is computed correctly",
               cap.get("agrees") == (cap["model_pick"]["name"] == recorded[0]),
               f"agrees={cap.get('agrees')}")


def qa_alerts_and_notify(meta, squad):
    print("\nALERTS AND NOTIFICATION")
    alerts = load("alerts")
    notify = load("notify")

    if alerts and alerts.get("available"):
        rows = alerts.get("alerts", [])
        ok("every alert has a known severity",
           all(a["severity"] in SEVERITIES for a in rows))
        ok("every alert has a known tag", all(a["tag"] in TAGS for a in rows))

        # Squad players must sort above everyone else, or the one alert that
        # matters is buried under 17 irrelevant ones.
        order = [a["tag"] for a in rows]
        rank = {"SQUAD": 0, "WATCH": 1, "OWNED": 2, "-": 3}
        ok("alerts are ordered by priority",
           all(rank[order[i]] <= rank[order[i + 1]] for i in range(len(order) - 1)),
           " ".join(order[:8]))

        urgent = alerts.get("urgent", [])
        derived = [a for a in rows if a["tag"] in ("SQUAD", "WATCH")
                   and a["severity"] in ("CRITICAL", "WARNING")]
        ok("`urgent` matches its own definition", len(urgent) == len(derived),
           f"{len(urgent)} listed, {len(derived)} derived")

        if squad:
            squad_ids = {p["element_id"] for p in squad["squad"]}
            mistagged = [a["name"] for a in rows
                         if a["tag"] == "SQUAD" and a["element_id"] not in squad_ids]
            # This is the check that would have caught the sold-players bug.
            ok("no alert is tagged SQUAD for a player you do not own",
               not mistagged, f"{mistagged}" if mistagged else "")
    else:
        check("alerts available", WARN, "not enough snapshots to compare")

    if not ok("notify.json published", notify is not None):
        return

    ok("should_email is a boolean", isinstance(notify.get("should_email"), bool),
       repr(notify.get("should_email")))
    if notify.get("should_email"):
        ok("an email that will send has a reason", bool(notify.get("reasons")))
        ok("an email that will send has a body", bool(notify.get("body", "").strip()))
    ok("subject fits in an email header",
       0 < len(notify.get("subject", "")) <= 180,
       f"{len(notify.get('subject', ''))} chars")

    # Nothing personal should end up in a file committed to a public repo.
    blob = json.dumps([alerts, notify, load("meta"), load("squad")], default=str)
    leaks = re.findall(r"[\w.+-]+@[\w-]+\.[\w.]+", blob)
    leaks = [l for l in leaks if not l.endswith("noreply.github.com")]
    ok("no email addresses in published data", not leaks, f"{set(leaks)}" if leaks else "")
    ok("no obvious credentials in published data",
       not re.search(r"(?i)(api[_-]?key|password|secret|bearer)\s*[\"':=]\s*\S", blob))


# ----------------------------------------------------------- CONSISTENT

def qa_consistency(conn, meta, squad):
    print("\nCONSISTENCY WITH THE DATABASE")

    # These checks only mean something when the JSON and the database came from
    # the same run. On a developer machine after `git pull`, site/data holds
    # what the cloud published while fpl.db is a different database — every
    # check below would fail for a reason that is not a defect. Detect that and
    # warn instead, rather than either crying wolf or silently weakening the
    # gate in CI, where both always come from the same run.
    here = "github-actions" if os.environ.get("GITHUB_ACTIONS") else "local"
    published_by = meta.get("published_by")
    if published_by and published_by != here:
        check("published data was produced by this environment", WARN,
              f"published by {published_by}, checking from {here} — "
              f"consistency skipped; run `python publish.py` to compare locally")
        return

    db_snap = conn.execute("SELECT MAX(id) FROM snapshots").fetchone()[0]
    ok("published snapshot is the newest in the database",
       meta.get("snapshot_id") == db_snap,
       f"published #{meta.get('snapshot_id')}, database #{db_snap}")

    db_gw = conn.execute(
        "SELECT next_gw FROM snapshots ORDER BY id DESC LIMIT 1").fetchone()[0]
    ok("published next_gw matches the database", meta.get("next_gw") == db_gw,
       f"published GW{meta.get('next_gw')}, database GW{db_gw}")

    for table, published in meta.get("row_counts", {}).items():
        actual = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        if actual != published:
            check(f"row_counts.{table} matches the database", FAIL,
                  f"published {published}, actual {actual}")
            break
    else:
        check("row_counts match the database", PASS,
              f"{len(meta.get('row_counts', {}))} tables verified")

    if squad:
        db_ids = {r[0] for r in conn.execute(
            "SELECT element_id FROM my_squad WHERE gameweek ="
            " (SELECT MAX(gameweek) FROM my_squad)")}
        pub_ids = {p["element_id"] for p in squad["squad"]}
        ok("published squad matches my_squad exactly", db_ids == pub_ids,
           f"only in DB {db_ids - pub_ids}, only published {pub_ids - db_ids}")

        # Expected points must be the values that were logged, not recomputed.
        mismatched = []
        for p in squad["squad"]:
            row = conn.execute(
                "SELECT predicted FROM predictions WHERE element_id=? AND gameweek=?"
                " AND model='two_stage_ml'", (p["element_id"], meta["next_gw"])
            ).fetchone()
            if row and p["ep"] is not None and abs(round(row[0], 2) - p["ep"]) > 0.005:
                mismatched.append(p["name"])
        ok("published expected points match the logged predictions",
           not mismatched, f"{mismatched[:5]}" if mismatched else "")


def qa_benchmark(conn):
    """Is the record of Percival's own results still being updated?

    This is the benchmark the whole project is measured against: GW1 to GW3
    were picked on instinct, and anything the engine produces has to beat them.
    A benchmark that quietly stops advancing does not look broken. It looks
    like a season that has not happened yet.

    And it did stop. record_my_team.py was a dict transcribed from screenshots
    on 2026-09-07, replayed by the pipeline twice a day. The workflow step was
    called "Record squad", reported success every run, and could not have
    learned a new score without somebody editing the file. GW4 finished, was
    scored by every other part of the system, and sat at null here for days.
    """
    print("\nBENCHMARK (my own results)")

    finished = [
        r[0] for r in conn.execute(
            """SELECT event FROM fixtures
               WHERE snapshot_id = (SELECT MAX(snapshot_id) FROM fixtures)
                 AND event IS NOT NULL
               GROUP BY event
               HAVING SUM(CASE WHEN finished THEN 0 ELSE 1 END) = 0""")
    ]
    if not finished:
        check("benchmark is current", PASS, "no finished gameweeks yet")
        return

    recorded = dict(conn.execute(
        "SELECT gameweek, total_points FROM my_gameweeks"))
    missing = sorted(g for g in finished
                     if recorded.get(g) is None)

    entry_id = os.environ.get("FPL_ENTRY_ID", "").strip()

    if not entry_id:
        # Unconfigured is a warning, not a failure: the rest of the engine
        # still works and still has something useful to say. Configured but
        # broken is a failure, below, because that is a silent regression.
        check("benchmark is fetched, not transcribed", WARN,
              "FPL_ENTRY_ID is not set, so my own results are replayed from a "
              "hand transcription and stop at the last gameweek somebody typed "
              "in. Set it as a repository variable to the number in "
              "fantasy.premierleague.com/entry/NNNNNNN/")
        check("every finished gameweek has my score",
              PASS if not missing else WARN,
              f"GW{', GW'.join(str(g) for g in missing)} finished but unscored "
              f"here, which follows from the line above" if missing
              else f"{len(finished)} finished gameweeks, all recorded")
        return

    ok("benchmark is fetched, not transcribed", True,
       f"FPL_ENTRY_ID is set to {entry_id}")
    ok("every finished gameweek has my score", not missing,
       (f"GW{', GW'.join(str(g) for g in missing)} finished but carries no "
        f"total_points. FPL_ENTRY_ID is set, so the fetch itself is failing "
        f"and the benchmark has silently stopped advancing.")
       if missing else f"{len(finished)} finished gameweeks, all recorded")

    # A squad with no points at all, after the gameweek is over, means the
    # picks were stored but never joined to what they scored.
    for gw in finished:
        n = conn.execute(
            "SELECT COUNT(*) FROM my_squad WHERE gameweek=? AND points IS NOT NULL",
            (gw,)).fetchone()[0]
        total = conn.execute(
            "SELECT COUNT(*) FROM my_squad WHERE gameweek=?", (gw,)).fetchone()[0]
        if total:
            ok(f"GW{gw} picks are joined to their scores", n > 0,
               "" if n else f"{total} picks stored, not one carrying points")


def qa_honesty(conn, meta):
    print("\nHONESTY OF THE PREDICTION LOG")

    gw = meta.get("next_gw")
    dl = parse_iso(meta.get("deadline"))

    rows = conn.execute(
        "SELECT made_at FROM predictions WHERE gameweek=? AND model='two_stage_ml'",
        (gw,)).fetchall()
    if not rows:
        check("predictions logged for the upcoming gameweek", WARN, f"none for GW{gw}")
        return

    ok("predictions logged for the upcoming gameweek", True, f"{len(rows)} rows")

    # The single most important property of this project: a prediction written
    # after kick-off is not a prediction. If this ever fails, every accuracy
    # number in the app is worthless.
    if dl:
        late = [r[0] for r in rows if (parse_iso(r[0]) or dl) > dl]
        ok("every prediction was made before its deadline", not late,
           f"{len(late)} logged after the deadline")

    # Every gameweek whose fixtures have ALL finished must have actuals loaded.
    #
    # `predict.py --backfill` asks the API which gameweeks are finished and
    # stores each one. If that list comes back short — the API flips `finished`
    # late, the step is reordered, a request fails — the step prints a tidy
    # "GW1..GW3 stored" and exits 0 having quietly skipped the gameweek that
    # just ended. Nothing downstream complains: scoring has nothing to join, so
    # it says "nothing to score yet", and the published scoreboard stays empty
    # while every step of the pipeline reports success.
    #
    # That is how GW4 sat unscored for days in September 2026 across five
    # consecutive green runs. Checked against the stored fixture table rather
    # than the network, so the gate works offline and cannot be fooled by the
    # same API response that caused the miss.
    finished_gws = [
        r[0] for r in conn.execute(
            """SELECT event FROM fixtures
               WHERE snapshot_id = (SELECT MAX(snapshot_id) FROM fixtures)
                 AND event IS NOT NULL
               GROUP BY event
               HAVING SUM(CASE WHEN finished THEN 0 ELSE 1 END) = 0""")
    ]
    have_actuals = {r[0] for r in conn.execute(
        "SELECT DISTINCT gameweek FROM player_gw")}
    missing = sorted(g for g in finished_gws if g not in have_actuals)
    ok("every finished gameweek has its actual results loaded",
       not missing,
       ("GW" + ", GW".join(str(g) for g in missing) + " finished but absent from "
        "player_gw - predict.py --backfill reported success without storing them")
       if missing else f"{len(finished_gws)} finished gameweeks, all present")

    scored = conn.execute(
        """SELECT COUNT(*) FROM predictions p JOIN player_gw a
             ON a.element_id = p.element_id AND a.gameweek = p.gameweek""").fetchone()[0]
    acc = load("accuracy")
    if acc and acc.get("scored"):
        ok("accuracy figures are non-negative and finite",
           all(s["mae"] >= 0 and s["mae"] < 100 for s in acc["scored"]))
        ok("scored gameweeks have a positive sample size",
           all(s["n"] > 0 for s in acc["scored"]))
    else:
        # An empty scoreboard is only tolerable while nothing is scorable. Once
        # a gameweek has both predictions and results, an empty scoreboard is a
        # failure and not a note: the scoreboard is the product.
        joinable = sorted(g for g in finished_gws if g in have_actuals and conn.execute(
            "SELECT 1 FROM predictions WHERE gameweek=? LIMIT 1", (g,)).fetchone())
        ok("accuracy scored", not joinable,
           ("GW" + ", GW".join(str(g) for g in joinable) + " have both predictions "
            "and results, but the published scoreboard is empty")
           if joinable else f"nothing scorable yet ({scored} prediction/result joins)")


# ------------------------------------------------------------------ APP

def qa_app():
    print("\nSTREAMLIT APP CONSTRAINTS")

    app = ROOT / "app.py"
    if not ok("app.py exists", app.exists()):
        return
    source = app.read_text(encoding="utf-8")

    try:
        compile(source, "app.py", "exec")
        check("app.py compiles", PASS)
    except SyntaxError as e:
        check("app.py compiles", FAIL, str(e))

    # Comments must be stripped before scanning. requirements.txt explains
    # *why* scikit-learn is banned, and the first version of this check read
    # that explanation as a violation.
    def declared(path):
        return "\n".join(
            l.split("#")[0].strip().lower()
            for l in path.read_text(encoding="utf-8").splitlines()
            if l.split("#")[0].strip()
        )

    reqs = declared(ROOT / "requirements.txt")
    heavy = [b for b in ("scikit-learn", "scipy", "torch", "tensorflow") if b in reqs]
    ok("app requirements exclude heavy ML libraries", not heavy,
       f"found {heavy} — the free tier caps memory at 1 GB")
    ok("app requirements include streamlit", "streamlit" in reqs)

    # THE APP MUST NOT MIX STREAMLIT API GENERATIONS.
    #
    # `use_container_width` was removed from CHART elements while it still
    # works on dataframes and images, so a half-migrated file fails on the
    # charts alone and passes every other check. That is exactly what
    # happened on 2026-09-22: the Model accuracy tab raised a TypeError on
    # the deployed site while everything rendered fine locally, because the
    # local Streamlit was a version behind the cloud's.
    #
    # Checked here rather than left to a version pin, because the pin cannot
    # stop the two styles coexisting.
    stale = re.findall(r"use_container_width\s*=", source)
    ok("app.py uses one Streamlit sizing API", not stale,
       f"{len(stale)} use(s) of use_container_width remain; the app also uses "
       'width="stretch", and mixing them breaks charts on newer Streamlit')


    # NOTHING AT MODULE LEVEL MAY SHADOW A FUNCTION DEFINED AT MODULE LEVEL.
    #
    # THE BUG THIS EXISTS FOR, and it was live for two days on the deployed
    # site: the Bench tab ran `for line in bb["reasoning"]`, and `line()` is
    # the chart function defined at the top of app.py. Streamlit executes the
    # whole script top to bottom on every interaction, and every `with tab:`
    # block runs whether or not anybody is looking at that tab, so by the
    # time the Model accuracy tab called `line(...)` it was a string.
    #
    # It hid for months because `line()` is only called from the second
    # scored gameweek onwards. GW5 was scored and the tab began raising
    # "TypeError: 'str' object is not callable" the same day.
    #
    # It also survived a fix. On 2026-09-22 this exact symptom was diagnosed
    # as the app mixing two Streamlit sizing APIs, which was a real problem
    # and was really fixed, and was NOT this. Reproducing the pre-fix file
    # afterwards gave the same 'str' object message, so the page was never
    # repaired. A check that reads the code is the only thing that would have
    # separated the two, because both are a TypeError on the same tab.
    tree = ast.parse(source)
    functions = {n.name for n in tree.body if isinstance(n, ast.FunctionDef)}

    def targets(node):
        """Every name a for-loop or with-statement binds."""
        out = []
        for child in ast.walk(node):
            if isinstance(child, ast.For):
                for t in ast.walk(child.target):
                    if isinstance(t, ast.Name):
                        out.append((t.id, t.lineno))
            elif isinstance(child, ast.withitem) and child.optional_vars:
                for t in ast.walk(child.optional_vars):
                    if isinstance(t, ast.Name):
                        out.append((t.id, t.lineno))
        return out

    shadowed = []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            continue          # a local name inside a function is fine
        for name, lineno in targets(node):
            if name in functions:
                shadowed.append(f"{name} at line {lineno}")

    ok("no module level name shadows a function in app.py", not shadowed,
       f"{shadowed} rebinds a module level function. Streamlit reruns this "
       "whole file on every interaction, so the function is gone for every "
       "later caller. This is what broke the Model accuracy tab")

    imports = re.findall(r"^\s*(?:import|from)\s+([\w.]+)", source, re.M)
    banned = [i for i in imports if i.split(".")[0] in
              ("sklearn", "scipy", "sqlite3", "torch", "tensorflow")]
    ok("app.py imports nothing heavy", not banned, f"imports {banned}")

    # The app must only read files the pipeline actually writes.
    loaded = set(re.findall(r'load\("(\w+)"\)', source))
    unknown = loaded - set(EXPECTED_FILES)
    ok("app only loads files the publisher produces", not unknown,
       f"app reads {unknown} which publish.py never writes")

    ok("app handles missing data instead of crashing",
       "st.stop()" in source and "is None" in source)

    pipeline_reqs = (ROOT / "requirements-pipeline.txt")
    ok("pipeline requirements exist separately", pipeline_reqs.exists())
    if pipeline_reqs.exists():
        pr = declared(pipeline_reqs)
        ok("pipeline requirements include scikit-learn", "scikit-learn" in pr)
        ok("pipeline requirements include scipy for the optimiser", "scipy" in pr)

    qa_house_rules(source)


# ------------------------------------------------------------ STATIC SITE

# The whole point of the static front end is that it is small and depends on
# nothing. Both of those are properties that decay silently: one CDN script
# added in a hurry, one 2 MB photograph dropped in the assets folder, and it
# is just another heavy page that happens not to use a framework.
#
# So the budget is a FAILURE, not a warning. Streamlit serves the same data
# for 5171 KB. If this ever needs more than a quarter of a megabyte, the
# argument for it has gone.
STATIC_BUDGET_KB = 250


def qa_static_site():
    print("\nSTATIC FRONT END")

    site = ROOT / "site"
    index = site / "index.html"
    appjs = site / "assets" / "app.js"
    css = site / "assets" / "pfl.css"

    if not ok("site/index.html exists", index.exists(),
              "the static front end is the low weight alternative to Streamlit"):
        return
    if not ok("site/assets/app.js exists", appjs.exists()):
        return

    html = index.read_text(encoding="utf-8")
    js = appjs.read_text(encoding="utf-8")

    # ---- nothing may be loaded from another host ------------------------
    #
    # A CDN is a third party that can change or vanish under a page nobody is
    # watching, and this one runs unattended between gameweeks. Checked on
    # the ATTRIBUTES rather than on the raw text, because both files discuss
    # URLs in their comments and a naive scan for "https://" flags the
    # explanation of the rule as a breach of it.
    external = []
    for attr in re.findall(r'(?:src|href)\s*=\s*"([^"]+)"', html):
        if re.match(r"https?://|//", attr):
            external.append(attr)
    for url in re.findall(r"""fetch\(\s*['"]([^'"]+)""", js):
        if re.match(r"https?://|//", url):
            external.append(url)
    ok("the static page loads nothing from another host", not external,
       f"external references: {external}")

    # ---- weight ---------------------------------------------------------
    page_files = [index, appjs, css]
    data_files = sorted((site / "data").glob("*.json"))
    photos = sorted((site / "assets" / "photos").glob("*.jpg")) + \
        sorted((site / "assets" / "photos").glob("*.webp"))

    page_kb = sum(f.stat().st_size for f in page_files) / 1024
    data_kb = sum(f.stat().st_size for f in data_files) / 1024
    photo_kb = sum(f.stat().st_size for f in photos) / 1024
    total_kb = page_kb + data_kb + photo_kb

    ok(f"the whole site fits in {STATIC_BUDGET_KB} KB",
       total_kb <= STATIC_BUDGET_KB,
       f"{total_kb:.0f} KB: page {page_kb:.0f}, data {data_kb:.0f}, "
       f"photos {photo_kb:.0f}. Streamlit serves the same data for 5171 KB; "
       "if this needs a bigger budget the reason for it has gone")
    check("weight breakdown", PASS,
          f"page {page_kb:.0f} KB, data {data_kb:.0f} KB, photos {photo_kb:.0f} KB, "
          f"total {total_kb:.0f} KB")

    # ---- it may only read files the publisher writes --------------------
    fetched = set()
    m = re.search(r"var FILES = \[(.*?)\];", js, re.S)
    if m:
        fetched = set(re.findall(r"'(\w+)'", m.group(1)))
    ok("the page only reads files the publisher produces",
       fetched and not (fetched - set(EXPECTED_FILES)),
       f"reads {sorted(fetched - set(EXPECTED_FILES))} which publish.py never writes")

    missing = [n for n in fetched if not (site / "data" / f"{n}.json").exists()]
    ok("every file the page reads has actually been published", not missing,
       f"missing: {missing}")

    # ---- the two front ends must agree ----------------------------------
    #
    # Two pages drawing one dataset is two places for the same fact to live.
    # These are the ones that have already tried to drift.
    acc_path = site / "data" / "accuracy.json"
    if acc_path.exists():
        acc = json.loads(acc_path.read_text(encoding="utf-8"))
        ok("accuracy.json names which model is the champion",
           bool(acc.get("champion")),
           "both front ends draw the champion in the kit colour and the "
           "baselines behind it. Without this key each one works it out by "
           "hardcoding the name, which is two copies of a fact that changes "
           "the day a challenger is promoted")
        if acc.get("champion"):
            ok("the champion is a model that is actually being scored",
               acc["champion"] in (acc.get("models") or []),
               f"{acc.get('champion')} is not in {acc.get('models')}")

    # NO FRONT END MAY HARDCODE THE CHAMPION'S NAME.
    #
    # Checked against code with comments stripped, because both files
    # legitimately discuss the name in their comments and the point is
    # whether either of them BRANCHES on it.
    for name, path in (("app.py", ROOT / "app.py"),
                       ("site/assets/app.js", appjs)):
        if not path.exists():
            continue
        text = path.read_text(encoding="utf-8")
        code = re.sub(r"/\*.*?\*/", "", text, flags=re.S)
        code = "\n".join(
            l.split("//")[0] if path.suffix == ".js" else l.split("#")[0]
            for l in code.splitlines())
        code = re.sub(r'"""[\s\S]*?"""', "", code)
        ok(f"{name} does not hardcode the champion's name",
           "two_stage_ml" not in code,
           "read it from accuracy.json instead, so promoting a challenger "
           "does not mean editing two front ends")

    # The noise floor is published, never typed. A second copy stays at 0.17
    # forever after somebody remeasures it, and the honesty of this whole
    # project rests on that number being current.
    js_code = re.sub(r"/\*.*?\*/", "", js, flags=re.S)
    js_code = "\n".join(l.split("//")[0] for l in js_code.splitlines())
    ok("the static page does not hardcode the noise floor",
       "0.17" not in js_code,
       "it arrives in transfers.json, measured by the pipeline")

    # ---- the photo captions must match ----------------------------------
    #
    # app.py globs the photo folder because Streamlit has a filesystem; the
    # static page reads photos.json because a file host has no directory
    # listing. Two implementations of one filename rule, so they are compared
    # against the files actually present.
    photos_json = site / "data" / "photos.json"
    if photos_json.exists() and photos:
        published = {p["file"]: p["caption"]
                     for p in json.loads(photos_json.read_text(encoding="utf-8"))}
        app_src = (ROOT / "app.py").read_text(encoding="utf-8")
        disagree = []
        if "def caption(path)" in app_src:
            for p in photos:
                stem = p.stem.split("_", 1)[-1] if p.stem[:2].isdigit() else p.stem
                expected = stem.replace("-", " ").replace("_", " ").strip()
                if published.get(p.name) != expected:
                    disagree.append(f"{p.name}: json={published.get(p.name)!r} "
                                    f"app.py={expected!r}")
        ok("both front ends caption the photographs identically", not disagree,
           "; ".join(disagree))
        listed = set(published)
        on_disk = {p.name for p in photos}
        ok("photos.json lists every photograph on disk", listed == on_disk,
           f"only in json: {sorted(listed - on_disk)}, "
           f"only on disk: {sorted(on_disk - listed)}. A static host has no "
           "directory listing, so a photo missing from this file is a photo "
           "nobody will ever see")

    # ---- house rules apply to this page too -----------------------------
    #
    # Only interface copy. The comments in these files have to be able to
    # name the things they ban.
    copy = re.sub(r"<!--[\s\S]*?-->", "", html)
    js_copy = re.sub(r"/\*[\s\S]*?\*/", "", js)
    js_copy = "\n".join(l.split("//")[0] for l in js_copy.splitlines())

    for label, text in (("index.html", copy), ("app.js", js_copy)):
        ok(f"no em dashes in {label} interface copy",
           "—" not in text,
           "house rule: no em dashes")
        emoji = [c for c in text if ord(c) > 0x2500]
        ok(f"no emoji used as icons in {label}", not emoji,
           f"house rule: found {emoji}")

    # ---- it must say something when nothing has been published ----------
    ok("the page handles missing data instead of rendering blank",
       "Nothing published yet" in js,
       "a page that renders an empty frame when the pipeline has not run "
       "looks broken rather than empty")

    ok("the page tells a reader without JavaScript where the data is",
       "<noscript>" in html and "site/data" in html)


# --------------------------------------------------------- HOUSE RULES

def qa_house_rules(app_source):
    """The five rules Percival gave for the interface, enforced.

    These are taste, not correctness, which is exactly why they need a check.
    Nothing breaks when an emoji creeps back into a heading, so nothing stops
    it, and six months later the page looks like every other dashboard. The
    same rules govern Bulawayo Chess Hub and live at the top of its
    static/css/bch.css.

    Only INTERFACE COPY is scanned. Comments and docstrings may discuss em
    dashes freely, and this file has to be able to name the thing it bans.
    """
    print("\nDESIGN HOUSE RULES")

    css = ROOT / "site" / "assets" / "pfl.css"
    if not ok("the design system exists", css.exists(),
              "site/assets/pfl.css is the single source of visual truth"):
        return
    css_text = css.read_text(encoding="utf-8")

    ok("the app loads the design system", "pfl.css" in app_source)
    ok("the house rules are written into the stylesheet",
       "HOUSE RULES" in css_text,
       "the rules must survive in the file, not only in someone's memory")

    # Interface copy means string literals. Walk the AST rather than the raw
    # text so a comment explaining a rule cannot trip the rule.
    literals = []
    try:
        for node in ast.walk(ast.parse(app_source)):
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                literals.append(node.value)
    except SyntaxError:
        check("house rules scanned", WARN, "app.py does not parse")
        return

    # Docstrings are documentation, not interface copy, and this module's own
    # docstring names the banned characters.
    copy = "\n".join(s for s in literals if len(s) < 400)

    EM_DASH = "—"
    EN_DASH = "–"
    ok("no em dashes in interface copy",
       EM_DASH not in copy and EN_DASH not in copy,
       "Percival named this first and it is the most common tell. "
       "Use a comma, a colon, or two sentences.")

    # Emoji, not every non-ASCII character: player names legitimately carry
    # accents and this must never flag "Joao Pedro" or "Hornicek".
    EMOJI = re.compile(
        "[" "\U0001F300-\U0001FAFF" "\U00002600-\U000027BF"
        "\U0001F000-\U0001F0FF" "\U0000FE0F" "\U00002B00-\U00002BFF"
        "\U00002190-\U000021FF" "\U00002700-\U000027BF" "]")
    # Reported by codepoint, never by printing the character. Printing it
    # raised UnicodeEncodeError on a Windows cp1252 console and took the whole
    # suite down with a traceback, so the check that found a real violation
    # destroyed the report it was supposed to appear in. A codepoint is also
    # the more useful thing to be told: it is searchable.
    found = sorted({f"U+{ord(c):04X}" for c in EMOJI.findall(copy + css_text)})
    ok("no emoji used as icons", not found,
       f"found {', '.join(found)}. Use a word: a word survives a font "
       f"fallback, reads correctly in a screen reader, and does not render "
       f"as an empty box on an older Android handset.")

    # A colour blend, not the function. Flat bands with hard stops are how the
    # pitch markings and the chess board motif are both drawn.
    blends = []
    for m in re.finditer(r"(linear|radial|conic)-gradient\(([^;]*)\)", css_text):
        body = m.group(2)
        stops = re.findall(r"(#[0-9a-fA-F]{3,8}|var\(--[\w-]+\)|transparent)"
                           r"\s+([\d.]+)%", body)
        # A hard stop repeats the same position for two colours, or uses the
        # "colour A B%" pairs that tile rather than fade.
        positions = [p for _, p in stops]
        if len(positions) >= 2 and len(set(positions)) == len(positions) \
                and "0 " not in body and not re.search(r"\d%\s+\d", body):
            blends.append(m.group(0)[:60])
    ok("no blended gradients in the stylesheet", not blends,
       f"{blends}. Flat bands with hard stops are fine and are how the pitch "
       f"markings are drawn. A fade between two colours is not.")

    ok("charts are on the palette", "SERIES" in app_source and "#1f7a3d" in app_source,
       "Streamlit's default blue belongs to no part of this design")


# ------------------------------------------------------------- WORKFLOW

def qa_workflow():
    print("\nGITHUB ACTIONS WORKFLOW")

    if not ok("workflow file exists", WORKFLOW.exists(), str(WORKFLOW)):
        return
    raw = WORKFLOW.read_text(encoding="utf-8")

    try:
        import yaml
    except ImportError:
        check("workflow YAML parses", WARN, "pyyaml not installed; skipped")
        return

    try:
        wf = yaml.safe_load(raw)
        check("workflow YAML parses", PASS)
    except yaml.YAMLError as e:
        check("workflow YAML parses", FAIL, str(e))
        return

    # PyYAML resolves a bare `on:` key to the boolean True — the well-known
    # "Norway problem". Accept either form rather than reporting a false failure.
    triggers = wf.get("on", wf.get(True, {})) or {}
    ok("workflow runs on a schedule", "schedule" in triggers,
       f"triggers: {list(triggers)}")
    ok("workflow can be run manually before a deadline",
       "workflow_dispatch" in triggers)

    crons = [c.get("cron") for c in triggers.get("schedule", [])]
    ok("at least two runs a day", len(crons) >= 2, f"{crons}")

    ok("workflow may write to the repository",
       wf.get("permissions", {}).get("contents") == "write",
       str(wf.get("permissions")))

    conc = wf.get("concurrency", {})
    ok("concurrent runs cannot race on the database",
       conc.get("group") and conc.get("cancel-in-progress") is False,
       str(conc))

    job = next(iter(wf.get("jobs", {}).values()), {})
    steps = job.get("steps", [])
    names = [s.get("name", s.get("uses", "")) for s in steps]

    ok("workflow has a timeout", "timeout-minutes" in job,
       f"{job.get('timeout-minutes')} minutes")

    def index_of(fragment):
        for i, n in enumerate(names):
            if fragment.lower() in n.lower():
                return i
        return -1

    qa_i, commit_i, email_i = index_of("QA"), index_of("Commit"), index_of("Email")
    ok("a QA gate exists in the workflow", qa_i >= 0)
    # Ordering is the whole point of the gate: bad data must never be committed
    # to a public page or emailed out.
    ok("QA runs before anything is committed", 0 <= qa_i < commit_i,
       f"QA at step {qa_i}, commit at step {commit_i}")
    ok("QA runs before anything is emailed", 0 <= qa_i < email_i,
       f"QA at step {qa_i}, email at step {email_i}")

    # ------------------------------------------------------------------
    # Most scripts here print their docstring and exit 0 when given no mode
    # flag. Invoking one bare from the workflow therefore looks like success
    # while doing nothing at all — `python backfill_history.py` loaded no
    # history, and the run only broke four steps later with "Training on 0
    # historical rows". A green step that did nothing is the worst kind of
    # failure, so every invocation is checked against the script's own flags.
    # ------------------------------------------------------------------
    # Comments must be stripped first. The workflow *explains* this very trap
    # in a comment containing a bare `python backfill_history.py`, and the
    # first version of this check flagged that prose as a real invocation.
    commands = "\n".join(
        l.split("#")[0] for l in raw.splitlines() if l.split("#")[0].strip()
    )
    invocations = re.findall(r"python\s+(\w+\.py)([^\n|;&]*)", commands)
    noop_risk = []
    for script, args in invocations:
        path = ROOT / script
        if not path.exists():
            noop_risk.append(f"{script} (missing)")
            continue
        src = path.read_text(encoding="utf-8")
        main_body = src.split("def main(", 1)[-1]
        # Does this script no-op without a flag?
        if "print(__doc__)" not in main_body:
            continue
        flags = set(re.findall(r'"(--[a-z_]+)"\s+in\s+sys\.argv', main_body))
        if flags and not any(f in args for f in flags):
            noop_risk.append(f"{script} (needs one of {sorted(flags)})")

    ok("no workflow step invokes a script that would silently do nothing",
       not noop_risk, "; ".join(noop_risk))

    ok("bootstrap asserts it actually loaded history",
       "history rows after bootstrap" in raw or "COUNT(*) FROM history" in raw,
       "a bootstrap that loads nothing must fail loudly, not four steps later")

    # "Store database" is a substring of "Restore database from release", so a
    # loose match here silently compared the restore step against itself and
    # reported a failure that did not exist. Anchor on the full step name.
    ok("the database is restored before use",
       0 <= index_of("Restore database") < index_of("Take snapshot"))
    ok("the database is stored back after the run",
       index_of("Store database back") > index_of("Publish JSON"))

    ok("no hardcoded credentials in the workflow",
       not re.search(r"(?i)(password|token)\s*:\s*['\"]?[A-Za-z0-9_\-]{16,}", raw))
    ok("mail credentials come from secrets",
       "secrets.MAIL_PASSWORD" in raw and "MAIL_PASSWORD:" in raw)
    ok("email step is skipped when secrets are absent",
       "env.MAIL_USERNAME != ''" in raw)


# -------------------------------------------------------------- HYGIENE

def qa_hygiene():
    print("\nREPOSITORY HYGIENE")

    gi = (ROOT / ".gitignore")
    if not ok(".gitignore exists", gi.exists()):
        return
    ignored = gi.read_text(encoding="utf-8")

    ok("fpl.db is not committed", "fpl.db" in ignored)
    ok(".env is not committed", ".env" in ignored)

    # The whole architecture depends on site/data being tracked. If someone
    # ever adds it to .gitignore the app silently freezes at its last state.
    lines = [l.strip() for l in ignored.splitlines()
             if l.strip() and not l.strip().startswith("#")]
    blocked = [l for l in lines if l.rstrip("/") in ("site", "site/data", "*.json")]
    ok("site/data is NOT gitignored", not blocked,
       f"{blocked} would stop the app ever updating")

    db = DB_PATH
    if db.exists():
        mb = db.stat().st_size / 1e6
        check("database size is under control",
              PASS if mb < 100 else WARN, f"{mb:.1f} MB")

    stray = [p.name for p in ROOT.glob("*.env")] + \
            [p.name for p in ROOT.glob("*.key")]
    ok("no stray secret files in the project root", not stray, f"{stray}")

    ok("README documents the deployment",
       "streamlit" in (ROOT / "README.md").read_text(encoding="utf-8").lower())


# --------------------------------------------------------------- report

def summary():
    n_fail = sum(1 for _, s, _ in results if s == FAIL)
    n_warn = sum(1 for _, s, _ in results if s == WARN)
    n_pass = sum(1 for _, s, _ in results if s == PASS)

    print("\n" + "=" * 62)
    print(f"  {n_pass} passed, {n_warn} warnings, {n_fail} failures "
          f"({len(results)} checks)")
    if n_fail:
        print("\n  FAILURES:")
        for name, s, d in results:
            if s == FAIL:
                print(f"    - {name}: {d}")
    if n_warn:
        print("\n  WARNINGS:")
        for name, s, d in results:
            if s == WARN:
                print(f"    - {name}: {d}")
    print("=" * 62)
    return n_fail


def main():
    if not DATA.exists():
        print(f"  Nothing published yet at {DATA}. Run:  python publish.py")
        sys.exit(1)

    meta = qa_contract()
    if meta is None:
        sys.exit(1 if summary() else 0)

    squad = qa_squad()
    qa_recommendations(meta)
    qa_captain(squad)
    qa_alerts_and_notify(meta, squad)

    if DB_PATH.exists():
        conn = sqlite3.connect(DB_PATH)
        try:
            qa_consistency(conn, meta, squad)
            qa_honesty(conn, meta)
            qa_benchmark(conn)
        finally:
            conn.close()
    else:
        check("database available for consistency checks", WARN, "fpl.db not found")

    qa_app()
    qa_static_site()
    qa_workflow()
    qa_hygiene()

    sys.exit(1 if summary() else 0)


if __name__ == "__main__":
    main()
