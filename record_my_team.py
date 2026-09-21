"""
Record Percival's actual FPL squad and results, gameweek by gameweek.

WHY THIS MATTERS: this is the benchmark. Any model we build has to beat these
decisions, not just beat a naive baseline. Without a record of what he actually
picked, and what it scored, there is no way to know whether the system is
helping or just producing confident-looking numbers.

Team: "Progeny"

HOW IT USED TO WORK, AND WHY THAT WAS WRONG
===========================================
Until 2026-09-15 the gameweeks below were the ONLY source: a dict transcribed
by hand from screenshots on 2026-09-07. The pipeline ran this script twice a
day, and it dutifully rewrote the same four gameweeks every time.

So the workflow step called "Record squad" went green forever while recording
nothing new. GW4 finished, was scored by every other part of the system, and
still showed `points: null` on the site, because no code anywhere could learn
his score without someone editing this file.

That is the same failure this project keeps finding: a step that reports
success having done nothing. It is now fetched from the FPL API instead.

WHAT IS FETCHED, AND WHAT STILL CANNOT BE
=========================================
Two public endpoints, no login and no token:

    entry/{id}/history/          points, transfers, hits, bench points per GW
    entry/{id}/event/{gw}/picks/ the fifteen picks, captain, vice, multipliers

Picks carry `element` ids directly, which retires the whole name matching
problem below: FPL web_names are not unique, and matching "Palmer" or
"Martinez" by name alone silently picked the wrong player more than once.

The one field that cannot be fetched is `decided_by`, because whether a call
came from instinct or from the engine is a fact about Percival, not about FPL.
It is kept as a small override map and defaults to "unrecorded".

CONFIGURATION
=============
    FPL_ENTRY_ID    his manager id. The number in the URL when he views his
                    own team: fantasy.premierleague.com/entry/NNNNNNN/event/4

Set as a repository VARIABLE (not a secret): an entry id is not a credential,
it grants no access, and anyone can already look up any public team with it.

Without it this script falls back to the transcription and says so loudly, and
qa_deploy.py raises a warning. With it set but failing, qa_deploy.py FAILS,
because that means the benchmark has silently stopped updating.

Usage:
    python record_my_team.py           # fetch if FPL_ENTRY_ID is set, else seed
    python record_my_team.py --seed    # force the hand transcription
    python record_my_team.py --show    # print what is recorded
    python record_my_team.py --verify  # compare the API against the transcription
"""

import json
import os
import sqlite3
import sys
import urllib.error
import urllib.request
from pathlib import Path
from migrate import migrate

DB_PATH = Path(__file__).parent / "fpl.db"
API = "https://fantasy.premierleague.com/api"
ENTRY_ID = os.environ.get("FPL_ENTRY_ID", "").strip()

# Whether a gameweek was decided by instinct or by the engine. The only field
# here that is a fact about the manager rather than about FPL, so the only one
# that cannot be fetched. Add a line when a gameweek is decided differently.
DECIDED_BY = {
    1: "gut",
    2: "gut",
    3: "gut",
    4: "gut+model",
}
DEFAULT_DECIDED_BY = "unrecorded"

# --- The hand transcription ------------------------------------------------
# Kept deliberately after the switch to fetching. It is no longer the source of
# truth; it is the CROSS CHECK. `--verify` scores the API against it, which is
# how we know the element id mapping and the multiplier arithmetic are right:
# two independent records of the same four gameweeks, one typed by a human from
# screenshots and one pulled from the API, agreeing.
#
# Each entry: (name, points, started?, is_captain, is_vice)
# points of None = did not play / no score shown.
GAMEWEEKS = {
    1: {
        "total_points": 53,
        "transfers": 0,
        "formation": "3-5-2",
        "squad": [
            ("Martinez",     None, True,  False, False),
            ("Virgil",          2, True,  False, False),
            ("Cash",           -1, True,  False, False),
            ("Shaw",            1, True,  False, False),
            ("B.Fernandes",     2, True,  False, False),
            ("Palmer",         13, True,  False, True),
            ("Szoboszlai",      8, True,  False, False),
            ("Mbeumo",          2, True,  False, False),
            ("Rice",            3, True,  False, False),
            ("Wood",            1, True,  False, False),
            ("João Pedro",     22, True,  True,  False),
            ("Phillips",     None, False, False, False),
            ("Palestra",     None, False, False, False),
            ("Gyökeres",     None, False, False, False),
            ("Mykolenko",       6, False, False, False),
        ],
    },
    2: {
        "total_points": 79,
        "transfers": 1,
        "formation": "3-5-2",
        "squad": [
            ("Martinez",        2, True,  False, False),
            ("Virgil",          1, True,  False, False),
            ("Shaw",            2, True,  False, False),
            ("Mykolenko",       4, True,  False, False),
            ("B.Fernandes",    23, True,  False, True),
            ("Palmer",          7, True,  False, False),
            ("Szoboszlai",      4, True,  False, False),
            ("Mbeumo",         11, True,  False, False),
            ("Rice",            5, True,  False, False),
            ("Havertz",         2, True,  False, False),
            ("João Pedro",     18, True,  True,  False),
            ("Phillips",     None, False, False, False),
            ("Wood",         None, False, False, False),
            ("Cash",            2, False, False, False),
            ("Palestra",     None, False, False, False),
        ],
    },
    3: {
        "total_points": 48,
        "transfers": 1,
        "formation": "4-4-2",
        "squad": [
            ("Martinez",        3, True,  False, False),
            ("Virgil",          6, True,  False, False),
            ("Cash",            6, True,  False, False),
            ("Shaw",            4, True,  False, False),
            ("Hall",            4, True,  False, False),
            ("B.Fernandes",     4, True,  True,  False),
            ("Palmer",          1, True,  False, False),
            ("Mbeumo",          8, True,  False, False),
            ("Szoboszlai",      3, True,  False, False),
            ("João Pedro",      1, True,  False, True),
            ("Havertz",         8, True,  False, False),
            ("Phillips",     None, False, False, False),
            ("Wood",            1, False, False, False),
            ("Rice",            5, False, False, False),
            ("Mykolenko",       1, False, False, False),
        ],
    },
    # ---- GW4: post-wildcard squad, XI SET, not yet played -----------------
    #
    # WHY THIS ENTRY EXISTS AT ALL, WITH NO POINTS IN IT: the wildcard was
    # played before GW4, replacing 10 of the 15 players above. Until this was
    # recorded, `my_squad` topped out at GW3, so everything that asks "who do I
    # own?" — alert.py's [SQUAD] tag, optimise.py --transfers — was answering
    # with a team two thirds of which had been sold. The alerter was warning
    # about Cash's muscular injury (sold) while filing Gakpo, an actual owned
    # player, under [WATCH].
    #
    # Confirmed from the FPL app 2026-09-11, ~19h before the deadline:
    # Gakpo -> Wirtz is DONE (Gakpo was a 75% thigh doubt; Wirtz is clean), the
    # XI is picked in a 3-5-2, and PALMER IS CAPTAIN.
    #
    # `decided_by` is "gut+model" and the distinction matters for every future
    # comparison. The 15 came from his own wildcard, made before this project
    # existed. The Gakpo -> Wirtz swap was agreed jointly off the availability
    # alert. The captaincy — the single biggest lever in FPL — was HIS call
    # AGAINST the model, which ranked B.Fernandes ahead of Palmer. Scoring this
    # gameweek therefore settles a real disagreement, not a formality.
    4: {
        "total_points": None,          # not played yet
        "transfers": 1,                # Gakpo -> Wirtz; free, wildcard active
        "formation": "3-5-2",
        "decided_by": "gut+model",
        "squad": [
            # name            pts  started captain vice
            #
            # Keeper switched Horníček -> Pickford on 2026-09-11 after the
            # comparison: same fixture difficulty, so the fixture overlay does
            # not move it, and the model had Pickford ahead 3.38 to 3.09. The
            # only selection disagreement of the gameweek, now resolved in the
            # engine's favour.
            ("Pickford",     None, True,  False, False),
            ("Rúben",        None, True,  False, False),
            ("Mitchell",     None, True,  False, False),
            ("Hall",         None, True,  False, False),
            ("Rogers",       None, True,  False, False),
            ("Palmer",       None, True,  True,  False),   # captain — his call
            ("B.Fernandes",  None, True,  False, False),   # model wanted him captain
            ("Szoboszlai",   None, True,  False, False),
            ("Wirtz",        None, True,  False, False),   # in for Gakpo
            ("João Pedro",   None, True,  False, True),    # vice
            ("Isak",         None, True,  False, False),
            # bench, in order
            ("Horníček",     None, False, False, False),
            ("Barry",        None, False, False, False),
            ("Thomas",       None, False, False, False),
            ("Davis",        None, False, False, False),
        ],
    },
}


# --- Schema ----------------------------------------------------------------

def create_schema(conn):
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS my_gameweeks (
            gameweek     INTEGER PRIMARY KEY,
            total_points INTEGER,
            transfers    INTEGER,
            formation    TEXT,
            decided_by   TEXT DEFAULT 'gut',
            -- All three arrive on the same history call that was already
            -- being made for points, and all three were being dropped.
            -- `bank` is the one that matters: without it the transfer
            -- optimiser assumed zero money and could only ever suggest
            -- swaps that were exactly affordable, which is almost none of
            -- them. In tenths of a million, as the API sends it.
            bank         INTEGER,
            squad_value  INTEGER,
            bench_points INTEGER
        );

        CREATE TABLE IF NOT EXISTS my_squad (
            gameweek   INTEGER NOT NULL,
            name       TEXT NOT NULL,
            element_id INTEGER,
            points     INTEGER,               -- NULL = not yet scored
            started    INTEGER,
            is_captain INTEGER,
            is_vice    INTEGER,
            PRIMARY KEY (gameweek, name)
        );
        """
    )


# --- The API ---------------------------------------------------------------

def get(path):
    """One GET against the public FPL API."""
    req = urllib.request.Request(f"{API}/{path}",
                                 headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=30) as fh:
        return json.load(fh)

    # Existing databases do not get new columns from CREATE TABLE
    # IF NOT EXISTS. See migrate.py: omitting this killed three
    # scheduled runs on 2026-09-19.
    migrate(conn, verbose=True)



def element_names(conn):
    """element_id -> (web_name, element_type) from the most recent snapshot."""
    row = conn.execute("SELECT MAX(snapshot_id) FROM players").fetchone()
    if not row or row[0] is None:
        return {}
    return {
        eid: (name, pos)
        for eid, name, pos in conn.execute(
            "SELECT element_id, web_name, position FROM players WHERE snapshot_id=?",
            (row[0],))
    }


def actual_points(conn, gw):
    """element_id -> points actually scored in that gameweek."""
    return dict(conn.execute(
        "SELECT element_id, total_points FROM player_gw WHERE gameweek=?", (gw,)))


POS_ORDER = {"GKP": 0, "DEF": 1, "MID": 2, "FWD": 3}


def formation_of(starters):
    """'3-5-2' from the outfield starters. Goalkeeper is implied, as in FPL."""
    counts = {"DEF": 0, "MID": 0, "FWD": 0}
    for pos in starters:
        if pos in counts:
            counts[pos] += 1
    return f"{counts['DEF']}-{counts['MID']}-{counts['FWD']}"


def fetch(conn):
    """Pull every gameweek this entry has played, and store it.

    Every gameweek is refetched, not just the newest. Bonus points land a day
    after a match, an appeal can change a score, and FPL occasionally corrects
    one retrospectively. Fetching only the latest would freeze whatever was
    true at the moment of the first read.
    """
    if not ENTRY_ID:
        return False

    try:
        history = get(f"entry/{ENTRY_ID}/history/")
    except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError) as ex:
        print(f"  COULD NOT REACH THE FPL API for entry {ENTRY_ID}: {ex}")
        print("  Nothing written. The benchmark is now stale, and qa_deploy.py "
              "will fail on it rather than let that pass quietly.")
        return False

    played = history.get("current", [])
    if not played:
        print(f"  Entry {ENTRY_ID} has no gameweeks yet.")
        return True

    names = element_names(conn)
    print(f"  Entry {ENTRY_ID}: {len(played)} gameweek(s) reported by the API.")

    for row in played:
        gw = row["event"]
        try:
            picks = get(f"entry/{ENTRY_ID}/event/{gw}/picks/")
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError) as ex:
            print(f"  GW{gw}: picks unavailable ({ex}), skipped")
            continue

        scored = actual_points(conn, gw)
        starters = []
        squad_rows = []

        for p in picks.get("picks", []):
            eid = p["element"]
            name, pos = names.get(eid, (f"element {eid}", None))
            mult = p.get("multiplier", 0)
            started = 1 if mult > 0 else 0
            if started and pos:
                starters.append(pos)

            # The FPL app shows a captain's points already doubled, and the
            # hand transcription copied what the app showed. Multiplying here
            # keeps the two records comparable, and keeps a triple captain
            # honest too.
            raw = scored.get(eid)
            pts = None if raw is None else raw * max(mult, 1) if started else raw

            squad_rows.append((gw, name, eid, pts, started,
                               1 if p.get("is_captain") else 0,
                               1 if p.get("is_vice_captain") else 0))

        conn.execute("DELETE FROM my_squad WHERE gameweek=?", (gw,))
        conn.executemany(
            "INSERT INTO my_squad (gameweek, name, element_id, points, started,"
            " is_captain, is_vice) VALUES (?,?,?,?,?,?,?)", squad_rows)

        # event_transfers_cost is the hit, in points. Recorded as transfers
        # made; the cost shows up in total_points already.
        conn.execute(
            "INSERT OR REPLACE INTO my_gameweeks"
            " (gameweek, total_points, transfers, formation, decided_by,"
            "  bank, squad_value, bench_points)"
            " VALUES (?,?,?,?,?,?,?,?)",
            (gw, row.get("points"), row.get("event_transfers"),
             formation_of(starters),
             DECIDED_BY.get(gw, DEFAULT_DECIDED_BY),
             row.get("bank"), row.get("value"), row.get("points_on_bench")))

        chip = picks.get("active_chip")
        unscored = sum(1 for r in squad_rows if r[3] is None)
        print(f"  GW{gw}: {row.get('points')} pts, "
              f"{row.get('event_transfers')} transfer(s), "
              f"{formation_of(starters)}"
              + (f", chip {chip}" if chip else "")
              + (f", {unscored} player(s) not yet scored" if unscored else ""))

    conn.commit()
    return True


# --- The hand transcription, as a fallback and as a cross check -------------

POSITIONS = {
    "Martinez": "GKP", "Phillips": "GKP",
    "Virgil": "DEF", "Cash": "DEF", "Shaw": "DEF", "Hall": "DEF",
    "Mykolenko": "DEF", "Palestra": "DEF",
    "B.Fernandes": "MID", "Palmer": "MID", "Szoboszlai": "MID",
    "Mbeumo": "MID", "Rice": "MID",
    "Wood": "FWD", "João Pedro": "FWD", "Havertz": "FWD", "Gyökeres": "FWD",
    "Pickford": "GKP", "Horníček": "GKP",
    "Thomas": "DEF", "Mitchell": "DEF", "Davis": "DEF", "Rúben": "DEF",
    "Rogers": "MID", "Gakpo": "MID", "Wirtz": "MID",
    "Barry": "FWD", "Isak": "FWD",
}


def seed(conn):
    """Write the hand transcription. Used only when there is no entry id."""
    for gw, data in GAMEWEEKS.items():
        conn.execute(
            "INSERT OR REPLACE INTO my_gameweeks"
            " (gameweek, total_points, transfers, formation, decided_by)"
            " VALUES (?,?,?,?,?)",
            (gw, data["total_points"], data["transfers"], data["formation"],
             DECIDED_BY.get(gw, DEFAULT_DECIDED_BY)))
        for name, pts, started, cap, vice in data["squad"]:
            conn.execute(
                "INSERT OR REPLACE INTO my_squad"
                " (gameweek, name, element_id, points, started, is_captain, is_vice)"
                " VALUES (?,?,?,?,?,?,?)",
                (gw, name, None, pts, int(started), int(cap), int(vice)))
    conn.commit()
    match_element_ids(conn)
    print(f"  Seeded {len(GAMEWEEKS)} transcribed gameweek(s).")


def match_element_ids(conn):
    """Link transcribed names to element_ids.

    Only needed for the seeded path. Fetched picks carry element ids already,
    which is the entire reason fetching is better: FPL web_names are not
    unique, there are two "Palmer" and two "Martinez", and matching on name
    alone silently picked the wrong player.
    """
    row = conn.execute("SELECT MAX(snapshot_id) FROM players").fetchone()
    if not row or row[0] is None:
        print("  (no snapshot yet, run fpl_collect.py first to enable matching)")
        return
    snap = row[0]

    unmatched, ambiguous, fixed = [], [], 0
    for (name,) in conn.execute(
            "SELECT DISTINCT name FROM my_squad WHERE element_id IS NULL").fetchall():
        pos = POSITIONS.get(name)
        if pos:
            cands = conn.execute(
                "SELECT element_id FROM players"
                " WHERE snapshot_id=? AND web_name=? AND position=?",
                (snap, name, pos)).fetchall()
        else:
            cands = conn.execute(
                "SELECT element_id FROM players WHERE snapshot_id=? AND web_name=?",
                (snap, name)).fetchall()

        if len(cands) == 1:
            conn.execute("UPDATE my_squad SET element_id=? WHERE name=? "
                         "AND element_id IS NULL", (cands[0][0], name))
            fixed += 1
        elif len(cands) > 1:
            ambiguous.append(f"{name} ({len(cands)} candidates)")
        else:
            unmatched.append(name)

    conn.commit()
    print(f"  Matched {fixed} player(s) by (name, position).")
    if ambiguous:
        print(f"  AMBIGUOUS, not guessed: {', '.join(ambiguous)}")
    if unmatched:
        print(f"  Could not match: {', '.join(unmatched)}")


def verify(conn):
    """Score what the API returned against what was typed from screenshots.

    Two independent records of the same gameweeks. If they agree, the element
    id mapping and the captain multiplier arithmetic are right. If they do not,
    one of them is wrong and it matters which.
    """
    if not ENTRY_ID:
        print("  FPL_ENTRY_ID is not set, so there is nothing to verify against.")
        return 1

    print("\nAPI versus the hand transcription\n")
    problems = 0
    for gw, data in sorted(GAMEWEEKS.items()):
        row = conn.execute(
            "SELECT total_points, transfers, formation FROM my_gameweeks"
            " WHERE gameweek=?", (gw,)).fetchone()
        if not row:
            print(f"  GW{gw}: not in the database at all")
            problems += 1
            continue

        # On a chip week FPL reports event_transfers as 0, because a wildcard
        # or free hit makes transfers unlimited and free so it does not count
        # them at all. The transcription recorded the change actually made.
        # Neither record is wrong; they answer different questions, and
        # comparing them is meaningless. Found on the first real run of this
        # check, against GW4, which was a wildcard.
        chip = None
        try:
            chip = get(f"entry/{ENTRY_ID}/event/{gw}/picks/").get("active_chip")
        except Exception:
            pass

        checks = [
            ("total points", data["total_points"], row[0]),
            ("formation", data["formation"], row[2]),
        ]
        if chip:
            print(f"  GW{gw}: chip '{chip}' played, so transfers are not "
                  f"compared (FPL reports 0 on a chip week)")
        else:
            checks.append(("transfers", data["transfers"], row[1]))
        for label, typed, stored in checks:
            # An unplayed gameweek has no transcribed score to compare.
            if typed is None or stored is None:
                continue
            if str(typed) != str(stored):
                print(f"  GW{gw} {label}: transcribed {typed!r}, API {stored!r}")
                problems += 1

        typed_caps = {n for n, _, _, c, _ in data["squad"] if c}
        stored_caps = {n for (n,) in conn.execute(
            "SELECT name FROM my_squad WHERE gameweek=? AND is_captain=1", (gw,))}
        if typed_caps and stored_caps and typed_caps != stored_caps:
            print(f"  GW{gw} captain: transcribed {typed_caps}, API {stored_caps}")
            problems += 1

    if problems:
        print(f"\n  {problems} disagreement(s). One of the two records is wrong.")
    else:
        print("  Every transcribed gameweek agrees with the API.")
    return 1 if problems else 0


# --- Reporting -------------------------------------------------------------

def show(conn):
    rows = conn.execute(
        "SELECT gameweek, total_points, transfers, formation, decided_by"
        " FROM my_gameweeks ORDER BY gameweek").fetchall()
    if not rows:
        print("Nothing recorded.")
        return

    print(f"\n{'GW':>3}  {'PTS':>4}  {'TR':>3}  {'FORMATION':<10} {'DECIDED BY':<12} CAPTAIN")
    print("  " + "-" * 62)
    total = 0
    played = 0
    for gw, pts, tr, form, by in rows:
        cap = conn.execute(
            "SELECT name, points FROM my_squad WHERE gameweek=? AND is_captain=1",
            (gw,)).fetchone()
        cap_txt = f"{cap[0]} ({cap[1]})" if cap and cap[1] is not None else (
            cap[0] if cap else "none")
        print(f"{gw:>3}  {'' if pts is None else pts:>4}  {tr:>3}  "
              f"{form or '':<10} {by or '':<12} {cap_txt}")
        if pts is not None:
            total += pts
            played += 1
    print("  " + "-" * 62)
    if played:
        print(f"  {played} gameweek(s) scored, {total} points, "
              f"{total / played:.1f} average\n")


def main():
    conn = sqlite3.connect(DB_PATH)
    try:
        create_schema(conn)

        if "--show" in sys.argv:
            show(conn)
            return 0
        if "--verify" in sys.argv:
            return verify(conn)

        if "--seed" in sys.argv:
            seed(conn)
        elif ENTRY_ID:
            if not fetch(conn):
                return 1
        else:
            # Loud on purpose. The whole reason this file was rewritten is that
            # it used to do nothing in silence, twice a day, for over a week.
            print("  FPL_ENTRY_ID IS NOT SET.")
            print("  Falling back to the hand transcription, which means this")
            print("  benchmark stops at the last gameweek somebody typed in by")
            print("  hand and will never learn a new score on its own.")
            print("  Set it to the number in fantasy.premierleague.com/entry/NNNNNNN/")
            seed(conn)

        show(conn)
        return 0
    finally:
        conn.close()


if __name__ == "__main__":
    raise SystemExit(main())
