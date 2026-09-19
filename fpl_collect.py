"""
FPL data collector — Phase 0.

Takes a timestamped snapshot of the Fantasy Premier League API and appends it
to a local SQLite database.

WHY SNAPSHOTS: the FPL API only ever shows the CURRENT state. Prices, ownership,
injury news and availability change constantly, and once they change the old
values are gone forever. Every run here appends a new snapshot rather than
overwriting, so you build a history you can later train on. A gameweek you
didn't capture cannot be recovered.

Standard library only — no pip install required.

Usage:
    python fpl_collect.py              # take a snapshot
    python fpl_collect.py --summary    # show what's in the database
"""

import json
import sqlite3
import sys
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

BASE = "https://fantasy.premierleague.com/api"
DB_PATH = Path(__file__).parent / "fpl.db"
UA = "fpl-research/0.1 (personal research project)"

# Positions come back as ints; map them once for readability.
POSITIONS = {1: "GKP", 2: "DEF", 3: "MID", 4: "FWD"}


def fetch(endpoint):
    """GET an FPL endpoint and return parsed JSON."""
    url = f"{BASE}/{endpoint}"
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read().decode("utf-8"))


def create_schema(conn):
    """Create tables if they don't exist. Safe to call on every run."""
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS snapshots (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            taken_at      TEXT NOT NULL,
            current_gw    INTEGER,
            next_gw       INTEGER,
            next_deadline TEXT
        );

        CREATE TABLE IF NOT EXISTS players (
            snapshot_id         INTEGER NOT NULL,
            element_id          INTEGER NOT NULL,
            web_name            TEXT,
            team_id             INTEGER,
            position            TEXT,
            price               REAL,
            total_points        INTEGER,
            form                REAL,
            minutes             INTEGER,
            starts              INTEGER,
            selected_by_percent REAL,
            transfers_in_event  INTEGER,
            transfers_out_event INTEGER,
            status              TEXT,
            chance_next_round   INTEGER,
            news                TEXT,
            news_added          TEXT,
            ep_next             REAL,
            -- Set-piece duty. Arrives free on every bootstrap-static call and
            -- was being kept only inside the `raw` blob, where nothing read
            -- it. 1 means first choice. A first-choice penalty taker has a
            -- scoring route no model feature captures: penalties are the
            -- highest-probability shot in football and they are assigned by
            -- the manager, not earned by form.
            penalties_order     INTEGER,
            freekicks_order     INTEGER,
            corners_order       INTEGER,
            raw                 TEXT,
            PRIMARY KEY (snapshot_id, element_id)
        );

        CREATE TABLE IF NOT EXISTS teams (
            snapshot_id INTEGER NOT NULL,
            team_id     INTEGER NOT NULL,
            name        TEXT,
            short_name  TEXT,
            strength    INTEGER,
            PRIMARY KEY (snapshot_id, team_id)
        );

        CREATE TABLE IF NOT EXISTS fixtures (
            snapshot_id   INTEGER NOT NULL,
            fixture_id    INTEGER NOT NULL,
            event         INTEGER,
            kickoff_time  TEXT,
            team_h        INTEGER,
            team_a        INTEGER,
            difficulty_h  INTEGER,
            difficulty_a  INTEGER,
            finished      INTEGER,
            PRIMARY KEY (snapshot_id, fixture_id)
        );

        CREATE INDEX IF NOT EXISTS idx_players_element ON players(element_id);
        CREATE INDEX IF NOT EXISTS idx_fixtures_event  ON fixtures(event);
        """
    )


def to_float(value, default=0.0):
    """FPL returns several numeric fields as strings; coerce safely."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def take_snapshot(conn):
    print("Fetching bootstrap-static ...")
    boot = fetch("bootstrap-static/")
    print("Fetching fixtures ...")
    fixtures = fetch("fixtures/")

    events = boot.get("events", [])
    current = next((e for e in events if e.get("is_current")), None)
    nxt = next((e for e in events if e.get("is_next")), None)

    taken_at = datetime.now(timezone.utc).isoformat()
    cur = conn.execute(
        "INSERT INTO snapshots (taken_at, current_gw, next_gw, next_deadline) VALUES (?, ?, ?, ?)",
        (
            taken_at,
            current["id"] if current else None,
            nxt["id"] if nxt else None,
            nxt["deadline_time"] if nxt else None,
        ),
    )
    snap_id = cur.lastrowid

    # --- players ---
    rows = []
    for p in boot.get("elements", []):
        rows.append(
            (
                snap_id,
                p["id"],
                p.get("web_name"),
                p.get("team"),
                POSITIONS.get(p.get("element_type"), "?"),
                p.get("now_cost", 0) / 10.0,          # API stores tenths of a million
                p.get("total_points"),
                to_float(p.get("form")),
                p.get("minutes"),
                p.get("starts"),
                to_float(p.get("selected_by_percent")),
                p.get("transfers_in_event"),
                p.get("transfers_out_event"),
                p.get("status"),
                p.get("chance_of_playing_next_round"),
                p.get("news") or None,
                p.get("news_added"),
                to_float(p.get("ep_next")),
                p.get("penalties_order"),
                p.get("direct_freekicks_order"),
                p.get("corners_and_indirect_freekicks_order"),
                json.dumps(p, separators=(",", ":")),  # keep everything, for later
            )
        )
    # Named columns. Nineteen positional question marks is a row that breaks
    # silently and invisibly the moment the table gains a field, and it just
    # gained three.
    conn.executemany(
        "INSERT OR REPLACE INTO players (snapshot_id, element_id, web_name, "
        "team_id, position, price, total_points, form, minutes, starts, "
        "selected_by_percent, transfers_in_event, transfers_out_event, "
        "status, chance_next_round, news, news_added, ep_next, "
        "penalties_order, freekicks_order, corners_order, raw) "
        "VALUES (" + ",".join("?" * 22) + ")", rows
    )

    # --- teams ---
    conn.executemany(
        "INSERT OR REPLACE INTO teams VALUES (?,?,?,?,?)",
        [
            (snap_id, t["id"], t.get("name"), t.get("short_name"), t.get("strength"))
            for t in boot.get("teams", [])
        ],
    )

    # --- fixtures ---
    conn.executemany(
        "INSERT OR REPLACE INTO fixtures VALUES (?,?,?,?,?,?,?,?,?)",
        [
            (
                snap_id,
                f["id"],
                f.get("event"),
                f.get("kickoff_time"),
                f.get("team_h"),
                f.get("team_a"),
                f.get("team_h_difficulty"),
                f.get("team_a_difficulty"),
                1 if f.get("finished") else 0,
            )
            for f in fixtures
        ],
    )

    conn.commit()

    flagged = sum(1 for r in rows if r[15])  # news column
    print()
    print(f"  Snapshot #{snap_id} saved at {taken_at}")
    print(f"  Players:  {len(rows)}   (with injury/news text: {flagged})")
    print(f"  Fixtures: {len(fixtures)}")
    if current:
        print(f"  Current gameweek: {current['id']}")
    if nxt:
        print(f"  Next deadline:    {nxt['deadline_time']}  (GW{nxt['id']})")
    return snap_id


# How many recent snapshots keep their full JSON blob. One is enough to
# diagnose an FPL schema change; older ones are pure storage cost.
KEEP_RAW_SNAPSHOTS = 1


def prune_raw(conn, keep=KEEP_RAW_SNAPSHOTS, verbose=True):
    """
    Clear the `raw` JSON blob from all but the newest snapshot(s).

    WHY: `players.raw` stores the complete FPL JSON for all 654 players on every
    run — about 1.7 MB per snapshot, twice a day. Measured, that is 94% of all
    database growth and would project to roughly 3 GB per season, exhausting a
    5 GB Railway volume within two seasons.

    Nothing in the codebase reads this column; it exists only so that a field we
    did not think to store can still be recovered. Keeping it for the newest
    snapshot preserves that safety net at ~1.7 MB instead of gigabytes.

    Safe to run repeatedly. Rows are kept, only the blob is nulled.
    """
    keep_ids = [r[0] for r in conn.execute(
        "SELECT id FROM snapshots ORDER BY id DESC LIMIT ?", (keep,))]
    if not keep_ids:
        return 0

    placeholders = ",".join("?" * len(keep_ids))
    before = conn.execute(
        f"SELECT COUNT(*) FROM players WHERE raw IS NOT NULL"
        f" AND snapshot_id NOT IN ({placeholders})", keep_ids
    ).fetchone()[0]
    if before == 0:
        if verbose:
            print("  Nothing to prune.")
        return 0

    conn.execute(
        f"UPDATE players SET raw = NULL WHERE snapshot_id NOT IN ({placeholders})",
        keep_ids,
    )
    conn.commit()
    if verbose:
        print(f"  Pruned raw JSON from {before:,} player rows "
              f"(kept snapshot(s) {keep_ids}).")
    return before


def vacuum(conn, verbose=True):
    """
    Reclaim the freed pages. SQLite does not shrink the file on DELETE/UPDATE,
    so without this the space stays allocated and the whole exercise is
    pointless. VACUUM cannot run inside a transaction.
    """
    before = DB_PATH.stat().st_size
    conn.isolation_level = None          # leave implicit-transaction mode
    conn.execute("VACUUM")
    conn.isolation_level = ""            # restore default
    after = DB_PATH.stat().st_size
    if verbose:
        print(f"  Vacuumed: {before/1e6:.1f} MB -> {after/1e6:.1f} MB "
              f"(reclaimed {(before-after)/1e6:.1f} MB)")
    return before, after


def summary(conn):
    snaps = conn.execute(
        "SELECT id, taken_at, current_gw, next_gw FROM snapshots ORDER BY id"
    ).fetchall()
    if not snaps:
        print("No snapshots yet. Run:  python fpl_collect.py")
        return

    print(f"{len(snaps)} snapshot(s) in {DB_PATH.name}:\n")
    for s in snaps:
        n = conn.execute(
            "SELECT COUNT(*) FROM players WHERE snapshot_id=?", (s[0],)
        ).fetchone()[0]
        print(f"  #{s[0]:<3} {s[1][:19]}   GW{s[2]}  ->  GW{s[3]}   {n} players")

    print("\nMost-owned players in the latest snapshot:")
    latest = snaps[-1][0]
    for row in conn.execute(
        """SELECT web_name, position, price, selected_by_percent, total_points
           FROM players WHERE snapshot_id=?
           ORDER BY selected_by_percent DESC LIMIT 8""",
        (latest,),
    ):
        # Plain ASCII only: the default Windows console codepage mangles
        # symbols like the pound sign.
        print(f"  {row[0]:<16} {row[1]}  {row[2]:>5.1f}m  {row[3]:>5.1f}% owned  {row[4]:>3} pts")

    print("\nCurrently flagged (injury / availability):")
    for row in conn.execute(
        """SELECT web_name, status, chance_next_round, news
           FROM players WHERE snapshot_id=? AND news IS NOT NULL
           ORDER BY selected_by_percent DESC LIMIT 8""",
        (latest,),
    ):
        chance = "n/a" if row[2] is None else f"{row[2]}%"
        print(f"  {row[0]:<16} [{row[1]}] {chance:>4}  {row[3][:52]}")


def main():
    conn = sqlite3.connect(DB_PATH)
    try:
        create_schema(conn)
        if "--summary" in sys.argv:
            summary(conn)
        elif "--prune" in sys.argv:
            # Manual/one-off cleanup. The scheduled path prunes automatically.
            prune_raw(conn)
            vacuum(conn)
        else:
            take_snapshot(conn)
            # Prune AND vacuum on every run.
            #
            # Pruning alone is not enough, and this was measured rather than
            # assumed: UPDATE ... SET raw = NULL shrinks rows in place, leaving
            # gaps inside pages rather than whole free pages (freelist_count
            # stayed at 0). New inserts cannot use those fragments, so the file
            # kept extending by ~2.7 MB per snapshot — worse than doing nothing.
            # VACUUM rewrites the file and is what actually reclaims the space.
            prune_raw(conn, verbose=False)
            vacuum(conn, verbose=False)
            print(f"\n  Database: {DB_PATH}")
            print("  See what's stored with:  python fpl_collect.py --summary")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
