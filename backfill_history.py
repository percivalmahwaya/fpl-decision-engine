"""
Load historical FPL seasons into fpl.db.

WHY: the current season gives ~1,900 player-gameweeks, which proved far too
few to tell two models apart (see the Phase 2 step 2 ranking test — the two
candidate models produced an identical top 20). Four historical seasons give
roughly 110,000 rows, which is enough to actually train and validate.

Source: github.com/vaastav/Fantasy-Premier-League  (merged_gw.csv per season)
These files already include expected_goals / expected_assists / ict_index, so
the xG-style features OpenFPL relies on come for free — no Understat scraping
needed to get started.

NOTE ON IDs: the `element` id is season-specific — the same number refers to
different players in different seasons. Always key on (season, element), and
join across seasons by name if you ever need to.

Standard library only.

Usage:
    python backfill_history.py --load     # download + store all seasons
    python backfill_history.py --verify   # sanity-check what was loaded
"""

import csv
import io
import sqlite3
import sys
import urllib.request
from pathlib import Path

DB_PATH = Path(__file__).parent / "fpl.db"
RAW = "https://raw.githubusercontent.com/vaastav/Fantasy-Premier-League/master/data"
SEASONS = ["2022-23", "2023-24", "2024-25", "2025-26"]
UA = "fpl-research/0.1 (personal research project)"



# The upstream archive spells goalkeeper "GK". Everything else in this
# project - the live API, fpl_collect.POSITIONS, the model's is_gkp feature,
# the optimiser, the bench - spells it "GKP".
#
# That single-letter difference silently dropped 12,500 rows, 11% of the
# archive, out of every training run for months, because model.py filters on
# the live spelling and nothing ever matched. `is_gkp` was a constant zero
# while 71 goalkeepers a week were predicted anyway. Nothing crashed and
# nothing warned.
#
# Normalised HERE, at the boundary where the data enters, rather than taught
# to the twenty call sites downstream that already agree with each other.
HISTORY_POSITION_ALIASES = {"GK": "GKP"}


def normalise_position(position):
    """Map the archive's spelling onto the one the rest of the system uses."""
    return HISTORY_POSITION_ALIASES.get(position, position)


def create_schema(conn):
    conn.executescript(
        """
        -- One row per player PER FIXTURE, not per gameweek.
        -- In a double gameweek a player has two fixtures in the same GW, so
        -- (season, gw, element) is NOT unique — keying on that silently drops
        -- one fixture per DGW appearance, which is precisely the data you most
        -- care about (triple-captain and chip weeks). `fixture` is in the key.
        -- For per-gameweek totals, SUM over fixtures grouped by (season, gw, element).
        CREATE TABLE IF NOT EXISTS history (
            season        TEXT    NOT NULL,
            gw            INTEGER NOT NULL,
            element       INTEGER NOT NULL,
            fixture       INTEGER NOT NULL,
            name          TEXT,
            position      TEXT,
            team          TEXT,
            opponent_team INTEGER,
            was_home      INTEGER,
            minutes       INTEGER,
            starts        INTEGER,
            total_points  INTEGER,
            xp            REAL,      -- FPL's own expected points, a free benchmark
            goals         INTEGER,
            assists       INTEGER,
            clean_sheets  INTEGER,
            bps           INTEGER,
            ict           REAL,
            influence     REAL,
            creativity    REAL,
            threat        REAL,
            xg            REAL,
            xa            REAL,
            xgi           REAL,
            xgc           REAL,
            value         INTEGER,   -- price in tenths of a million
            selected      INTEGER,   -- ownership count
            PRIMARY KEY (season, gw, element, fixture)
        );
        CREATE INDEX IF NOT EXISTS idx_hist_player  ON history(season, element);
        CREATE INDEX IF NOT EXISTS idx_hist_gw      ON history(season, gw);
        CREATE INDEX IF NOT EXISTS idx_hist_minutes ON history(minutes);
        """
    )


def num(row, key, cast=float, default=None):
    """Columns vary between seasons; missing or blank -> default."""
    v = row.get(key)
    if v is None or v == "":
        return default
    try:
        return cast(float(v))
    except (TypeError, ValueError):
        return default


def load_season(conn, season):
    url = f"{RAW}/{season}/gws/merged_gw.csv"
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=120) as resp:
        text = resp.read().decode("utf-8", errors="replace")

    reader = csv.DictReader(io.StringIO(text))
    rows = []
    for r in reader:
        gw = num(r, "GW", int) or num(r, "round", int)
        el = num(r, "element", int)
        if gw is None or el is None:
            continue
        fx = num(r, "fixture", int, -1)
        rows.append((
            season, gw, el, fx,
            r.get("name"), normalise_position(r.get("position")), r.get("team"),
            num(r, "opponent_team", int),
            1 if str(r.get("was_home", "")).lower() == "true" else 0,
            num(r, "minutes", int, 0),
            num(r, "starts", int, 0),
            num(r, "total_points", int, 0),
            num(r, "xP"),
            num(r, "goals_scored", int, 0),
            num(r, "assists", int, 0),
            num(r, "clean_sheets", int, 0),
            num(r, "bps", int, 0),
            num(r, "ict_index"),
            num(r, "influence"),
            num(r, "creativity"),
            num(r, "threat"),
            num(r, "expected_goals"),
            num(r, "expected_assists"),
            num(r, "expected_goal_involvements"),
            num(r, "expected_goals_conceded"),
            num(r, "value", int),
            num(r, "selected", int),
        ))

    conn.executemany(
        "INSERT OR REPLACE INTO history VALUES (" + ",".join("?" * 27) + ")", rows
    )
    conn.commit()
    have_xg = sum(1 for r in rows if r[21] is not None)
    print(f"  {season}: {len(rows):>6} rows   (with xG data: {have_xg})")
    return len(rows)


def load(conn):
    total = 0
    for s in SEASONS:
        try:
            total += load_season(conn, s)
        except Exception as e:                      # keep going if one season fails
            print(f"  {s}: FAILED - {e}")
    print(f"\n  Total historical rows: {total:,}")


def verify(conn):
    rows = conn.execute(
        "SELECT season, COUNT(*), COUNT(DISTINCT element), MAX(gw) FROM history"
        " GROUP BY season ORDER BY season"
    ).fetchall()
    if not rows:
        print("  Nothing loaded. Run: python backfill_history.py --load")
        return

    print(f"  {'season':<10}{'rows':>9}{'players':>10}{'GWs':>6}")
    for s, n, p, g in rows:
        print(f"  {s:<10}{n:>9,}{p:>10}{g:>6}")
    total = conn.execute("SELECT COUNT(*) FROM history").fetchone()[0]
    print(f"  {'TOTAL':<10}{total:>9,}")

    print("\n  Minutes distribution across ALL history (compare to the 3-GW sample):")
    z, part, full = conn.execute(
        "SELECT SUM(minutes=0), SUM(minutes>0 AND minutes<60), SUM(minutes>=60) FROM history"
    ).fetchone()
    tot = z + part + full
    print(f"    0 min    {z:>7,}  ({100*z/tot:.1f}%)")
    print(f"    1-59     {part:>7,}  ({100*part/tot:.1f}%)")
    print(f"    60+      {full:>7,}  ({100*full/tot:.1f}%)")

    all_pts = conn.execute("SELECT SUM(total_points) FROM history").fetchone()[0]
    pts60 = conn.execute(
        "SELECT SUM(total_points) FROM history WHERE minutes>=60"
    ).fetchone()[0]
    print(f"\n    Share of all points from 60+ minute appearances: {100*pts60/all_pts:.1f}%")

    print("\n  FPL's own expected-points column (xP) as a free benchmark:")
    row = conn.execute(
        "SELECT COUNT(*), AVG(ABS(xp - total_points)) FROM history WHERE xp IS NOT NULL"
    ).fetchone()
    hi = conn.execute(
        "SELECT COUNT(*), AVG(ABS(xp - total_points)) FROM history"
        " WHERE xp IS NOT NULL AND total_points >= 5"
    ).fetchone()
    if row[0]:
        print(f"    overall      MAE {row[1]:.3f}  (n={row[0]:,})")
        print(f"    high-return  MAE {hi[1]:.3f}  (n={hi[0]:,})   <- the number to beat")


def main():
    conn = sqlite3.connect(DB_PATH)
    try:
        create_schema(conn)
        if "--load" in sys.argv:
            load(conn)
        elif "--verify" in sys.argv:
            verify(conn)
        else:
            print(__doc__)
    finally:
        conn.close()


if __name__ == "__main__":
    main()
