"""
Add columns to tables that already exist.

WHY THIS EXISTS
===============
Every schema in this project is declared with CREATE TABLE IF NOT EXISTS,
which is exactly right for a fresh database and does NOTHING to an existing
one. Add a column to the declaration and a new database gets it; every
database already out there silently does not.

That broke the pipeline on 2026-09-19. Three tables gained columns that day
- `players` gained set-piece duty, `predictions` gained the two halves of the
two-stage prediction, `my_gameweeks` gained the bank - and all three were
migrated by hand with ALTER TABLE on the local copy. CI restores its database
from a GitHub release asset, so it had none of them, and the first insert
after the change failed with:

    table players has no column named penalties_order

Three scheduled runs died before anyone looked. The collector is the first
step in the pipeline, so nothing downstream ran at all: no predictions, no
alerts, no email.

THE RULE THIS ENFORCES
======================
A column added to a CREATE TABLE statement must also be added here. Calling
`ensure_columns` is idempotent and costs nothing on a database that already
has them, so it runs on every startup rather than being something to remember.
"""
import sqlite3


def ensure_columns(conn, table, columns):
    """Add any of `columns` the table is missing. {name: sql_type}.

    Idempotent. SQLite's ALTER TABLE ADD COLUMN cannot add a NOT NULL column
    without a default, which is fine: every column added this way is nullable
    on purpose, because rows written before it existed genuinely do not have
    a value and pretending otherwise would be inventing data.
    """
    existing = {row[1] for row in
                conn.execute(f"PRAGMA table_info({table})").fetchall()}
    if not existing:
        return []          # table does not exist yet; CREATE TABLE will make it

    added = []
    for name, sql_type in columns.items():
        if name in existing:
            continue
        conn.execute(f"ALTER TABLE {table} ADD COLUMN {name} {sql_type}")
        added.append(name)

    if added:
        conn.commit()
    return added


# Every column added to an existing table since the schemas were first
# written. Keep this list in step with the CREATE TABLE statements; the test
# in test_migrate.py fails if they drift apart.
MIGRATIONS = {
    "players": {
        "penalties_order": "INTEGER",
        "freekicks_order": "INTEGER",
        "corners_order": "INTEGER",
    },
    "predictions": {
        "p60": "REAL",
        "cond": "REAL",
    },
    "my_gameweeks": {
        "bank": "INTEGER",
        "squad_value": "INTEGER",
        "bench_points": "INTEGER",
    },
}


def normalise_history_positions(conn, verbose=True):
    """Rewrite the archive's "GK" as "GKP", once.

    THE BUG THIS CLOSES: `history` stores goalkeepers as "GK" and every other
    part of this project - the live API, the model's position filter, the
    is_gkp feature, the optimiser, the bench - uses "GKP". model.py filters
    on the live spelling, so 12,500 rows, 11% of the archive, never reached
    training. `is_gkp` was a constant zero while 71 goalkeepers a week were
    predicted from the live feed regardless. Nothing crashed, nothing warned,
    and it went unnoticed for months.

    Data migration rather than a read-time alias, because twenty call sites
    already agree that a goalkeeper is "GKP" and the archive is the only
    dissenter. One of them should change, and it should be the odd one out.

    Idempotent: a second run matches nothing.
    """
    if not conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='history'"
    ).fetchone():
        return 0

    cur = conn.execute("UPDATE history SET position = 'GKP' WHERE position = 'GK'")
    conn.commit()
    if cur.rowcount and verbose:
        print(f"  normalised {cur.rowcount:,} history rows: GK -> GKP")
    return cur.rowcount


def migrate(conn, verbose=True):
    """Bring any database up to the current schema. Safe to call always."""
    total = []
    for table, columns in MIGRATIONS.items():
        added = ensure_columns(conn, table, columns)
        if added and verbose:
            print(f"  migrated {table}: added {', '.join(added)}")
        total.extend(f"{table}.{c}" for c in added)

    moved = normalise_history_positions(conn, verbose=verbose)
    if moved:
        total.append(f"history.position GK->GKP x{moved}")
    return total


if __name__ == "__main__":
    import sys
    db = sys.argv[1] if len(sys.argv) > 1 else "fpl.db"
    conn = sqlite3.connect(db)
    added = migrate(conn)
    print(f"  {db}: {len(added)} column(s) added"
          if added else f"  {db}: already up to date")
