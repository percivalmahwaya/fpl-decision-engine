"""
Tests for migrate.py.

    python test_migrate.py

THE BUG THESE EXIST FOR: on 2026-09-19 three tables gained columns, the
declarations were updated, and CREATE TABLE IF NOT EXISTS did nothing to the
databases already out there. CI restores its database from a release asset,
so it had none of the new columns, and the collector - the FIRST step in the
pipeline - died on the first insert. Three scheduled runs produced nothing at
all before anybody looked.

The most important test here is the last one: it reads the CREATE TABLE
statements out of the source and fails if a column exists in a declaration
but not in MIGRATIONS. Without it, this whole file just tests that ALTER
TABLE works, and the next column added silently repeats the same outage.
"""
import re
import sqlite3
import unittest
from pathlib import Path

from migrate import MIGRATIONS, ensure_columns, migrate

ROOT = Path(__file__).resolve().parent

# Which file declares which table.
DECLARED_IN = {
    "players": "fpl_collect.py",
    "predictions": "predict.py",
    "my_gameweeks": "record_my_team.py",
}


class EnsureColumnsTest(unittest.TestCase):
    def setUp(self):
        self.conn = sqlite3.connect(":memory:")
        self.conn.execute("CREATE TABLE t (a INTEGER, b TEXT)")

    def test_it_adds_a_missing_column(self):
        added = ensure_columns(self.conn, "t", {"c": "REAL"})
        self.assertEqual(added, ["c"])
        cols = {r[1] for r in self.conn.execute("PRAGMA table_info(t)")}
        self.assertIn("c", cols)

    def test_it_is_idempotent(self):
        ensure_columns(self.conn, "t", {"c": "REAL"})
        self.assertEqual(ensure_columns(self.conn, "t", {"c": "REAL"}), [],
                         "running twice must not error or duplicate")

    def test_it_leaves_existing_columns_alone(self):
        self.conn.execute("INSERT INTO t VALUES (1, 'keep me')")
        ensure_columns(self.conn, "t", {"c": "REAL"})
        self.assertEqual(
            self.conn.execute("SELECT b FROM t").fetchone()[0], "keep me",
            "migrating must never touch data")

    def test_a_table_that_does_not_exist_yet_is_skipped_not_created(self):
        """CREATE TABLE IF NOT EXISTS makes it a moment later. Trying to ALTER
        a table that is not there yet would crash a fresh database."""
        self.assertEqual(ensure_columns(self.conn, "nope", {"x": "INTEGER"}), [])

    def test_new_columns_are_null_on_existing_rows(self):
        """Rows written before a column existed genuinely have no value for
        it. Inventing a default would be inventing data."""
        self.conn.execute("INSERT INTO t VALUES (1, 'old row')")
        ensure_columns(self.conn, "t", {"c": "REAL"})
        self.assertIsNone(self.conn.execute("SELECT c FROM t").fetchone()[0])


class MigrateTest(unittest.TestCase):
    def test_it_brings_an_old_database_up_to_date(self):
        """The exact shape of the CI database on 2026-09-19."""
        conn = sqlite3.connect(":memory:")
        conn.executescript("""
            CREATE TABLE players (snapshot_id INT, element_id INT, raw TEXT);
            CREATE TABLE predictions (gameweek INT, element_id INT, predicted REAL);
            CREATE TABLE my_gameweeks (gameweek INT, total_points INT);
        """)
        added = migrate(conn, verbose=False)
        self.assertEqual(len(added), 8)

        cols = {r[1] for r in conn.execute("PRAGMA table_info(players)")}
        self.assertIn("penalties_order", cols)
        # The insert that actually failed in CI.
        conn.execute("INSERT INTO players (snapshot_id, penalties_order) "
                     "VALUES (1, 1)")

    def test_running_it_on_a_current_database_changes_nothing(self):
        conn = sqlite3.connect(":memory:")
        conn.executescript("""
            CREATE TABLE players (snapshot_id INT, penalties_order INT,
                                  freekicks_order INT, corners_order INT);
        """)
        self.assertEqual(migrate(conn, verbose=False), [])


class DeclarationsAndMigrationsAgreeTest(unittest.TestCase):
    """THE TEST THAT ACTUALLY PREVENTS A REPEAT.

    Everything above proves ALTER TABLE works, which was never in doubt. The
    outage happened because somebody (me) added a column to a CREATE TABLE
    statement and nowhere else. This reads the declarations back out of the
    source and fails if the two have drifted.
    """

    def _declared_columns(self, filename, table):
        source = (ROOT / filename).read_text(encoding="utf-8")
        match = re.search(
            r"CREATE TABLE IF NOT EXISTS\s+%s\s*\((.*?)\n\s*\);" % table,
            source, re.S)
        self.assertIsNotNone(
            match, f"could not find the {table} declaration in {filename}")

        columns = set()
        for line in match.group(1).splitlines():
            line = line.strip()
            if not line or line.startswith("--") or line.startswith("PRIMARY KEY"):
                continue
            columns.add(line.split()[0].strip(","))
        return columns

    def test_every_migrated_column_is_actually_declared(self):
        """A migration for a column no longer in the schema is dead weight."""
        for table, columns in MIGRATIONS.items():
            declared = self._declared_columns(DECLARED_IN[table], table)
            for name in columns:
                self.assertIn(
                    name, declared,
                    f"{table}.{name} is migrated but not declared in "
                    f"{DECLARED_IN[table]}")

    def test_no_column_was_added_to_a_schema_without_a_migration(self):
        """The direction that caused the outage.

        This is deliberately a hardcoded baseline rather than something
        clever: it is the set of columns that existed when the tables were
        first written. Anything beyond it must appear in MIGRATIONS.
        """
        original = {
            "players": {
                "snapshot_id", "element_id", "web_name", "team_id", "position",
                "price", "total_points", "form", "minutes", "starts",
                "selected_by_percent", "transfers_in_event",
                "transfers_out_event", "status", "chance_next_round", "news",
                "news_added", "ep_next", "raw",
            },
            "predictions": {
                "gameweek", "element_id", "model", "predicted", "made_at",
            },
            "my_gameweeks": {
                "gameweek", "total_points", "transfers", "formation",
                "decided_by",
            },
        }

        for table, baseline in original.items():
            declared = self._declared_columns(DECLARED_IN[table], table)
            added_since = declared - baseline
            migrated = set(MIGRATIONS.get(table, {}))
            missing = added_since - migrated
            self.assertEqual(
                missing, set(),
                f"{table}: {sorted(missing)} added to the CREATE TABLE in "
                f"{DECLARED_IN[table]} but NOT to MIGRATIONS in migrate.py. "
                "Every existing database will be missing them, and the first "
                "insert will fail. This is exactly what broke the pipeline "
                "on 2026-09-19.")



class NoPositionalInsertsIntoMigratedTablesTest(unittest.TestCase):
    """A table that gains columns must never be written to positionally.

    `INSERT INTO t VALUES (?,?,?)` requires the value count to match the
    column count EXACTLY. The moment such a table gains a column, every
    positional insert into it starts failing with

        table predictions has 7 columns but 5 values were supplied

    which is the second half of the 2026-09-19 outage. The migration fixed
    the schema, the run got one step further, and then died on this instead.
    recommend.py had been converted to named columns; predict.py had the same
    line and was missed.
    """

    SOURCES = ["fpl_collect.py", "predict.py", "recommend.py",
               "record_my_team.py", "backfill_history.py"]

    def test_no_migrated_table_is_written_positionally(self):
        offenders = []
        for filename in self.SOURCES:
            path = ROOT / filename
            if not path.exists():
                continue
            source = path.read_text(encoding="utf-8")
            for table in MIGRATIONS:
                # "INSERT ... INTO <table> VALUES" with no column list.
                pattern = r"INSERT[^\"']*INTO\s+%s\s+VALUES" % table
                for match in re.finditer(pattern, source):
                    line = source[:match.start()].count("\n") + 1
                    offenders.append(f"{filename}:{line} -> {table}")

        self.assertEqual(
            offenders, [],
            "positional INSERT into a table that gains columns: "
            + "; ".join(offenders)
            + ". Name the columns, or the next migration breaks the pipeline.")

if __name__ == "__main__":
    unittest.main(verbosity=2, exit=False)
