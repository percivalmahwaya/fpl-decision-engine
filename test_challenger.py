"""
Tests for challenger.py.

    python test_challenger.py

A challenger is a research model running in production. The two things that
matter are therefore not about accuracy at all:

  1. It must never write under the champion's name.
  2. It must never be able to break the champion's run.

Everything else is commentary.
"""
import re
import sqlite3
import unittest
from pathlib import Path

import pandas as pd

import challenger
from challenger import (CHALLENGER_NAME, attach_current_season_opponent,
                        opponents_by_gameweek)

ROOT = Path(__file__).resolve().parent


def fixtures_db():
    conn = sqlite3.connect(":memory:")
    conn.execute("CREATE TABLE fixtures (snapshot_id INT, event INT, "
                 "team_h INT, team_a INT)")
    conn.execute("CREATE TABLE predictions (gameweek INT, element_id INT, "
                 "model TEXT, predicted REAL, made_at TEXT, p60 REAL, cond REAL)")
    return conn


class IsolationFromTheChampionTest(unittest.TestCase):
    """The two properties that actually matter."""

    def test_it_never_writes_under_the_champions_name(self):
        """Checked against the PARSED code, not the raw text.

        The docstring quotes the champion's live scores, which is exactly
        the sort of thing a naive grep flags and a reader then learns to
        ignore. What matters is whether the name appears as a value the code
        could actually write.
        """
        import ast
        tree = ast.parse((ROOT / "challenger.py").read_text(encoding="utf-8"))

        docstrings = set()
        for node in ast.walk(tree):
            if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef)):
                doc = ast.get_docstring(node, clean=False)
                if doc:
                    docstrings.add(doc)

        literals = [n.value for n in ast.walk(tree)
                    if isinstance(n, ast.Constant) and isinstance(n.value, str)
                    and n.value not in docstrings]

        self.assertNotIn("two_stage_ml", literals,
                         "the challenger must not be able to write under the "
                         "champion's model name")
        names = [n.id for n in ast.walk(tree) if isinstance(n, ast.Name)]
        self.assertNotIn("MODEL_NAME", names)

    def test_the_name_is_distinct(self):
        self.assertNotEqual(CHALLENGER_NAME, "two_stage_ml")

    def test_a_failure_is_swallowed_rather_than_raised(self):
        """If this file has a bug, the worst case must be a missing row in the
        scoreboard, not a lost gameweek."""
        conn = fixtures_db()
        original = challenger.fit_and_predict
        challenger.fit_and_predict = lambda *a, **k: (_ for _ in ()).throw(
            RuntimeError("deliberate"))
        try:
            written = challenger.log(conn, None, None, 6, "now")
        finally:
            challenger.fit_and_predict = original
        self.assertEqual(written, 0)
        self.assertEqual(
            conn.execute("SELECT COUNT(*) FROM predictions").fetchone()[0], 0)

    def test_it_is_called_after_the_champion_is_logged(self):
        """Order matters: if the challenger ran first and hung, the champion's
        predictions would never be written at all."""
        source = (ROOT / "recommend.py").read_text(encoding="utf-8")
        champion_insert = source.index("INSERT OR REPLACE INTO predictions")
        challenger_call = source.index("challenger.log(")
        self.assertLess(champion_insert, challenger_call,
                        "the champion must be logged before the challenger runs")

    def test_nothing_reads_the_challenger_for_a_decision(self):
        """It must not reach the optimiser, the captain, or the squad."""
        for filename in ("optimise.py", "captain.py", "publish.py",
                         "transfers.py", "bench.py"):
            path = ROOT / filename
            if not path.exists():
                continue
            source = path.read_text(encoding="utf-8")
            self.assertNotIn(
                CHALLENGER_NAME, source,
                f"{filename} references the challenger; it is for scoring "
                "only and must drive no decision")


class OpponentLookupTest(unittest.TestCase):
    def test_each_team_gets_its_opponent_both_ways(self):
        conn = fixtures_db()
        conn.execute("INSERT INTO fixtures VALUES (7, 6, 1, 2)")
        table = opponents_by_gameweek(conn)
        pairs = {(r.team_id, r.opponent) for r in table.itertuples()}
        self.assertEqual(pairs, {(1, 2), (2, 1)})

    def test_only_the_latest_snapshot_is_read(self):
        """`fixtures` holds one row per fixture PER SNAPSHOT. Reading them all
        reports each team playing dozens of times a week, which would make
        every gameweek look like a double."""
        conn = fixtures_db()
        for snapshot in range(1, 20):
            conn.execute("INSERT INTO fixtures VALUES (?, 6, 1, 2)", (snapshot,))
        table = opponents_by_gameweek(conn)
        self.assertEqual(len(table), 2, "one fixture, two sides, once")

    def test_an_empty_fixture_table_does_not_crash(self):
        self.assertTrue(opponents_by_gameweek(fixtures_db()).empty)

    def test_players_get_their_teams_opponent(self):
        conn = fixtures_db()
        conn.execute("INSERT INTO fixtures VALUES (7, 6, 1, 2)")
        df = pd.DataFrame({"gw": [6, 6], "element": [10, 20],
                           "team_id": [1, 2]})
        out = attach_current_season_opponent(df, conn)
        self.assertEqual(list(out.sort_values("element")["opponent"]), [2, 1])

    def test_a_double_gameweek_does_not_duplicate_players(self):
        """Two fixtures in one gameweek would otherwise produce two rows per
        player, and the model would predict each of them twice."""
        conn = fixtures_db()
        conn.execute("INSERT INTO fixtures VALUES (7, 6, 1, 2)")
        conn.execute("INSERT INTO fixtures VALUES (7, 6, 3, 1)")
        df = pd.DataFrame({"gw": [6], "element": [10], "team_id": [1]})
        out = attach_current_season_opponent(df, conn)
        self.assertEqual(len(out), 1)

    def test_a_frame_with_no_team_id_is_returned_untouched(self):
        conn = fixtures_db()
        df = pd.DataFrame({"gw": [6], "element": [10]})
        self.assertEqual(len(attach_current_season_opponent(df, conn)), 1)


class ScoringPicksItUpAutomaticallyTest(unittest.TestCase):
    def test_the_accuracy_builder_is_model_agnostic(self):
        """No model names are hardcoded in the scoreboard, so a new one is
        graded without touching publish.py. If this ever stops being true,
        registering a challenger becomes a two-file change and somebody will
        do half of it."""
        source = (ROOT / "publish.py").read_text(encoding="utf-8")
        builder = source[source.index("def build_accuracy"):]
        builder = builder[:builder.index("\ndef ", 1)]
        self.assertIn("SELECT DISTINCT model FROM predictions", builder)


if __name__ == "__main__":
    unittest.main(verbosity=2, exit=False)
