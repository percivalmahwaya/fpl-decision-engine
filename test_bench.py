"""
Tests for bench.py.

The ordering result is the thing that matters here, because the WRONG answer
is the intuitive one. Anybody maintaining this later will look at
`order_bench` sorting by `points_if_played`, think "surely that should be
expected points", change it, and every test that only checks "a good player
is near the front" will stay green. So the ordering is pinned by brute force
against the actual objective function rather than by example.

    python test_bench.py
"""
import itertools
import sqlite3
import sys
import unittest

from bench import (
    BENCH_BOOST_FLOOR, Player, autosub_value, bench_boost_advice,
    bench_boost_value, build_plan, can_cover_anyone, can_substitute,
    choose_starting_xi, find_double_gameweeks, formation_of, is_legal_xi,
    order_bench,
)


def p(name, position, p_play, points_if_played, **kw):
    return Player(element_id=abs(hash(name)) % 100000, name=name,
                  position=position, p_play=p_play,
                  points_if_played=points_if_played, **kw)


def brute_force_best_order(bench):
    """The objective, evaluated over every permutation.

    This is the definition of correct. order_bench has to match it.
    """
    def gain(order):
        total, remaining = 0.0, 1.0
        for player in order:
            total += remaining * player.p_play * player.points_if_played
            remaining *= 1 - player.p_play
        return total

    return max(itertools.permutations(bench), key=gain), gain


class OrderingTheoremTest(unittest.TestCase):
    """Order by E[points | played], not by expected points."""

    def test_the_worked_example_from_the_docstring(self):
        a = p("A", "MID", 0.95, 2.0)
        b = p("B", "MID", 0.40, 5.0)
        c = p("C", "MID", 0.70, 3.0)

        ordered = order_bench([a, b, c])
        self.assertEqual([x.name for x in ordered], ["B", "C", "A"])

        _, gain = brute_force_best_order([a, b, c])
        by_expected_points = sorted([a, b, c], key=lambda x: -x.expected_points)
        self.assertAlmostEqual(gain(ordered), 3.602, places=3)
        self.assertAlmostEqual(gain(by_expected_points), 3.042, places=3)

    def test_ordering_by_expected_points_is_measurably_worse(self):
        """The mistake this module exists to prevent, priced."""
        bench = [p("A", "MID", 0.95, 2.0),
                 p("B", "MID", 0.40, 5.0),
                 p("C", "MID", 0.70, 3.0)]
        _, gain = brute_force_best_order(bench)
        correct = gain(order_bench(bench))
        intuitive = gain(sorted(bench, key=lambda x: -x.expected_points))
        self.assertGreater(correct - intuitive, 0.5,
                           "sorting by expected points should cost real points")

    def test_it_matches_brute_force_on_many_random_benches(self):
        """The general claim, not one example.

        If somebody 'fixes' order_bench to sort by expected_points, this fails
        on a large fraction of these cases.
        """
        import random
        rng = random.Random(20260919)
        mismatches = 0
        for _ in range(400):
            bench = [p(f"P{i}", "MID", rng.uniform(0.05, 0.99),
                       rng.uniform(0.2, 8.0)) for i in range(3)]
            best, gain = brute_force_best_order(bench)
            if abs(gain(order_bench(bench)) - gain(best)) > 1e-9:
                mismatches += 1
        self.assertEqual(mismatches, 0,
                         f"{mismatches}/400 benches ordered suboptimally")

    def test_probability_of_playing_does_not_change_the_order(self):
        """The surprising half of the result: p cancels out.

        Two players keep their relative order however their availability
        changes, as long as E[points | played] keeps its order.
        """
        for p_low in (0.05, 0.3, 0.6, 0.95):
            bench = [p("Reliable", "MID", 0.99, 2.0),
                     p("Explosive", "MID", p_low, 6.0)]
            self.assertEqual([x.name for x in order_bench(bench)],
                             ["Explosive", "Reliable"],
                             f"order flipped at p={p_low}")

    def test_the_goalkeeper_is_always_first_and_never_reordered(self):
        """Not a choice. A benched keeper can only replace the keeper, so FPL
        fixes them in the first slot."""
        bench = [p("Winger", "MID", 0.8, 5.0),
                 p("Keeper", "GKP", 0.9, 4.0),
                 p("Fullback", "DEF", 0.7, 3.0)]
        ordered = order_bench(bench)
        self.assertEqual(ordered[0].name, "Keeper")

    def test_ties_are_broken_stably(self):
        """Two identical players must not swap places between page loads."""
        bench = [p("Zeta", "MID", 0.8, 3.0), p("Alpha", "MID", 0.8, 3.0)]
        self.assertEqual([x.name for x in order_bench(bench)],
                         [x.name for x in order_bench(bench)])


class StartingXiTest(unittest.TestCase):
    def squad(self):
        return (
            [p("GK1", "GKP", 0.99, 4.0), p("GK2", "GKP", 0.10, 3.0)]
            + [p(f"D{i}", "DEF", 0.9, 4.0 - i * 0.3) for i in range(5)]
            + [p(f"M{i}", "MID", 0.9, 6.0 - i * 0.4) for i in range(5)]
            + [p(f"F{i}", "FWD", 0.9, 5.0 - i * 0.5) for i in range(3)]
        )

    def test_the_xi_is_legal(self):
        xi, bench = choose_starting_xi(self.squad())
        self.assertTrue(is_legal_xi(xi))
        self.assertEqual(len(bench), 4)

    def test_exactly_one_goalkeeper_starts(self):
        xi, bench = choose_starting_xi(self.squad())
        self.assertEqual(sum(1 for x in xi if x.position == "GKP"), 1)
        self.assertEqual(sum(1 for b in bench if b.position == "GKP"), 1)

    def test_a_greedy_pick_would_be_illegal_and_this_is_not(self):
        """Greedy by expected points starts six midfielders here, which FPL
        will not accept. Exhaustive search over formations is the reason."""
        squad = (
            [p("GK1", "GKP", 0.99, 4.0), p("GK2", "GKP", 0.10, 1.0)]
            + [p(f"D{i}", "DEF", 0.9, 1.0) for i in range(5)]
            + [p(f"M{i}", "MID", 0.95, 9.0) for i in range(5)]
            + [p(f"F{i}", "FWD", 0.9, 1.0) for i in range(3)]
        )
        xi, _ = choose_starting_xi(squad)
        self.assertTrue(is_legal_xi(xi))
        self.assertLessEqual(sum(1 for x in xi if x.position == "MID"), 5)

    def test_the_best_players_start(self):
        squad = self.squad()
        xi, bench = choose_starting_xi(squad)
        worst_starter = min(x.expected_points for x in xi
                            if x.position != "GKP")
        best_outfield_sub = max((b.expected_points for b in bench
                                 if b.position != "GKP"), default=0)
        # Not a strict inequality: formation rules can legitimately bench a
        # higher-scoring player than one who starts.
        self.assertGreaterEqual(worst_starter, best_outfield_sub - 2.0)

    def test_a_squad_of_the_wrong_size_is_refused(self):
        with self.assertRaises(ValueError):
            build_plan([p("Only", "MID", 0.9, 5.0)])


class FormationTest(unittest.TestCase):
    def test_formation_string(self):
        xi = ([p("GK", "GKP", 1, 4)]
              + [p(f"D{i}", "DEF", 1, 4) for i in range(3)]
              + [p(f"M{i}", "MID", 1, 4) for i in range(5)]
              + [p(f"F{i}", "FWD", 1, 4) for i in range(2)])
        self.assertEqual(formation_of(xi), "3-5-2")

    def test_two_defenders_is_illegal(self):
        xi = ([p("GK", "GKP", 1, 4)]
              + [p(f"D{i}", "DEF", 1, 4) for i in range(2)]
              + [p(f"M{i}", "MID", 1, 4) for i in range(5)]
              + [p(f"F{i}", "FWD", 1, 4) for i in range(3)])
        self.assertFalse(is_legal_xi(xi))

    def test_substitution_depends_on_who_blanked_not_just_who_is_benched(self):
        """THE RULE THE FIRST VERSION GOT WRONG.

        Asking "could this sub come on for anybody" is almost always yes,
        because a like-for-like swap leaves the formation untouched. FPL's
        real rule is narrower: the sub replaces the SPECIFIC player who
        blanked, and that has to be legal.

        Behind a 3-4-3, a midfielder cannot cover a blanking defender -- that
        leaves two at the back -- but can cover a blanking midfielder.
        """
        defenders = [p(f"D{i}", "DEF", 1, 4) for i in range(3)]
        midfielders = [p(f"M{i}", "MID", 1, 4) for i in range(4)]
        xi = ([p("GK", "GKP", 1, 4)] + defenders + midfielders
              + [p(f"F{i}", "FWD", 1, 4) for i in range(3)])
        self.assertTrue(is_legal_xi(xi))

        spare_mid = p("M9", "MID", 1, 4)
        self.assertFalse(can_substitute(xi, defenders[0], spare_mid),
                         "covering a defender with a midfielder in a back "
                         "three leaves two at the back")
        self.assertTrue(can_substitute(xi, midfielders[0], spare_mid),
                        "like for like is always fine")

    def test_a_fourth_forward_is_not_stranded_because_it_replaces_a_forward(self):
        """The case that exposed the bug. A fourth forward behind a 3-4-3 can
        come on: it swaps with one of the three already playing."""
        xi = ([p("GK", "GKP", 1, 4)]
              + [p(f"D{i}", "DEF", 1, 4) for i in range(3)]
              + [p(f"M{i}", "MID", 1, 4) for i in range(4)]
              + [p(f"F{i}", "FWD", 1, 4) for i in range(3)])
        self.assertTrue(can_cover_anyone(xi, p("F4", "FWD", 1, 4)))


class BenchBoostTest(unittest.TestCase):
    def test_value_is_the_sum_of_expected_points(self):
        bench = [p("A", "GKP", 0.9, 4.0), p("B", "MID", 0.8, 5.0),
                 p("C", "DEF", 0.5, 3.0), p("D", "FWD", 0.2, 6.0)]
        self.assertAlmostEqual(bench_boost_value(bench),
                               0.9 * 4 + 0.8 * 5 + 0.5 * 3 + 0.2 * 6)

    def test_a_weak_bench_is_told_to_hold(self):
        advice = bench_boost_advice(gameweek=6, value_now=5.0)
        self.assertEqual(advice.verdict, "hold")
        self.assertTrue(any("rule of thumb" in r for r in advice.reasoning))

    def test_a_coming_double_gameweek_beats_good_form(self):
        """The one thing here that is knowable rather than predicted."""
        advice = bench_boost_advice(
            gameweek=6, value_now=20.0,
            recent=[(3, 8.0), (4, 9.0), (5, 10.0)], doubles=[6, 12])
        self.assertEqual(advice.verdict, "hold for the double")
        self.assertIn(12, advice.double_gameweeks)
        self.assertNotIn(6, advice.double_gameweeks,
                         "a double in the CURRENT gameweek is not something "
                         "to wait for")

    def test_an_exceptional_week_with_no_double_says_play_it(self):
        advice = bench_boost_advice(
            gameweek=6, value_now=22.0,
            recent=[(3, 8.0), (4, 9.0), (5, 10.0)], doubles=[])
        self.assertEqual(advice.verdict, "play it")

    def test_it_always_says_ordering_stops_mattering(self):
        """The one week the rest of this module is irrelevant, said out loud."""
        advice = bench_boost_advice(gameweek=6, value_now=20.0)
        self.assertTrue(any("order" in r.lower() for r in advice.reasoning))

    def test_no_history_does_not_crash_and_says_so(self):
        advice = bench_boost_advice(gameweek=1, value_now=14.0)
        self.assertTrue(any("no history" in r for r in advice.reasoning))


class DoubleGameweekDetectionTest(unittest.TestCase):
    """Read from the LATEST snapshot only.

    The fixtures table keeps one row per fixture PER SNAPSHOT. Counting across
    all of them reports every team as playing forty-three times a week, which
    would make every gameweek look like a double and the advice worthless.
    """

    def setUp(self):
        self.conn = sqlite3.connect(":memory:")
        self.conn.execute(
            "CREATE TABLE fixtures (snapshot_id INT, fixture_id INT, event INT,"
            " team_h INT, team_a INT)")

    def add(self, snapshot, event, home, away):
        self.conn.execute(
            "INSERT INTO fixtures VALUES (?,?,?,?,?)",
            (snapshot, abs(hash((snapshot, event, home, away))) % 99999,
             event, home, away))

    def test_a_normal_gameweek_is_not_a_double(self):
        for snapshot in (1, 2):
            self.add(snapshot, 6, 1, 2)
            self.add(snapshot, 6, 3, 4)
        self.assertEqual(find_double_gameweeks(self.conn, 1), [])

    def test_the_same_fixture_across_snapshots_is_not_a_double(self):
        """The trap. Without filtering to one snapshot this reports GW6 as a
        double purely because the fixture was collected forty times."""
        for snapshot in range(1, 41):
            self.add(snapshot, 6, 1, 2)
        self.assertEqual(find_double_gameweeks(self.conn, 1), [])

    def test_a_real_double_is_found(self):
        self.add(7, 12, 1, 2)
        self.add(7, 12, 3, 1)   # team 1 plays twice in GW12
        self.assertEqual(find_double_gameweeks(self.conn, 1), [12])

    def test_past_gameweeks_are_ignored(self):
        self.add(7, 3, 1, 2)
        self.add(7, 3, 3, 1)
        self.assertEqual(find_double_gameweeks(self.conn, 5), [])

    def test_an_empty_fixture_table_does_not_crash(self):
        self.assertEqual(find_double_gameweeks(self.conn, 1), [])


class WholePlanTest(unittest.TestCase):
    def squad(self):
        return (
            [p("Pickford", "GKP", 0.98, 3.4), p("Hornicek", "GKP", 0.08, 2.7)]
            + [p("Ruben", "DEF", 0.92, 3.9), p("Hall", "DEF", 0.88, 3.7),
               p("Mitchell", "DEF", 0.80, 3.0), p("Davis", "DEF", 0.30, 2.8),
               p("Thomas", "DEF", 0.25, 2.4)]
            + [p("Fernandes", "MID", 0.95, 5.9), p("Rogers", "MID", 0.90, 5.5),
               p("Palmer", "MID", 0.93, 5.1), p("Szoboszlai", "MID", 0.85, 4.7),
               p("Wirtz", "MID", 0.40, 6.2)]
            + [p("Isak", "FWD", 0.90, 5.2), p("Barry", "FWD", 0.75, 4.4),
               p("JoaoPedro", "FWD", 0.75, 4.6, flagged=True)]
        )

    def test_a_plan_is_coherent(self):
        plan = build_plan(self.squad())
        self.assertEqual(len(plan.starting_xi), 11)
        self.assertEqual(len(plan.bench), 4)
        self.assertTrue(is_legal_xi(plan.starting_xi))
        self.assertEqual(plan.bench[0].position, "GKP")
        self.assertGreater(plan.bench_boost_value, 0)

    def test_wirtz_is_benched_but_ordered_first_among_outfielders(self):
        """The result in one player. Wirtz has the squad's highest
        E[points|played] at 6.2 and a 40% chance of appearing, so expected
        points benches him -- correctly, a starter must actually play -- while
        the ordering puts him first, because if somebody blanks he is the most
        valuable cover available."""
        plan = build_plan(self.squad())
        benched = [b.name for b in plan.bench]
        self.assertIn("Wirtz", benched)
        outfield = [b.name for b in plan.bench if b.position != "GKP"]
        self.assertEqual(outfield[0], "Wirtz")

    def test_a_flag_on_a_starter_is_called_out(self):
        plan = build_plan(self.squad())
        self.assertTrue(any("flag" in n.lower() for n in plan.notes))

    def test_autosub_value_is_positive_and_finite(self):
        plan = build_plan(self.squad())
        self.assertGreater(plan.autosub_value, 0)
        self.assertLess(plan.autosub_value, 50)


if __name__ == "__main__":
    unittest.main(verbosity=2, exit=False)
