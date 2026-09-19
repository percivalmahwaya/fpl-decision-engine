"""
Tests for transfers.py.

    python test_transfers.py

The rules being enforced here are FPL's, not the model's, and they are the
kind that fail silently: a suggestion that breaks the three-per-club limit or
overspends the bank looks perfectly reasonable on screen and is simply
impossible to make.
"""
import unittest

from transfers import (HIT_COST, MAX_PER_CLUB, NOISE_FLOOR, Candidate,
                       suggest, summarise)


def c(name, pos="MID", club="BHA", price=5.0, ep=4.0, **kw):
    return Candidate(element_id=abs(hash(name)) % 99999, name=name,
                     position=pos, club=club, price=price,
                     expected_points=ep, **kw)


class AffordabilityTest(unittest.TestCase):
    """The bank is the constraint that makes a recommendation real."""

    def test_a_player_you_cannot_afford_is_never_suggested(self):
        squad = [c("Cheap", price=5.0, ep=3.0)]
        market = [c("Haaland", price=15.6, ep=9.0)]
        self.assertEqual(suggest(squad, market, bank=0.3), [])

    def test_the_bank_widens_what_is_reachable(self):
        squad = [c("Cheap", price=5.0, ep=3.0)]
        market = [c("Better", price=6.0, ep=5.0)]
        self.assertEqual(suggest(squad, market, bank=0.3), [],
                         "5.0 + 0.3 does not reach 6.0")
        moves = suggest(squad, market, bank=1.0)
        self.assertEqual(len(moves), 1, "5.0 + 1.0 does reach 6.0")

    def test_the_bank_after_a_move_is_reported(self):
        squad = [c("Out", price=6.0, ep=3.0)]
        market = [c("In", price=5.5, ep=5.0)]
        move = suggest(squad, market, bank=0.3)[0]
        self.assertAlmostEqual(move.cost, -0.5)
        self.assertAlmostEqual(move.bank_after, 0.8,
                               msg="selling dearer than you buy adds to the bank")


class FplRulesTest(unittest.TestCase):
    def test_positions_must_match(self):
        """A squad has fixed position slots. A midfielder cannot replace a
        defender, however much better he is."""
        squad = [c("Defender", pos="DEF", ep=2.0)]
        market = [c("Superstar", pos="MID", ep=9.0)]
        self.assertEqual(suggest(squad, market, bank=10.0), [])

    def test_the_three_per_club_limit_is_respected(self):
        squad = [c(f"ARS{i}", club="ARS", ep=3.0) for i in range(3)] + \
                [c("Other", club="BHA", ep=2.0)]
        market = [c("FourthGunner", club="ARS", ep=9.0)]
        moves = suggest(squad, market, bank=10.0)
        self.assertTrue(all(m.into.club != "ARS" or m.out.club == "ARS"
                            for m in moves),
                        "a fourth player from one club is not a legal squad")

    def test_swapping_within_the_same_club_stays_legal(self):
        """Three from a club is fine; replacing one of them with another from
        the same club keeps it at three, so it must not be blocked."""
        squad = [c(f"ARS{i}", club="ARS", ep=2.0) for i in range(3)]
        market = [c("BetterGunner", club="ARS", ep=6.0)]
        moves = suggest(squad, market, bank=10.0)
        self.assertEqual(len(moves), 1,
                         "three squad members could each be sold for him, but "
                         "he can only be bought once, so that is ONE idea")
        self.assertEqual(moves[0].into.name, "BetterGunner")

    def test_an_injured_target_is_never_suggested(self):
        """The model sees form, not this morning's team news."""
        squad = [c("Fine", ep=3.0)]
        market = [c("Injured", ep=9.0, status="i"),
                  c("Doubtful", ep=8.0, chance=25)]
        self.assertEqual(suggest(squad, market, bank=10.0), [])


class WorthItTest(unittest.TestCase):
    """The engine should decline to have an opinion it does not have."""

    def test_a_gain_inside_the_noise_floor_is_not_worth_it(self):
        squad = [c("Out", ep=4.00)]
        market = [c("In", ep=4.00 + NOISE_FLOOR / 2)]
        move = suggest(squad, market, bank=5.0)[0]
        self.assertFalse(move.worth_it,
                         "a gain smaller than the model's own reseeding "
                         "variance is not a preference")

    def test_a_real_gain_is_worth_it(self):
        squad = [c("Out", ep=3.0)]
        market = [c("In", ep=5.0)]
        self.assertTrue(suggest(squad, market, bank=5.0)[0].worth_it)

    def test_a_hit_is_subtracted_from_the_gain(self):
        """A move gaining three that costs four is a move that loses one."""
        squad = [c("A", ep=1.0), c("B", ep=1.0)]
        market = [c("X", ep=5.0), c("Y", ep=4.0)]
        moves = suggest(squad, market, bank=5.0, free_transfers=1)
        self.assertTrue(moves[0].free)
        self.assertFalse(moves[1].free)
        self.assertAlmostEqual(moves[1].net, moves[1].gain - HIT_COST, places=3)

    def test_the_free_transfer_goes_to_the_best_move(self):
        """Not to whichever was evaluated first."""
        squad = [c("A", ep=1.0), c("B", ep=1.0)]
        market = [c("Small", ep=2.0), c("Big", ep=8.0)]
        moves = suggest(squad, market, bank=5.0, free_transfers=1)
        self.assertEqual(moves[0].into.name, "Big")
        self.assertTrue(moves[0].free)

    def test_a_move_that_loses_points_is_never_proposed(self):
        squad = [c("Good", ep=6.0)]
        market = [c("Worse", ep=3.0)]
        self.assertEqual(suggest(squad, market, bank=5.0), [])


class PenaltyTest(unittest.TestCase):
    """penalties_order is collected and shown, not fed to the model."""

    def test_a_penalty_taker_is_called_out(self):
        squad = [c("Out", ep=3.0)]
        market = [c("Gross", ep=5.0, penalties_order=1)]
        move = suggest(squad, market, bank=5.0)[0]
        self.assertTrue(move.into.takes_penalties)
        self.assertTrue(any("penalty" in r.lower() for r in move.reasons))

    def test_it_is_not_mentioned_when_swapping_one_taker_for_another(self):
        """Nothing is gained, so nothing should be claimed."""
        squad = [c("Out", ep=3.0, penalties_order=1)]
        market = [c("In", ep=5.0, penalties_order=1)]
        move = suggest(squad, market, bank=5.0)[0]
        self.assertFalse(any("penalty" in r.lower() for r in move.reasons))

    def test_free_kicks_are_mentioned_when_penalties_are_not(self):
        squad = [c("Out", ep=3.0)]
        market = [c("In", ep=5.0, freekicks_order=1)]
        move = suggest(squad, market, bank=5.0)[0]
        self.assertTrue(any("free kick" in r.lower() for r in move.reasons))


class SummaryTest(unittest.TestCase):
    def test_no_worthwhile_move_says_holding_is_a_move(self):
        out = summarise([], bank=0.3, free_transfers=1)
        self.assertFalse(out["has_recommendation"])
        self.assertIn("Holding is a move", out["headline"])

    def test_a_worthwhile_move_becomes_the_headline(self):
        squad = [c("Mitchell", ep=3.0)]
        market = [c("Justin", ep=5.0)]
        moves = suggest(squad, market, bank=0.3)
        out = summarise(moves, 0.3, 1)
        self.assertTrue(out["has_recommendation"])
        self.assertIn("Mitchell", out["headline"])
        self.assertIn("Justin", out["headline"])

    def test_a_marginal_move_is_listed_but_not_recommended(self):
        """Shown so the arithmetic is visible; not offered as advice."""
        squad = [c("Out", ep=4.0)]
        market = [c("In", ep=4.0 + NOISE_FLOOR / 2)]
        out = summarise(suggest(squad, market, bank=5.0), 5.0, 1)
        self.assertFalse(out["has_recommendation"])
        self.assertEqual(len(out["moves"]), 1)

    def test_one_suggestion_per_player_leaving(self):
        """Ten variations on selling the same defender is a list nobody
        reads."""
        squad = [c("Out", ep=2.0)]
        market = [c(f"Option{i}", ep=3.0 + i * 0.1) for i in range(10)]
        self.assertEqual(len(suggest(squad, market, bank=5.0)), 1)


if __name__ == "__main__":
    unittest.main(verbosity=2, exit=False)
