"""
Who to buy, who to drop, and whether it is worth doing.

WHY THIS EXISTS
===============
The engine could already rank every player in the league by expected points,
and it could already solve for an optimal fifteen. What it could not do was
answer the only question actually asked on a Friday: *given the squad I have
and the money I have, what one change should I make?*

The gap showed up concretely. Groß at Brighton is a first-choice penalty
taker on the best form of any penalty taker in the league, at 5.7m. The model
rated him 3.93 expected points for GW5 and never mentioned him, because the
Recommendations tab ranks by raw expected points and Haaland's 7.01 sits at
the top of that list forever. A 15.6m striker you cannot afford is not a
recommendation.

WHAT THIS DOES DIFFERENTLY
==========================
  * It only proposes players you can actually afford: the replacement has to
    cost no more than the player leaving plus what is in the bank. The bank is
    FETCHED, not typed. It was a command-line flag defaulting to zero, which
    silently restricted every suggestion to an exact-price swap.

  * It prices the hit. A transfer beyond your free ones costs four points, so
    a move gaining three is a move that loses one. Reported as net, never
    gross.

  * It respects the noise floor. A gain under 0.17 expected points is inside
    the model's own reseeding variance, measured in FINDINGS_2026-09-11.md.
    Proposing such a move is the engine inventing a preference it does not
    have, which is the mistake the captain panel was already fixed for.

  * It says when a player takes penalties. This is not a model feature and is
    not pretending to be one: it is a fact about how the player scores that
    the model cannot see, shown next to the number so a human can weigh it.


ON PENALTIES, HONESTLY
======================
`penalties_order` is collected and surfaced here, but it is NOT fed to the
model as a feature, and the distinction matters.

Feeding it in means retraining, and retraining cannot happen mid-gameweek
because `recommend.py` uses INSERT OR REPLACE and would overwrite predictions
already logged. It also cannot happen honestly until the goalkeeper training
bug is fixed, since that silently drops 11% of the archive. Both are on the
roadmap in that order.

Until then the penalty flag is decoration with a purpose: it tells you
something true that the number beside it does not account for.
"""
from __future__ import annotations

from dataclasses import dataclass, field

# Measured in FINDINGS_2026-09-11.md by reseeding the model: predictions move
# by 0.169 expected points on average from the random seed alone. A suggested
# transfer gaining less than this is the engine reporting a preference it does
# not actually have.
NOISE_FLOOR = 0.17

# What FPL charges for a transfer beyond the free ones.
HIT_COST = 4.0

# FPL allows at most three players from any one club.
MAX_PER_CLUB = 3


@dataclass
class Candidate:
    element_id: int
    name: str
    position: str
    club: str
    price: float
    expected_points: float
    penalties_order: int | None = None
    freekicks_order: int | None = None
    status: str = "a"
    chance: int | None = None
    owned_by_percent: float = 0.0

    @property
    def takes_penalties(self) -> bool:
        return self.penalties_order == 1

    @property
    def available(self) -> bool:
        """Not injured, suspended or unavailable.

        The model sees form, not this morning's team news, so this gate is
        applied on top of it rather than trusted to emerge from the number.
        """
        if self.status not in ("a", "d"):
            return False
        return (self.chance is None) or (self.chance >= 75)


@dataclass
class Move:
    out: Candidate
    into: Candidate
    gain: float                 # gross expected-points gain
    net: float                  # after any hit
    cost: float                 # money it takes out of the bank
    bank_after: float
    free: bool                  # was this within the free transfers
    reasons: list[str] = field(default_factory=list)

    @property
    def worth_it(self) -> bool:
        return self.net > NOISE_FLOOR


def _club_counts(squad: list[Candidate]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for player in squad:
        counts[player.club] = counts.get(player.club, 0) + 1
    return counts


def suggest(squad: list[Candidate], market: list[Candidate], bank: float,
            free_transfers: int = 1, limit: int = 5) -> list[Move]:
    """The best single swaps available, best first.

    One transfer at a time on purpose. Two-transfer combinations explode the
    search and, more importantly, are not the decision being made: a manager
    with one free transfer wants to know the one move, and a manager with two
    can apply this twice and re-run.
    """
    owned_ids = {p.element_id for p in squad}
    counts = _club_counts(squad)
    moves: list[Move] = []

    for out in squad:
        budget = out.price + bank
        for into in market:
            if into.element_id in owned_ids:
                continue
            if into.position != out.position:
                continue          # FPL squads are fixed by position
            if into.price > budget + 1e-9:
                continue
            if not into.available:
                continue
            # The three-per-club rule, evaluated as it would be AFTER the move.
            after = counts.get(into.club, 0) + (0 if into.club == out.club else 1)
            if after > MAX_PER_CLUB:
                continue

            gain = into.expected_points - out.expected_points
            if gain <= 0:
                continue

            free = len(moves) < free_transfers
            net = gain if free_transfers > 0 else gain - HIT_COST
            cost = into.price - out.price

            reasons = []
            if into.takes_penalties and not out.takes_penalties:
                reasons.append(
                    f"{into.name} is his club's first-choice penalty taker. "
                    "The model cannot see that: it is a scoring route assigned "
                    "by the manager rather than earned in the form data.")
            if into.freekicks_order == 1 and not into.takes_penalties:
                reasons.append(f"{into.name} takes direct free kicks.")
            if out.status not in ("a",):
                reasons.append(f"{out.name} is flagged in the team news.")
            if into.price < out.price:
                reasons.append(
                    f"Frees up {out.price - into.price:.1f}m for later.")
            if into.owned_by_percent < 10 and gain > NOISE_FLOOR:
                reasons.append(
                    f"Owned by only {into.owned_by_percent:.0f}%, so this is a "
                    "differential rather than a template move.")

            moves.append(Move(
                out=out, into=into, gain=round(gain, 3),
                net=round(net, 3), cost=round(cost, 2),
                bank_after=round(bank - cost, 2), free=free, reasons=reasons))

    moves.sort(key=lambda m: -m.gain)

    # Recompute free/net now the ordering is known: the best move is the one
    # that uses the free transfer, not whichever happened to be built first.
    best: list[Move] = []
    seen_out: set[int] = set()
    seen_in: set[int] = set()
    for move in moves:
        # One suggestion per player leaving, and one per player arriving.
        # Without the second, three squad members who could each be sold for
        # the same target produce three entries that are really one idea, and
        # you can only buy him once.
        if move.out.element_id in seen_out or move.into.element_id in seen_in:
            continue
        seen_out.add(move.out.element_id)
        seen_in.add(move.into.element_id)
        move.free = len(best) < free_transfers
        move.net = round(move.gain if move.free else move.gain - HIT_COST, 3)
        if not move.free and move.net <= 0:
            move.reasons.append(
                f"Costs a {HIT_COST:.0f} point hit, which this gain does not "
                "cover. Shown so the arithmetic is visible, not as advice.")
        best.append(move)
        if len(best) >= limit:
            break
    return best


def summarise(moves: list[Move], bank: float, free_transfers: int) -> dict:
    """The shape the publisher and the emailer both read."""
    recommended = [m for m in moves if m.free and m.worth_it]
    return {
        "bank": round(bank, 2),
        "free_transfers": free_transfers,
        "noise_floor": NOISE_FLOOR,
        "has_recommendation": bool(recommended),
        "headline": (
            f"{recommended[0].out.name} to {recommended[0].into.name}, "
            f"+{recommended[0].net:.2f} expected points"
            if recommended else
            "No transfer clears the noise floor this week. Holding is a move."
        ),
        "moves": [
            {
                "out": m.out.name, "out_price": m.out.price,
                "out_ep": round(m.out.expected_points, 2),
                "in": m.into.name, "in_price": m.into.price,
                "in_ep": round(m.into.expected_points, 2),
                "in_club": m.into.club, "position": m.into.position,
                "gain": m.gain, "net": m.net, "cost": m.cost,
                "bank_after": m.bank_after, "free": m.free,
                "penalties": m.into.takes_penalties,
                "worth_it": m.worth_it, "reasons": m.reasons,
            }
            for m in moves
        ],
    }
