"""
The bench: who to leave out, in what order, and when to spend Bench Boost.

WHY THIS EXISTS
===============
GW4 left 17 points on the bench. That is a bigger single leak than any
captaincy call this season, and until now the engine said nothing about it.
It ranked players and picked a squad; what happened at the bottom of that
squad was left to the manager.

Three separate decisions live here, and they are genuinely different problems:

  1. WHICH four to bench       — an optimisation over the starting XI
  2. IN WHAT ORDER             — an autosub problem with a provable answer
  3. WHEN TO PLAY BENCH BOOST  — a chip-timing problem under uncertainty


THE ORDERING RESULT, AND WHY THE OBVIOUS ANSWER IS WRONG
========================================================
FPL autosubs work like this: if a starter plays zero minutes, the first
bench player who *did* play, and whose introduction keeps the formation
legal, comes on in their place.

So with a bench ordered [A, B, C] and exactly one starter blanking:

    E[gain] = pA·eA + (1-pA)·pB·eB + (1-pA)(1-pB)·pC·eC

where p is P(the player appears) and e is E[their points | they appear].

Compare swapping the first two:

    [A,B] - [B,A]  =  pA·eA + (1-pA)pB·eB - pB·eB - (1-pB)pA·eA
                   =  pA·pB·(eA - eB)

Both probabilities are positive, so **[A,B] beats [B,A] exactly when
eA > eB**. The probability of playing cancels out of the comparison
entirely.

    ORDER THE BENCH BY E[points | they play].
    DO NOT order it by expected points.

This is not a small distinction. Expected points is p·e, and sorting by it
is what any reasonable person would do — it is the number the engine shows
everywhere else and the only number the predictions table used to store.
Worked example, three real-shaped bench players:

    player   P(play)   E[pts|play]   xP = p·e
    A          0.95        2.0         1.90
    B          0.40        5.0         2.00
    C          0.70        3.0         2.10

    order by xP        -> C, B, A   = 3.04 expected points
    order by E[pts|play] -> B, C, A = 3.60 expected points   <- optimal

Sorting by the intuitive number gives the THIRD best of six orderings and
costs 0.56 expected points per blank.

The intuition for why: a bench player who is unlikely to play is not a
wasted slot, because the autosub simply falls through to the next player.
Low availability costs you nothing in the ordering. What it cannot do is
make a low-scoring player worth promoting.

THE CAVEAT: this assumes exactly one starter blanks. With two or more, the
bench is consumed top-down and the same ordering still applies, so the
result holds. It also assumes every bench player is formation-legal as a
substitute, which is handled separately in `legal_substitutions`.


WHAT THIS MODULE DOES NOT KNOW
==============================
  * It cannot see a manager's press conference. A player the model rates
    highly and the manager has quietly dropped will be ordered too high.
  * Double gameweeks are detected from the fixture list, which is only as
    current as the last snapshot. Postponements land late.
  * Bench Boost advice beyond the next gameweek rests on predictions the
    model is not really built to make that far out. Said plainly where it
    is reported rather than hidden behind a number.
"""
from __future__ import annotations

from dataclasses import dataclass, field

# FPL squad rules. A legal starting XI is exactly one goalkeeper, at least
# three defenders, at least two midfielders and at least one forward.
MIN_BY_POSITION = {"GKP": 1, "DEF": 3, "MID": 2, "FWD": 1}
MAX_BY_POSITION = {"GKP": 1, "DEF": 5, "MID": 5, "FWD": 3}
SQUAD_SIZE = 15
XI_SIZE = 11


@dataclass
class Player:
    """One squad member, with the two halves of the prediction kept apart."""

    element_id: int
    name: str
    position: str
    p_play: float          # P(appears), stage one of the two-stage model
    points_if_played: float  # E[points | appears], stage two
    price: float = 0.0
    flagged: bool = False    # team news says doubtful or worse

    @property
    def expected_points(self) -> float:
        """The number shown everywhere else in the engine.

        Correct for choosing WHO to start. Wrong for ordering the bench, which
        is the entire point of this module.
        """
        return self.p_play * self.points_if_played


@dataclass
class BenchPlan:
    starting_xi: list[Player]
    bench: list[Player] = field(default_factory=list)  # ordered: GK, then 1,2,3
    formation: str = ""
    autosub_value: float = 0.0
    bench_boost_value: float = 0.0
    notes: list[str] = field(default_factory=list)


def formation_of(players: list[Player]) -> str:
    """"3-5-2" and so on, counting outfielders only, as FPL writes it."""
    counts = {p: 0 for p in MIN_BY_POSITION}
    for player in players:
        counts[player.position] = counts.get(player.position, 0) + 1
    return f"{counts['DEF']}-{counts['MID']}-{counts['FWD']}"


def is_legal_xi(players: list[Player]) -> bool:
    if len(players) != XI_SIZE:
        return False
    counts = {pos: 0 for pos in MIN_BY_POSITION}
    for player in players:
        if player.position not in counts:
            return False
        counts[player.position] += 1
    return all(MIN_BY_POSITION[pos] <= counts[pos] <= MAX_BY_POSITION[pos]
               for pos in counts)


def choose_starting_xi(squad: list[Player]) -> tuple[list[Player], list[Player]]:
    """Pick the XI maximising expected points, subject to a legal formation.

    Expected points IS the right criterion here, unlike for ordering: a
    starter contributes p·e whether or not they play, because a blank is a
    zero rather than a fall-through.

    Exhaustive over formations rather than greedy. There are only a handful
    of legal shapes and a greedy pick by expected points can produce an
    illegal one, most commonly by starting six midfielders.
    """
    by_position: dict[str, list[Player]] = {pos: [] for pos in MIN_BY_POSITION}
    for player in squad:
        if player.position in by_position:
            by_position[player.position].append(player)
    for pos in by_position:
        by_position[pos].sort(key=lambda p: -p.expected_points)

    best_xi: list[Player] | None = None
    best_value = float("-inf")

    for n_def in range(MIN_BY_POSITION["DEF"], MAX_BY_POSITION["DEF"] + 1):
        for n_mid in range(MIN_BY_POSITION["MID"], MAX_BY_POSITION["MID"] + 1):
            for n_fwd in range(MIN_BY_POSITION["FWD"], MAX_BY_POSITION["FWD"] + 1):
                if 1 + n_def + n_mid + n_fwd != XI_SIZE:
                    continue
                need = {"GKP": 1, "DEF": n_def, "MID": n_mid, "FWD": n_fwd}
                if any(len(by_position[pos]) < n for pos, n in need.items()):
                    continue
                xi = [p for pos, n in need.items() for p in by_position[pos][:n]]
                value = sum(p.expected_points for p in xi)
                if value > best_value:
                    best_value, best_xi = value, xi

    if best_xi is None:
        raise ValueError("no legal starting XI from this squad")

    chosen = {id(p) for p in best_xi}
    bench = [p for p in squad if id(p) not in chosen]
    return best_xi, bench


def order_bench(bench: list[Player]) -> list[Player]:
    """The goalkeeper first, then the outfielders by E[points | they play].

    The goalkeeper's position on the bench is not a choice: a benched keeper
    can only ever replace the starting keeper, so FPL fixes them in the first
    slot and the three ordered slots are the outfielders.

    Sorting the outfielders by points_if_played rather than by expected_points
    is the whole result. See the module docstring for the derivation and for
    what it costs to get wrong.
    """
    keepers = [p for p in bench if p.position == "GKP"]
    outfield = [p for p in bench if p.position != "GKP"]
    outfield.sort(key=lambda p: (-p.points_if_played, -p.expected_points, p.name))
    return keepers + outfield


def can_substitute(xi: list[Player], going_off: Player, coming_on: Player) -> bool:
    """Is the XI still legal if `coming_on` replaces `going_off`?

    THIS IS PER BLANKING PLAYER, and the first version of this function got it
    wrong by asking the looser question "could this sub come on for ANYBODY".

    The looser question is almost always yes, because a like-for-like swap
    leaves the formation untouched, which made the check useless. FPL's actual
    rule is narrower: the substitute comes on for the specific player who
    failed to appear, and the resulting formation has to be legal. So a
    midfielder cannot cover a blanking defender when you are playing three at
    the back, because that leaves two.

    Caught by a test asserting a fourth forward was stranded behind a 3-4-3.
    It is not: it replaces one of the three forwards. The test premise was
    wrong AND so was the function.
    """
    remaining = [x for x in xi if x is not going_off] + [coming_on]
    return is_legal_xi(remaining)


def can_cover_anyone(xi: list[Player], sub: Player) -> bool:
    """Is there any starter this substitute could legally replace?

    In practice almost always true, since a like-for-like swap is always
    legal and the XI holds at least one of every outfield position. Kept
    because "almost always" is not "always", and a squad with an unusual
    shape should not silently count a dead slot as cover.
    """
    return any(can_substitute(xi, starter, sub) for starter in xi)


def autosub_value(xi: list[Player], ordered_bench: list[Player]) -> float:
    """Expected points the bench contributes through autosubs.

    Modelled as: each starter independently fails to appear with probability
    (1 - p_play), and each blank is covered by the next bench player who did
    appear. The expected number of blanks is small, so this sums the
    first-order term rather than enumerating every combination, and it is
    reported as a guide to how much the bench is worth rather than as a
    prediction.
    """
    outfield_xi = [x for x in xi if x.position != "GKP"]
    outfield_bench = [b for b in ordered_bench if b.position != "GKP"]
    if not outfield_xi or not outfield_bench:
        return 0.0

    # Worked per starter, because whether a substitute is eligible depends on
    # WHO blanked. A midfielder cannot cover a blanking defender in a back
    # three. Summing over starters weighted by their chance of blanking gives
    # the expected value of the bench as cover.
    total = 0.0
    for starter in outfield_xi:
        p_blank = 1 - starter.p_play
        if p_blank <= 0:
            continue
        gain, still_needed = 0.0, 1.0
        for sub in outfield_bench:
            if not can_substitute(xi, starter, sub):
                continue
            gain += still_needed * sub.p_play * sub.points_if_played
            still_needed *= 1 - sub.p_play
        total += p_blank * gain

    # A bench of three cannot cover more than three blanks, however many
    # starters are doubtful.
    cap = sum(b.p_play * b.points_if_played for b in outfield_bench)
    return min(total, cap)


def bench_boost_value(bench: list[Player]) -> float:
    """What Bench Boost would be worth this gameweek.

    Straightforward: with the chip active every bench player scores, so the
    value is the sum of their expected points. Ordering is irrelevant that
    week, which is worth saying out loud because it is the one week the
    ordering work above does not matter.
    """
    return sum(p.expected_points for p in bench)


def build_plan(squad: list[Player]) -> BenchPlan:
    """The whole bench decision for one gameweek."""
    if len(squad) != SQUAD_SIZE:
        raise ValueError(f"a squad is {SQUAD_SIZE} players, got {len(squad)}")

    xi, bench = choose_starting_xi(squad)
    ordered = order_bench(bench)
    plan = BenchPlan(
        starting_xi=sorted(xi, key=lambda p: -p.expected_points),
        bench=ordered,
        formation=formation_of(xi),
        autosub_value=autosub_value(xi, ordered),
        bench_boost_value=bench_boost_value(ordered),
    )

    # Things worth saying that a number cannot.
    stranded = [p.name for p in ordered
                if p.position != "GKP" and not can_cover_anyone(xi, p)]
    if stranded:
        plan.notes.append(
            f"{', '.join(stranded)} cannot come on in a {plan.formation}: "
            "no substitution keeps the formation legal, so that bench slot is "
            "dead this week whatever the order.")

    flagged = [p.name for p in xi if p.flagged]
    if flagged:
        plan.notes.append(
            f"Starting with a flag on {', '.join(flagged)}. The bench covers a "
            "blank, but only for the first one that happens.")

    if any(p.p_play < 0.5 for p in ordered if p.position != "GKP"):
        plan.notes.append(
            "At least one bench player is more likely than not to sit out. "
            "That costs nothing in the ordering, since the autosub falls "
            "through to the next player, but it does thin the cover.")

    return plan


# ---------------------------------------------------------------------------
# Bench Boost timing
# ---------------------------------------------------------------------------
#
# A different problem from the two above, and a harder one. Ordering the bench
# is a decision you remake every week and can get wrong cheaply. Bench Boost is
# ONE decision for the whole season: spend it in the wrong week and the chip is
# gone.
#
# The honest position is that this engine cannot predict gameweek 28 in
# September. The model leans on recent form -- minutes_last alone carries four
# times the importance of the next feature -- so a prediction ten weeks out is
# barely better than a guess, and dressing one up in two decimal places would
# be exactly the overclaiming the noise floor work was about.
#
# So this does not pretend to name the week. It reports three things that are
# actually knowable and lets the manager decide:
#
#   1. What the chip is worth RIGHT NOW, which is a real number.
#   2. What it has been worth in recent weeks, so "now" has a scale.
#   3. Which upcoming gameweeks are structurally better -- double gameweeks,
#      where players play twice -- because that comes from the fixture list
#      rather than from the model.

# A full bench of players who all start is worth roughly this much. Below it,
# the chip is being wasted; comfortably above it, it is a good week. Derived
# from four bench players at a typical starting-player return rather than
# fitted, and labelled as a rule of thumb wherever it is shown.
BENCH_BOOST_FLOOR = 12.0


@dataclass
class ChipAdvice:
    gameweek: int
    value_now: float
    recent: list[tuple[int, float]]          # (gameweek, value) history
    double_gameweeks: list[int]              # upcoming, from the fixture list
    verdict: str
    reasoning: list[str] = field(default_factory=list)


def find_double_gameweeks(conn, from_gw: int) -> list[int]:
    """Gameweeks in which some team plays twice.

    Read from the LATEST fixture snapshot only. The fixtures table keeps one
    row per fixture per snapshot, so counting across all of them reports every
    team as playing forty-three times a week.

    Doubles are the single biggest structural lift for this chip and they are
    knowable in advance, unlike form. They appear as the season goes on and
    cup rounds force postponements, so an empty list in September is the
    normal state rather than a failure.
    """
    snapshot = conn.execute("SELECT MAX(snapshot_id) FROM fixtures").fetchone()[0]
    if snapshot is None:
        return []
    rows = conn.execute(
        """
        SELECT event FROM (
            SELECT event, team_h AS team FROM fixtures
             WHERE snapshot_id = ? AND event IS NOT NULL AND event >= ?
            UNION ALL
            SELECT event, team_a AS team FROM fixtures
             WHERE snapshot_id = ? AND event IS NOT NULL AND event >= ?
        )
        GROUP BY event, team HAVING COUNT(*) > 1
        """,
        (snapshot, from_gw, snapshot, from_gw),
    ).fetchall()
    return sorted({r[0] for r in rows})


def bench_boost_advice(gameweek, value_now, recent=None, doubles=None) -> ChipAdvice:
    """Play it, hold it, or wait for a double.

    Deliberately a recommendation with its reasoning attached rather than a
    bare number, because this is a one-shot decision and the manager is the
    one living with it.
    """
    recent = recent or []
    doubles = doubles or []
    reasoning: list[str] = []

    if recent:
        average = sum(v for _, v in recent) / len(recent)
        best = max(v for _, v in recent)
        reasoning.append(
            f"Your bench has been worth {average:.1f} points a week on average "
            f"over the last {len(recent)} gameweeks, best {best:.1f}. "
            f"This week it is worth {value_now:.1f}.")
    else:
        average = None
        reasoning.append(
            f"This week the bench is worth {value_now:.1f} expected points. "
            "There is no history yet to compare it against.")

    upcoming_doubles = [gw for gw in doubles if gw > gameweek]
    if upcoming_doubles:
        reasoning.append(
            f"Double gameweek{'s' if len(upcoming_doubles) > 1 else ''} "
            f"scheduled: GW{', GW'.join(str(g) for g in upcoming_doubles[:3])}. "
            "Bench players play twice, which is normally worth far more than "
            "any difference in form, and it is the one thing here that can be "
            "known in advance.")

    if value_now < BENCH_BOOST_FLOOR:
        verdict = "hold"
        reasoning.append(
            f"Below the {BENCH_BOOST_FLOOR:.0f} point rule of thumb for a full "
            "bench of starters, so this is not the week.")
    elif upcoming_doubles:
        verdict = "hold for the double"
        reasoning.append(
            "Worth playing on form alone, but a double gameweek is coming and "
            "is usually the better week. Holding costs you this week's value; "
            "missing the double costs more.")
    elif average is not None and value_now > max(average * 1.25, BENCH_BOOST_FLOOR):
        verdict = "play it"
        reasoning.append(
            "Comfortably your best bench of the season so far, with no double "
            "scheduled to wait for.")
    else:
        verdict = "playable"
        reasoning.append(
            "Above the floor but not clearly your best week. No double is "
            "scheduled, so there is nothing specific to wait for either.")

    reasoning.append(
        "Bench order does not matter in a Bench Boost week: every player "
        "scores, so the ordering work is the one thing the chip switches off.")

    return ChipAdvice(
        gameweek=gameweek, value_now=value_now, recent=recent,
        double_gameweeks=upcoming_doubles, verdict=verdict, reasoning=reasoning)
