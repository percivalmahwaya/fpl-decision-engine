"""
Opponent-strength features — closing the fixture-blindness gap.

WHY THIS EXISTS: the model had 47 features and not one of them described who a
player was facing. It predicted the same score for Bruno Fernandes against Man
City as against Hull City. Percival spotted this from football knowledge before
any metric did.

WHAT IT ADDS: rather than FPL's own Fixture Difficulty Rating — which is a
subjective 1-5 label — this derives difficulty from the data itself:

    how many FPL points does this opponent actually concede,
    to players of this position, in recent matches?

That is a measured quantity, position-aware (a team leaky at the back concedes
to defenders differently than to forwards), and it updates as form changes.

NO TEAM-ID MAPPING NEEDED. Both clubs' players share a `fixture` id, so the
opponent's name is simply the other `team` value in that fixture. This is the
same trick used to pull Bruno's record against City.

LEAKAGE: the opponent's record is shifted by one gameweek before use, exactly
like every other feature — a row may only see matches that finished before it.

=============================================================================
RESULT: TESTED AND REJECTED. Kept as a documented negative result.
=============================================================================
The underlying effect is real and large. Across four seasons, players with 60+
minutes average 2.73 points against the toughest fifth of opponents and 4.33
against the leakiest - a 1.59-point swing, 45% of the average return.

But adding it to the model did not help:
  attempt 1, rolling 3/6-match windows  ranking 4.39 -> 4.28 (worse), noise
  attempt 2, season-to-date expanding   ranking 4.39 -> 4.27 (worse)
                                        high-return MAE 5.305 -> 5.169 (better)
  paired bootstrap over 38 gameweeks    -0.125, 95% CI [-0.287, +0.036]
                                        P(better) = 0.06

Most likely because player form already encodes fixture quality indirectly, and
the top-20 is dominated by premium players whose ability outweighs the fixture.

CONCLUSION: apply fixture difficulty as a HUMAN OVERLAY on the model's output -
especially for captaincy, where one extreme fixture cannot be averaged away -
rather than as a feature inside the model.

=============================================================================
REOPENED 2026-09-19. The rejection may have been on the wrong metric.
=============================================================================
That conclusion was reached before a single gameweek had been scored live.
GW4 was graded on 2026-09-15 and said the champion's high-return MAE is the
WORST of three models, 5.353 against form_fdr's 4.572, while its overall MAE
is the best. The model's real weakness is the tail, and the tail is what
these features improved.

Re-run as a 2x2 with the goalkeeper fix, on identical test rows:

    + opponent    high-return MAE  -0.168  CI [-0.182, -0.155]  P(better) 1.00
    + both        high-return MAE  -0.214  CI [-0.228, -0.200]  P(better) 1.00
    + both        ranking          -0.081  CI [-0.226, +0.062]  P(better) 0.14

The ranking cost is real but not statistically clear; the high-return gain is
unambiguous. See FINDINGS_2026-09-19.md. NOT promoted: registered as a
challenger to be scored live, because two backtest metrics disagreeing is not
enough to overturn the one the project was built on.
"""

import numpy as np
import pandas as pd



def attach_opponent(df):
    """Add an `opponent` column by pairing the two teams sharing a fixture."""
    pairs = (df[["season", "fixture", "team"]]
             .drop_duplicates()
             .dropna(subset=["team"]))

    # Exactly two team names per fixture; map each to the other.
    grouped = pairs.groupby(["season", "fixture"])["team"].apply(list)
    lookup = {}
    for (season, fixture), teams in grouped.items():
        if len(teams) == 2:
            lookup[(season, fixture, teams[0])] = teams[1]
            lookup[(season, fixture, teams[1])] = teams[0]

    df = df.copy()
    df["opponent"] = [
        lookup.get((s, f, t))
        for s, f, t in zip(df["season"], df["fixture"], df["team"])
    ]
    return df


def build_opponent_features(df):
    """
    For each row, how generous has the opponent been lately — overall and to
    this player's position?

    Returns the frame plus the list of new feature names.
    """
    df = attach_opponent(df)

    # Points conceded by a team in a gameweek = points scored by everyone
    # facing them. Grouping on `opponent` gives exactly that.
    conceded = (df.groupby(["season", "opponent", "gw"], dropna=True)["total_points"]
                  .sum().reset_index()
                  .rename(columns={"opponent": "team_faced", "total_points": "pts_conceded"}))

    by_pos = (df.groupby(["season", "opponent", "gw", "position"], dropna=True)["total_points"]
                .sum().reset_index()
                .rename(columns={"opponent": "team_faced", "total_points": "pts_conceded_pos"}))

    new_feats = []

    # --- overall generosity, shifted so the current match is never included ---
    #
    # SEASON-TO-DATE (expanding), not a short rolling window. A first attempt
    # used 3- and 6-match windows and made the model WORSE: team-level points
    # conceded over a handful of matches is extremely noisy, and the noise
    # swamped the signal. Measured, the underlying effect is large — players
    # facing the leakiest fifth of teams average 4.33 points versus 2.73 against
    # the toughest, a 1.59-point swing — so the estimator, not the idea, was the
    # problem. An expanding mean uses every match so far this season and is far
    # more stable.
    conceded = conceded.sort_values(["season", "team_faced", "gw"])
    prior = conceded.groupby(["season", "team_faced"], sort=False)["pts_conceded"].shift(1)
    conceded["opp_conceded_std"] = (
        prior.groupby([conceded["season"], conceded["team_faced"]], sort=False)
             .expanding().mean().reset_index(level=[0, 1], drop=True))
    new_feats.append("opp_conceded_std")

    # --- generosity to this specific position ---
    by_pos = by_pos.sort_values(["season", "team_faced", "position", "gw"])
    prior_p = by_pos.groupby(["season", "team_faced", "position"],
                             sort=False)["pts_conceded_pos"].shift(1)
    by_pos["opp_conceded_pos_std"] = (
        prior_p.groupby([by_pos["season"], by_pos["team_faced"], by_pos["position"]], sort=False)
               .expanding().mean().reset_index(level=[0, 1, 2], drop=True))
    new_feats.append("opp_conceded_pos_std")

    # --- join back onto the player rows ---
    df = df.merge(
        conceded[["season", "team_faced", "gw", "opp_conceded_std"]],
        left_on=["season", "opponent", "gw"], right_on=["season", "team_faced", "gw"],
        how="left").drop(columns=["team_faced"])

    df = df.merge(
        by_pos[["season", "team_faced", "gw", "position", "opp_conceded_pos_std"]],
        left_on=["season", "opponent", "gw", "position"],
        right_on=["season", "team_faced", "gw", "position"],
        how="left").drop(columns=["team_faced"])

    return df, new_feats
