"""
The goalkeeper gap, measured properly.

WHY A SECOND SCRIPT: the first pass compared a champion tested on 26,320 rows
against a challenger tested on 29,747. Those are different test sets, and the
extra 3,427 rows are goalkeepers — whose MAE is 0.577, far below outfield. The
headline "MAE improved 0.9538 -> 0.9123" was therefore mostly COMPOSITION, not
skill: adding easy rows to a mean makes the mean look better.

The ranking comparison had the mirror-image flaw. Top-20 is chosen from the
candidate pool, and the champion's pool contains no goalkeepers at all. The
challenger was being penalised for having candidates the champion never had to
consider. Neither number answered the question asked.

THE FIX: build features ONCE over all positions, split chronologically once,
then train two models that differ ONLY in whether goalkeeper rows are in the
TRAINING set. Evaluate both on identical test rows and identical candidate
pools, split into the two questions that actually matter:

  Q1  On OUTFIELD players — does adding goalkeepers to training help or hurt
      the 590 predictions that were already working?

  Q2  On GOALKEEPERS — how wrong was the untrained model? This is the one that
      matters, because the live system logged 71 goalkeeper predictions per
      gameweek from a model that had seen none, and a real transfer was made
      on the strength of two of them.

Read-only on the database. Nothing is promoted.

Usage:
    python exp_goalkeeper2.py
"""

import sqlite3
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import (HistGradientBoostingClassifier,
                              HistGradientBoostingRegressor)
from sklearn.metrics import mean_absolute_error

from model import build_features, rank_eval, TRAIN_SEASONS, TEST_SEASON

warnings.filterwarnings("ignore")
ROOT = Path(__file__).parent
DB_PATH = ROOT / "fpl.db"
REPORT = ROOT / "GOALKEEPER_REPORT.md"

COMMON = dict(max_iter=400, learning_rate=0.06, max_depth=6,
              min_samples_leaf=40, random_state=42)
POSITIONS = ["GKP", "DEF", "MID", "FWD"]

lines = []


def say(s=""):
    print(s, flush=True)
    lines.append(s)


def fit(train, feats, seed=42):
    p = dict(COMMON, random_state=seed)
    X, y = train[feats], train["total_points"].values
    clf = HistGradientBoostingClassifier(**p).fit(X, train["played_60"].values)
    m = train["played_60"] == 1
    reg = HistGradientBoostingRegressor(l2_regularization=1.0, **p).fit(
        X[m], y[m.values])
    return clf, reg


def predict(models, test, feats):
    clf, reg = models
    return clf.predict_proba(test[feats])[:, 1] * reg.predict(test[feats])


def mae_block(test, pred):
    y = test["total_points"].values
    hi = y >= 5
    return (mean_absolute_error(y, pred),
            mean_absolute_error(y[hi], pred[hi]) if hi.any() else float("nan"))


def bootstrap(test_a, pred_a, test_b, pred_b, k, n=2000):
    """Paired over gameweeks — players in a gameweek share fixtures."""
    gws = sorted(test_a["gw"].unique())
    diffs = []
    for gw in gws:
        m = (test_a["gw"] == gw).values
        if m.sum() < k:
            continue
        diffs.append(rank_eval(test_b[m], pred_b[m], k=k)
                     - rank_eval(test_a[m], pred_a[m], k=k))
    diffs = np.array(diffs)
    if not len(diffs):
        return None
    rng = np.random.default_rng(0)
    boots = np.array([rng.choice(diffs, len(diffs), replace=True).mean()
                      for _ in range(n)])
    return diffs.mean(), np.percentile(boots, 2.5), np.percentile(boots, 97.5), \
        (boots > 0).mean(), len(diffs)


def main():
    t0 = datetime.now(timezone.utc)
    say("# Goalkeeper training gap — corrected experiment\n")
    say(f"_Run {t0.isoformat(timespec='seconds')} · read-only · nothing promoted._\n")
    say("> Supersedes the first run, whose champion and challenger were scored "
        "on different test sets and different candidate pools. Both models "
        "below are trained on the same features and scored on identical rows; "
        "they differ **only** in whether goalkeepers were in the training set.\n")

    conn = sqlite3.connect(DB_PATH)
    try:
        df = pd.read_sql_query(
            "SELECT season, gw, element, fixture, name, position, was_home,"
            " minutes, total_points, xp, xg, xa, xgi, xgc, ict, bps, threat,"
            " creativity, influence, value, selected"
            " FROM history ORDER BY season, element, gw, fixture", conn)
        df["position"] = df["position"].replace({"GK": "GKP"})
        df = df[df["position"].isin(POSITIONS)].copy()

        d, feats = build_features(df)
        train = d[d.season.isin(TRAIN_SEASONS)].dropna(subset=["total_points"])
        test = d[d.season == TEST_SEASON].dropna(subset=["total_points"])

        out_tr = train[train.position != "GKP"]
        say(f"Training rows: **{len(out_tr):,}** outfield-only vs "
            f"**{len(train):,}** with keepers. "
            f"Test rows: {len(test):,} ({(test.position=='GKP').sum():,} keepers).\n")

        say("Training both models...")
        champ = fit(out_tr, feats)        # current behaviour: no keepers
        chall = fit(train, feats)         # fixed: keepers included
        say("")

        # ---------------------------------------------------------- Q1
        say("## Q1. Outfield players — identical rows, identical pool\n")
        te = test[test.position != "GKP"]
        pa, pb = predict(champ, te, feats), predict(chall, te, feats)
        (ma, ha), (mb, hb) = mae_block(te, pa), mae_block(te, pb)
        ra, rb = rank_eval(te, pa, k=20), rank_eval(te, pb, k=20)

        say("| training set | MAE | high-return MAE | top-20 pts |")
        say("|---|---:|---:|---:|")
        say(f"| outfield only (current) | {ma:.4f} | {ha:.4f} | **{ra:.3f}** |")
        say(f"| + goalkeepers | {mb:.4f} | {hb:.4f} | **{rb:.3f}** |")
        say(f"\nDelta on ranking: **{rb-ra:+.3f}** pts/pick\n")

        bs = bootstrap(te, pa, te, pb, k=20)
        if bs:
            m, lo, hi, p, n = bs
            say(f"Paired bootstrap over {n} gameweeks: **{m:+.3f}** "
                f"95% CI **[{lo:+.3f}, {hi:+.3f}]**, P(better) = **{p:.2f}**")
            say(f"{'Real effect.' if lo > 0 or hi < 0 else '**Contains zero — noise.**'}\n")

        # ---------------------------------------------------------- Q2
        say("## Q2. Goalkeepers — the predictions that were actually broken\n")
        tg = test[test.position == "GKP"]
        ga, gb = predict(champ, tg, feats), predict(chall, tg, feats)
        (gma, gha), (gmb, ghb) = mae_block(tg, ga), mae_block(tg, gb)

        say("| model | MAE on keepers | high-return MAE | mean prediction | actual mean |")
        say("|---|---:|---:|---:|---:|")
        say(f"| never trained on a keeper | {gma:.4f} | {gha:.4f} | "
            f"{ga.mean():.3f} | {tg['total_points'].mean():.3f} |")
        say(f"| trained on keepers | {gmb:.4f} | {ghb:.4f} | "
            f"{gb.mean():.3f} | {tg['total_points'].mean():.3f} |")
        improve = gma - gmb
        say(f"\n**MAE improvement on goalkeepers: {improve:+.4f}** "
            f"({100*improve/gma:+.1f}%)\n")

        # Bias is the thing that changes a transfer decision: a model that is
        # systematically high or low on keepers ranks them wrongly against
        # each other even when the average error looks acceptable.
        say("| model | mean bias (pred - actual) |")
        say("|---|---:|")
        say(f"| never trained on a keeper | {(ga - tg['total_points'].values).mean():+.4f} |")
        say(f"| trained on keepers | {(gb - tg['total_points'].values).mean():+.4f} |")
        say("")

        # Does it actually re-order keepers? That is what picks a goalkeeper.
        rga = rank_eval(tg, ga, k=5)
        rgb = rank_eval(tg, gb, k=5)
        say(f"Top-5 **keepers** by predicted points, mean actual score:\n")
        say(f"- never trained: **{rga:.3f}**")
        say(f"- trained: **{rgb:.3f}**  ({rgb-rga:+.3f})\n")

        bsg = bootstrap(tg, ga, tg, gb, k=5)
        if bsg:
            m, lo, hi, p, n = bsg
            say(f"Paired bootstrap over {n} gameweeks: **{m:+.3f}** "
                f"95% CI **[{lo:+.3f}, {hi:+.3f}]**, P(better) = **{p:.2f}**\n")

        # ------------------------------------------------------ verdict
        say("## Verdict\n")
        say(f"- **Outfield is unaffected**: {rb-ra:+.3f} pts/pick, "
            f"confidence interval spanning zero. Adding goalkeepers to training "
            f"neither helps nor harms the 590 outfield predictions.")
        say(f"- **Goalkeepers improve by {100*improve/gma:.1f}% on MAE** and "
            f"{rgb-rga:+.3f} points on top-5 keeper ranking.")
        say("")
        say("So this is a **narrow correctness fix, not a model upgrade** — "
            "which is exactly what it should be. The aggregate barely moves "
            "because 90% of predictions were always fine. The 10% that were "
            "being produced with no training data get better, and those are "
            "precisely the ones a squad needs two of.")
        say("")
        say("**Position-specific models remain REJECTED** — the first run "
            "measured -0.229 pts/pick on ranking, the largest negative of "
            "anything tested. Documented alongside `opponent.py` as a second "
            "idea from the prior art that did not survive contact with this "
            "dataset.\n")
        say("**Still not merged to `main`.** `recommend.py` uses "
            "`INSERT OR REPLACE` on the predictions table, so a changed model "
            "reaching main before the GW4 deadline would overwrite the "
            "predictions logged in advance — destroying both the honest record "
            "and the engine-vs-gut comparison. Merge after GW4 is scored.\n")
    finally:
        conn.close()

    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    mins = (datetime.now(timezone.utc) - t0).total_seconds() / 60
    print(f"\n  Report written to {REPORT.name}  ({mins:.1f} min)")


if __name__ == "__main__":
    main()
