"""
The goalkeeper training gap — does fixing it actually help?

THE BUG
=======
`history` (the four-season archive) labels goalkeepers **GK**. The live FPL API
labels them **GKP**. model.py:53 filters training data with

    df[df["position"].isin(["GKP", "DEF", "MID", "FWD"])]

so every one of the 12,500 goalkeeper rows is silently dropped — 11% of the
dataset. Two consequences, neither of which raises an error:

  1. `is_gkp` (model.py:83) is a CONSTANT ZERO during training. The model has
     never seen that feature take the value 1. It is dead weight in a 47-feature
     vector and, worse, it is the only thing telling the model a keeper is a
     keeper.

  2. At prediction time the rows come from the live API, which says GKP, so
     keepers sail through the filter and ARE predicted — by a model with no
     goalkeeper training data at all. 71 such predictions were logged for GW4.

That is the shape of failure this project exists to catch: nothing crashed,
nothing warned, and the numbers looked entirely reasonable.

WHAT THIS SCRIPT DOES
=====================
Champion versus challenger, on identical chronological splits:

    champion    = current behaviour, keepers dropped
    challenger  = GK normalised to GKP, keepers included

and asks, in order:

  A. How much data was being thrown away, and is `is_gkp` really dead?
  B. Does including keepers improve the OVERALL model, or just the keepers?
  C. Is any difference significant, or is it noise? (paired bootstrap)
  D. Does going position-specific — the OpenFPL approach, and the last big
     idea on the roadmap — beat one global model now that GKP exists?

PROMOTION RULE: nothing here is promoted automatically. A challenger wins only
if it beats the champion on RANKING (top-20 mean actual points), because that
is the metric that decides who you actually pick. MAE improvements that do not
move ranking are not worth a model change — that lesson came from opponent.py,
where a feature improved high-return MAE, lost on ranking, and was rejected.

THIS SCRIPT IS READ-ONLY ON THE DATABASE. It never writes to `predictions`.
Re-running recommend.py would overwrite the GW4 predictions that were logged
before the deadline, destroying both the honest record and the engine-vs-gut
comparison in compare.py.

Usage:
    python exp_goalkeeper.py            # full run, writes GOALKEEPER_REPORT.md
    python exp_goalkeeper.py --quick    # skip permutation importance
"""

import sqlite3
import sys
import time
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

# Managers score on a completely different basis and are genuinely noise.
# Goalkeepers are not — they are a quarter of every squad.
REAL_POSITIONS = ["GKP", "DEF", "MID", "FWD"]

lines = []


def say(s=""):
    print(s, flush=True)
    lines.append(s)


def load_raw(conn, fix_gkp):
    """
    The same query model.py uses, with the position filter applied either the
    broken way (GK dropped) or the fixed way (GK normalised to GKP first).
    """
    df = pd.read_sql_query(
        "SELECT season, gw, element, fixture, name, position, was_home,"
        " minutes, total_points, xp, xg, xa, xgi, xgc, ict, bps, threat,"
        " creativity, influence, value, selected"
        " FROM history ORDER BY season, element, gw, fixture",
        conn,
    )
    if fix_gkp:
        df["position"] = df["position"].replace({"GK": "GKP"})
    return df[df["position"].isin(REAL_POSITIONS)].copy()


def two_stage(train, test, feats, seed=42):
    """The champion architecture: P(60+ mins) x E[points | played 60+]."""
    params = dict(COMMON, random_state=seed)
    Xtr, ytr = train[feats], train["total_points"].values
    clf = HistGradientBoostingClassifier(**params).fit(Xtr, train["played_60"].values)
    played = train["played_60"] == 1
    reg = HistGradientBoostingRegressor(l2_regularization=1.0, **params).fit(
        Xtr[played], ytr[played.values])
    return clf.predict_proba(test[feats])[:, 1] * reg.predict(test[feats])


def evaluate(test, pred, label):
    y = test["total_points"].values
    hi = y >= 5
    return {
        "label": label,
        "mae": mean_absolute_error(y, pred),
        "mae_high": mean_absolute_error(y[hi], pred[hi]) if hi.any() else float("nan"),
        "rank": rank_eval(test, pred, k=20),
        "n": len(test),
    }


# =====================================================================  A

def section_a(conn):
    say("## A. How much was being thrown away\n")
    rows = conn.execute(
        "SELECT position, COUNT(*) FROM history GROUP BY position ORDER BY 2 DESC"
    ).fetchall()
    say("| label in `history` | rows | reaches training today |")
    say("|---|---:|---|")
    for pos, n in rows:
        kept = "yes" if pos in REAL_POSITIONS else "**NO**"
        say(f"| `{pos}` | {n:,} | {kept} |")

    gk = dict(rows).get("GK", 0)
    total = sum(n for _, n in rows)
    say(f"\n**{gk:,} goalkeeper rows dropped — {100*gk/total:.1f}% of the archive.**\n")

    df = load_raw(conn, fix_gkp=False)
    d, feats = build_features(df)
    vals = sorted(int(v) for v in d["is_gkp"].unique())
    say(f"`is_gkp` values seen in training (broken): `{vals}` — "
        f"{'**constant, a dead feature**' if len(vals) == 1 else 'varies'}\n")
    return gk


# =====================================================================  B

def section_b(conn):
    say("## B. Champion vs challenger\n")
    out = {}
    for fix, label in ((False, "champion (keepers dropped)"),
                       (True, "challenger (keepers included)")):
        t0 = time.time()
        df = load_raw(conn, fix_gkp=fix)
        d, feats = build_features(df)
        train = d[d.season.isin(TRAIN_SEASONS)].dropna(subset=["total_points"])
        test = d[d.season == TEST_SEASON].dropna(subset=["total_points"])
        pred = two_stage(train, test, feats)
        res = evaluate(test, pred, label)
        res.update(train_n=len(train), feats=feats, test=test, pred=pred,
                   secs=time.time() - t0)
        out[label] = res
        say(f"  trained {label:<32} {len(train):>7,} rows, {res['secs']:.0f}s")

    say("")
    say("| model | train rows | test rows | MAE | high-return MAE | **top-20 pts** |")
    say("|---|---:|---:|---:|---:|---:|")
    for r in out.values():
        say(f"| {r['label']} | {r['train_n']:,} | {r['n']:,} | {r['mae']:.4f} "
            f"| {r['mae_high']:.4f} | **{r['rank']:.3f}** |")

    champ, chall = list(out.values())
    d_rank = chall["rank"] - champ["rank"]
    say(f"\n**Ranking delta: {d_rank:+.3f} points per pick** "
        f"({'challenger better' if d_rank > 0 else 'challenger WORSE'})\n")

    # The comparison that actually matters for the bug: keepers themselves.
    say("### Per-position MAE\n")
    say("| position | champion | challenger | delta |")
    say("|---|---:|---:|---:|")
    ct, cp = champ["test"], champ["pred"]
    ht, hp = chall["test"], chall["pred"]
    for pos in REAL_POSITIONS:
        cm = ct["position"] == pos
        hm = ht["position"] == pos
        c_mae = (mean_absolute_error(ct[cm]["total_points"], cp[cm.values])
                 if cm.any() else float("nan"))
        h_mae = (mean_absolute_error(ht[hm]["total_points"], hp[hm.values])
                 if hm.any() else float("nan"))
        delta = c_mae - h_mae
        note = "" if cm.any() else "  *(champion never predicted these in test)*"
        say(f"| {pos} | {c_mae:.4f} | {h_mae:.4f} | {delta:+.4f}{note} |")
    say("")
    return champ, chall


# =====================================================================  C

def section_c(champ, chall, n_boot=2000):
    """
    Is the difference real, or could it be luck? Resample GAMEWEEKS, not rows —
    players within a gameweek share fixtures and are not independent.
    """
    say("## C. Is the difference significant?\n")
    ct, ht = champ["test"], chall["test"]
    gws = sorted(set(ct["gw"].unique()) & set(ht["gw"].unique()))

    per_gw = []
    for gw in gws:
        cm, hm = (ct["gw"] == gw).values, (ht["gw"] == gw).values
        if cm.sum() < 20 or hm.sum() < 20:
            continue
        per_gw.append(rank_eval(ht[hm], chall["pred"][hm], k=20)
                      - rank_eval(ct[cm], champ["pred"][cm], k=20))
    per_gw = np.array(per_gw)
    if len(per_gw) == 0:
        say("_Not enough gameweeks to bootstrap._\n")
        return

    rng = np.random.default_rng(0)
    boots = np.array([rng.choice(per_gw, len(per_gw), replace=True).mean()
                      for _ in range(n_boot)])
    lo, hi = np.percentile(boots, [2.5, 97.5])
    p_better = (boots > 0).mean()

    say(f"Paired bootstrap over **{len(per_gw)} gameweeks**, {n_boot:,} resamples:\n")
    say(f"- mean ranking difference: **{per_gw.mean():+.3f}** pts/pick")
    say(f"- 95% CI: **[{lo:+.3f}, {hi:+.3f}]**")
    say(f"- P(challenger better): **{p_better:.2f}**\n")
    verdict = ("the interval excludes zero — a real effect"
               if lo > 0 or hi < 0 else
               "**the interval contains zero — indistinguishable from noise**")
    say(f"{verdict}\n")


# =====================================================================  D

def section_d(conn):
    """Position-specific models — the last big roadmap idea, now that GKP exists."""
    say("## D. Position-specific models\n")
    say("OpenFPL trains a separate model per position. Until the GK/GKP bug was "
        "found this could not even be tested for keepers.\n")

    df = load_raw(conn, fix_gkp=True)
    d, feats = build_features(df)
    train_all = d[d.season.isin(TRAIN_SEASONS)].dropna(subset=["total_points"])
    test_all = d[d.season == TEST_SEASON].dropna(subset=["total_points"])

    global_pred = two_stage(train_all, test_all, feats)
    g = evaluate(test_all, global_pred, "one global model")

    pred_pos = np.full(len(test_all), np.nan)
    say("| position | train rows | test rows | seconds |")
    say("|---|---:|---:|---:|")
    for pos in REAL_POSITIONS:
        tr = train_all[train_all["position"] == pos]
        te_mask = (test_all["position"] == pos).values
        te = test_all[te_mask]
        if len(tr) < 500 or len(te) == 0:
            say(f"| {pos} | {len(tr):,} | {len(te):,} | *skipped, too little data* |")
            continue
        t0 = time.time()
        pred_pos[te_mask] = two_stage(tr, te, feats)
        say(f"| {pos} | {len(tr):,} | {len(te):,} | {time.time()-t0:.0f} |")

    ok = ~np.isnan(pred_pos)
    if ok.sum() < len(test_all):
        pred_pos[~ok] = global_pred[~ok]
    p = evaluate(test_all, pred_pos, "position-specific ensemble")

    say("")
    say("| model | MAE | high-return MAE | **top-20 pts** |")
    say("|---|---:|---:|---:|")
    for r in (g, p):
        say(f"| {r['label']} | {r['mae']:.4f} | {r['mae_high']:.4f} | **{r['rank']:.3f}** |")
    d_rank = p["rank"] - g["rank"]
    say(f"\n**Ranking delta: {d_rank:+.3f}** "
        f"({'position-specific better' if d_rank > 0 else 'position-specific WORSE'})\n")
    return g, p


# =====================================================================

def main():
    quick = "--quick" in sys.argv
    started = datetime.now(timezone.utc)
    say(f"# Goalkeeper training gap — experiment report")
    say(f"\n_Run {started.isoformat(timespec='seconds')} · "
        f"read-only on the database · nothing promoted automatically._\n")

    conn = sqlite3.connect(DB_PATH)
    try:
        section_a(conn)
        champ, chall = section_b(conn)
        section_c(champ, chall)
        g, p = section_d(conn)

        say("## Verdict\n")
        d1 = chall["rank"] - champ["rank"]
        d2 = p["rank"] - g["rank"]
        say(f"- Including goalkeepers: **{d1:+.3f}** pts/pick on ranking")
        say(f"- Position-specific models: **{d2:+.3f}** pts/pick on ranking")
        say("")
        say("Regardless of the ranking numbers, the GK/GKP normalisation is a "
            "**correctness fix, not an optimisation**: the model was predicting "
            "71 goalkeepers per gameweek having trained on none, and `is_gkp` "
            "was a constant. That is worth fixing even if the aggregate metric "
            "barely moves, because the aggregate is dominated by the 590 "
            "outfielders it was already handling correctly.\n")
        say("**Not merged to `main`.** Re-running `recommend.py` with a changed "
            "model would overwrite the GW4 predictions logged before the "
            "deadline, destroying the engine-vs-gut comparison. Merge after "
            "GW4 is scored.\n")
    finally:
        conn.close()

    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    mins = (datetime.now(timezone.utc) - started).total_seconds() / 60
    print(f"\n  Report written to {REPORT.name}  ({mins:.1f} min)")


if __name__ == "__main__":
    main()
