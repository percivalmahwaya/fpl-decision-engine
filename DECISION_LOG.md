# DECISION_LOG — what was tried, what was kept, what was rejected, what was reversed

**Why this file is called a decision log and not a history.** A history is a story and stories get
told favourably. This is a ledger: every row is a question that was asked, the method used to
answer it, the verdict, and — the column that matters most — **whether that verdict was later
overturned.** Two of the entries below were.

**Built from the contemporaneous record, not from memory.** Sources, in order of trustworthiness:
the git log (120 commits, 2026-09-09 to 2026-10-15, written at the time); `FINDINGS_2026-09-11.md`
and `FINDINGS_2026-09-19.md`; the `exp_*.py` scripts, each of which *is* a recorded experiment
still runnable today; and `MODEL_CHANGES` in `publish.py`, which publishes alongside the accuracy
data so the scoreboard cannot silently compare two different models under one name.

**Reproducibility rule for every row below:** if a verdict cannot be re-derived by running a
script in this repo, it is marked **[unreproducible]** and should be trusted less than one that can.

---

## The ledger

### 2026-09-09 — Opponent-strength features. REJECTED, then REVERSED.

**Question:** the model is fixture-blind. Does knowing the opponent help?
**Method:** `opponent.py`. Measured on ranking — mean actual points of the top-20 picks per
gameweek.
**Verdict at the time:** rejected. Ranking went 4.39 → 4.27, i.e. worse.
**What was recorded but not acted on:** high-return MAE *improved* in the same run, 5.305 → 5.169.
That was logged as a consolation prize rather than treated as a signal.
**Later overturned — see 2026-09-19.** The rejection was made on a metric chosen before any
gameweek had ever been scored live. [reproducible: `python opponent.py`]

### 2026-09-11 — The goalkeeper training gap. FOUND, DELIBERATELY NOT FIXED.

**Question:** why does `is_gkp` never fire?
**Method:** direct inspection of the archive against the live API's labels.
**Finding:** the archive spells the position `GK`, everything else spells it `GKP`. **12,500 rows
— 11% of the entire archive — never reached training**, while 71 keepers a week were predicted
anyway by a model that had never seen one. Nothing crashed; nothing warned.
**Decision: do not fix yet.** `recommend.py` uses `INSERT OR REPLACE`, so retraining mid-gameweek
would overwrite predictions already logged for that gameweek and destroy the ability to grade it.
Fix only straight after a gameweek is scored, never before a deadline.
**This is the best decision in the log** — a real bug, found, understood, and deliberately left
alone for three weeks because fixing it at the wrong moment would have corrupted the evidence.
[reproducible: `FINDINGS_2026-09-11.md`]

### 2026-09-11 — The noise floor. ACCEPTED, still in force.

**Question:** how big does a difference in expected points have to be before it means anything?
**Method:** `exp_stability.py`.
**Verdict:** **0.17 expected points.** Below that the engine has no real preference and must say
so rather than rank two players anyway. Now used in five places in the interface.
[reproducible: `python exp_stability.py`]

### 2026-09-15 — Own results: hand transcription → live API. REPLACED.

**Question:** the benchmark against Percival's own team was a dict transcribed from screenshots.
Can it be fetched?
**Finding:** yes, from two public endpoints, no login. The cron had been "succeeding" twice a day
while rewriting the same four hand-typed gameweeks forever — **a green step that had never once
done the thing it was named after.**
**Kept deliberately:** the transcription survives as a cross-check (`--verify`), not as the source.
It paid for itself immediately, catching a GW4 transfer count of 1 against the API's 0 — neither
wrong, because FPL reports 0 transfers on any chip week.

### 2026-09-19 — The opponent rejection, REOPENED on better evidence.

**Question:** was the 2026-09-09 rejection measured on the wrong thing?
**Why reopened:** GW4 was scored live on 09-15 — the first real accuracy data the project ever
had — and it said the champion model has the **best overall MAE and the worst high-return MAE**.
The model's actual weakness is the tail, which is exactly what the opponent features improved.
**Method:** re-run as a 2×2 with the goalkeeper fix, because both change what the model learns
from and two changes measured separately can each look neutral and still interact.
**Result:** every arm improves high-return MAE with P(better) = 1.00; `+ both` best at −0.214.
Every arm costs ~0.08–0.11 on ranking, none statistically clear. **A real trade-off, not a free win.**
**Verdict: nothing promoted.** Registered as a fourth model on the scoreboard instead, to be
decided by live gameweeks rather than by one backtest.
[reproducible: `python exp_reevaluate.py`]

### 2026-09-19 — A methodology error, caught by this project's own earlier document.

**What happened:** the first re-run let the goalkeeper fix change the *test* set as well as the
training set, producing 0.954 → 0.912 — numbers `FINDINGS_2026-09-11.md` had already flagged as
superseded, for the reason it gave: keeper rows have MAE ~0.58 and drag the mean down regardless
of skill. 26,320 test rows versus 29,747.
**Corrected to:** identical test rows for every arm; only training varies.
**Consequence:** the champion's real MAE is **0.905, not 0.954**, and the keeper fix makes overall
MAE slightly *worse*. It was shipped anyway, as a correctness fix, and the commit said so.
**This is the single most useful precedent in the log** — comparing on different rows measures
composition, not skill, and the project's own prior document is what caught it.

### 2026-09-22 — The goalkeeper fix, SHIPPED. And the scoreboard that would have lied.

**Window chosen deliberately:** GW5 scored, GW6 deadline 18 days out, zero GW6 predictions logged.
The safest moment the project would ever get.
**Fixed at the boundary, not the call sites:** twenty places already agreed on `GKP`; the archive
was the lone dissenter, so the archive changed. Confirmed in CI: 12,500 rows normalised, training
rows 100,760 → 113,260.
**The part that matters:** `two_stage_ml` now means something different from GW6 onward. GW4 and
GW5 were scored by a model that had never seen a keeper. `MODEL_CHANGES` records this, publishes
with the accuracy data, and renders **above** the numbers. **Add a row, never edit one.**

### 2026-10-15 — Captaincy as a decision, not a prediction. DEMOTED, then REVERSED SAME NIGHT.

**Question:** does following the engine's captain pick beat Percival's own judgement?
**First method:** paired comparison, GW4–GW6. **n = 3.**
**First verdict:** DEMOTE. Cumulative −28 points; engine better in 0 of 3.
**Reversed hours later.** The criterion had pre-registered, before any number was computed, that
n = 3 could not settle the question and that nothing counted until n ≥ 10. A proper test —
trained on 2022–25, tested on the fully held-out 2025-26 season, 38 gameweeks — found the model
beats the "captain your most expensive player" baseline by **+76 points**, and that P(haul) rises
monotonically with its prediction from 0.3% to 15.9% against a 4.1% base rate. **The model is not
blind to the tail.**
**What stands:** model > dumb baseline is measured. Model > *Percival* is not, at any usable
sample size. Those are different claims and only the first may be made in the product.
[reproducible: `osp-scout/tools/captaincy_diag.py`]

---

## What this log is for

Three uses, in order of how often they will matter:

1. **Before re-deriving anything, read this.** Two verdicts here were overturned, and both
   reversals came from asking "what was this measured on?" rather than from new data. A future
   session that re-runs a settled question without reading why it was settled will repeat the
   2026-09-09 rejection.
2. **Before trusting a number, check whether its row says `[reproducible]`.** A verdict that can
   be re-derived by running a script is a different kind of claim from one that cannot.
3. **Before promoting a model, add a `MODEL_CHANGES` row.** The scoreboard compares models by name
   across gameweeks. Change what a name means without recording it and the comparison silently
   becomes meaningless.

## The pattern across every reversal in this log

Both overturned verdicts failed the same way: **a real measurement, on the wrong population.**
Opponent features were judged on ranking before any gameweek had been scored live. Captaincy was
judged on three gameweeks when four seasons were sitting in the same database.

Neither was a modelling error. Both were sampling errors, and both were caught by asking what the
number was computed over — never by looking harder at the number itself.
