# Progeny FPL Engine

A data-driven decision support system for Fantasy Premier League: predicts player
points, picks the optimal squad under FPL's rules, recommends a captain, and alerts
on team-news changes before the deadline.

Built as a personal project applying MSc Data Science skills to a real weekly decision.
**Zero third-party dependencies beyond numpy / pandas / scikit-learn / scipy** — no
PuLP, no XGBoost, no scraping frameworks.

---

## Headline results

Measured on the **2025-26 season**, held out entirely from training (trained on
2022-23 → 2024-25, 74,440 rows; tested on 26,320 rows the model never saw).

| Model | MAE | High-return MAE | **Top-20 picks: mean actual points** |
|---|---|---|---|
| `xP` — FPL's own expected points | 1.115 | 6.534 | 3.29 |
| `direct` | 1.024 | **5.110** | 4.29 |
| **`two_stage` (champion)** | **0.954** | 5.305 | **4.39** |

**Both models beat FPL's own published expected-points model.** On the
decision-relevant ranking metric the champion delivers **4.39 points per pick vs
FPL's 3.29 — about +1.1 per pick (~33% better)**.

### Captaincy is worth more than squad selection

Simulated across all 38 gameweeks of 2025-26, captain chosen each week from the 50
most-owned players:

| Strategy | pts/GW | over 38 GWs |
|---|---|---|
| Most-owned (the template pick) | 9.63 | 366 |
| `xP` (FPL's own) | 13.11 | 498 |
| **`two_stage` (champion)** | **13.63** | **518** |
| `p_haul` — P(points ≥ 10) | 11.58 | 440 |
| `q90` — 90th-percentile ceiling | 9.21 | 350 |

**+4.00 pts/GW over the template ≈ +152 points per season.**

> **A common belief this project tested and disproved.** FPL wisdom says captaincy is
> a *ceiling* decision — pick the explosive player, not the highest average. Both
> ceiling strategies lost, and `q90` was the worst tested. The reason is simple:
> captain points double **linearly**, so maximising `E[2X]` is just maximising `E[X]`.
> Expected value is provably the right criterion. The ceiling framing only applies
> when chasing *rank* in a mini-league, where variance is deliberately desirable.

### Minutes dominate everything

Permutation importance on the direct model:

```
minutes_last        0.1806   <- 4x the next feature
minutes_m3          0.0417
ict_last            0.0343
ict_m3              0.0329
total_points_m10    0.0325
minutes_m5          0.0316
```

Four of the top ten features are minutes-based. Across 113,582 historical rows,
**27.6% of player-fixtures are 60+ minutes and they hold 84.9% of all points.**
This is why the news/availability layer matters more than any modelling refinement.

---

## Project status

| Phase | State |
|---|---|
| **0 · Data spine** | Complete — collector automated, 113,582 historical rows loaded |
| **1 · Baseline + scoreboard** | Complete — prediction log, honest backtest, ranking metric |
| **2 · Real model** | Complete — beats FPL's `xP`; live recommendations |
| **3 · Optimiser** | Complete — MILP squad / transfers / wildcard |
| **4 · Minutes & news edge** | Partial — alerter built and scheduled; European congestion still outstanding |
| **5 · Self-improving loop** | Built but **unproven** — no gameweek has been scored live yet |

> **Important caveat.** Every result above is a **backtest**. Predictions for GW4 are
> logged but not yet graded. The system has never made a verified live prediction.

---

## Quick start

Requires Python 3.12 with `numpy`, `pandas`, `scikit-learn`, `scipy`.

```bash
# 1. Build the database (~2 minutes)
python fpl_collect.py                 # snapshot the live FPL API
python backfill_history.py --load     # 4 seasons, 113,582 rows
python predict.py --backfill          # this season's actual results

# 2. Verify everything is sound
python qa.py --all                    # 33 checks

# 3. Get this week's recommendation
python recommend.py --run             # trains, predicts, logs before the deadline
python optimise.py --transfers --free 1
```

### Rebuilding from scratch

`fpl.db` is gitignored because it is ~40 MB and fully regenerable. The three commands
in step 1 rebuild it. Note that **snapshot history cannot be recovered** — the FPL API
only ever exposes current state, so prices, ownership and injury flags from past days
are gone unless they were captured at the time. This is why collection is automated.

---

## The scripts

| Script | Purpose | Modes |
|---|---|---|
| `fpl_collect.py` | Snapshot the live API into SQLite; prunes + vacuums automatically | `--summary` `--prune` |
| `backfill_history.py` | Load 4 historical seasons | `--load` `--verify` |
| `predict.py` | Baselines, prediction log, backtests | `--backfill` `--predict` `--score` `--backtest` `--rank` `--status` |
| `minutes.py` | Minutes analysis, persistence baseline | `--explore` `--persist` |
| `model.py` | Train and compare ML models | `--train` |
| `captain.py` | Season-long captaincy simulation | `--run` |
| `recommend.py` | Live recommendation + logs predictions | `--run` |
| `optimise.py` | MILP squad / transfer / wildcard | `--squad` `--transfers` `--wildcard` `--free N` `--bank X` `--budget X` `--horizon N` |
| `record_my_team.py` | Record actual squad + captaincy review | `--show` |
| `alert.py` | Diff snapshots for team-news changes | `--check [--hours N]` `--watch-wildcard` `--watchlist` |
| `opponent.py` | Opponent-strength features — **tested and rejected**, kept as a documented negative result | *(imported)* |
| `compare.py` | **Engine vs gut scoreboard** — XI and captaincy, expected then actual | `--gw N` `--fdr` `--score` `--season` |
| `publish.py` | Database → the small JSON files the web app reads | `--print` |
| `app.py` | Streamlit front end; reads JSON only, no database, no model | `streamlit run app.py` |
| `qa.py` | Modelling QA suite (33 checks) | `--all` |
| `qa_deploy.py` | Deployment QA suite (88 checks) — JSON contract, workflow, app constraints | *(no args)* |

---

## Data model

SQLite, one file. Key tables:

| Table | Rows | Notes |
|---|---|---|
| `snapshots` | grows | One per collection run; the timeline spine |
| `players` | 654 × snapshots | Prices, ownership, **`news`**, `chance_next_round`, `status` |
| `fixtures` | 380 × snapshots | With difficulty ratings |
| `history` | 113,582 | Four static seasons — **one row per player PER FIXTURE** |
| `player_gw` | grows | This season's actual results, 18 columns incl. xG/ICT/BPS |
| `predictions` | grows | Every prediction, timestamped **before the deadline** |
| `my_squad` / `my_gameweeks` | small | Your real team, for benchmarking |
| `watchlist` | 15 | Players the alerter tracks closely |

> **Trap: `history` is keyed `(season, gw, element, fixture)`.** In a *double gameweek*
> a player has two fixtures in one GW, so `(season, gw, element)` is **not unique** —
> keying on it silently drops one fixture per DGW (374 cases in 2024-25 alone), and
> those are exactly the weeks that decide chips. **For per-gameweek totals, SUM over
> fixtures.**

---

## Methodology notes

These are the decisions that make the numbers trustworthy.

**Chronological validation only.** Train on earlier seasons, test on a later one.
Random k-fold on time-series data leaks the future into the past and produces
flattering, useless scores.

**No feature leakage.** Every feature is built with
`groupby(season, element).shift(1)` *before* any rolling window, so row *N* sees rows
1..*N*-1 and nothing else. Verified by a negative control: training on a **shuffled
target** gives MAE 1.551 vs 1.537 for predicting the mean — i.e. no skill, which is
the definitive proof.

**Ranking, not MAE, is the headline metric.** You never consume a prediction of "4.2
points" — you pick the best ~15 players. `predict.py --rank` and the top-K evaluation
in `model.py` measure what the model's favourites actually scored.

**High-return MAE is biased and must not be used alone.** That subset is conditioned
on players who *did* play, which systematically punishes any model that correctly
discounts for the risk of not playing. This is why `two_stage` loses on high-return
MAE while winning on ranking.

**Predictions are logged before the deadline.** `recommend.py` refuses to log after
the deadline has passed, because a prediction made afterwards is not a prediction.

---

## QA

`python qa.py --all` — **33 checks**, exits non-zero on failure so it can gate CI.

- **DATA** — row counts, duplicate keys, double-gameweek preservation, ranges, NULLs
- **LOGIC** — captain doubling, predictions logged pre-deadline
- **LEAKAGE** — manual feature recomputation, target-copy check, negative control, chronology
- **MODEL** — reproducibility, overfit ratio, beats trivial baselines, sane prediction range
- **OPTIMISER** — squad legality re-verified against the FPL rulebook independently of the solver

Two lessons from building it:

1. **A test that cannot fail is not a test.** The original feature-recomputation check
   compared `0.0000` vs `0.0000` and passed while proving nothing. It now verifies 12
   consecutive rows against the season's top scorer.
2. **Both initial "failures" were bugs in the tests, not the pipeline** — a 60%
   target-match rate that was really 50% trivial `0 == 0`, and "points without minutes"
   that turned out to be *managers* (FPL added "AM" as a position in 2024-25).

---

## Deployment

The system runs on **GitHub Actions + Streamlit Community Cloud**, at zero cost.

```
GitHub Actions (the clock)              Streamlit Cloud (the window)
──────────────────────────              ───────────────────────────
06:00 & 19:00 UTC daily
  ├── fpl_collect.py    snapshot
  ├── predict.py        backfill, score
  ├── recommend.py      model + log
  ├── alert.py          diff availability
  ├── qa.py --all                        reads site/data/*.json
  ├── qa_deploy.py      ── gate ──       renders squad, captain,
  ├── publish.py        JSON      ────►  picks, news, accuracy
  ├── git commit site/data
  ├── fpl.db ──► release asset
  └── email if anything is urgent  ────► your inbox
```

### Why the work and the web page are separate

Streamlit Community Cloud cannot host this project on its own, and the reasons
are structural rather than cosmetic:

| Constraint | Consequence |
|---|---|
| Ephemeral filesystem | `fpl.db` is destroyed on every reboot and redeploy |
| No scheduler | Nothing runs unless a browser is open |
| Sleeps after 12 quiet hours | A decision system that only thinks while observed |
| 1 GB memory cap | Not enough headroom to train comfortably |

So Actions does the thinking on a schedule and commits ~24 KB of JSON;
Streamlit only draws it. The app opens no database and imports neither
scikit-learn nor scipy — `qa_deploy.py` fails the build if that ever changes.

`fpl.db` is 23 MB and would bloat git history if committed twice a day, so it
lives as a **release asset**: downloaded at the start of each run, uploaded at
the end.

### Why not Railway

Railway Hobby was measured and would work — full retrain **43.8 s / 210 MB
peak**, ~**$0.22/month** — but it draws on the *same $5 credit* that keeps
`bcaChessHub` alive, and credit exhaustion is what took that site down in
September 2026. Actions + Streamlit costs nothing and removes the contention
entirely. Actions usage is ~180 of the 2,000 free minutes per month; on a
public repo it is unmetered.

### Setting it up

1. Deploy `app.py` from this repo at [share.streamlit.io](https://share.streamlit.io).
2. Add three repository secrets for the email alerts (Settings → Secrets →
   Actions). Without them the pipeline still runs; it just does not email.

   | Secret | Value |
   |---|---|
   | `MAIL_USERNAME` | Gmail address |
   | `MAIL_PASSWORD` | Google **app password** (not the account password) |
   | `MAIL_TO` | where alerts should arrive |

3. Run the workflow once by hand (Actions → FPL pipeline → Run workflow) to
   bootstrap the database release.

> **Scheduled workflows are disabled after 60 days of repository inactivity.**
> This one commits twice a day, so it keeps itself alive — but if the pipeline
> is ever paused for two months, re-enable it in the Actions tab.

### Use cases

| Actor | Use cases |
|---|---|
| **Manager (you)** | View recommendation · Run optimiser · Plan wildcard · Review model performance |
| **Scheduler (cron)** | Collect snapshot · Detect news change · Send email alert · Retrain & score |
| **FPL API** | Provides player, fixture and availability data |

### Screens (built — `app.py`)

| Tab | Shows |
|---|---|
| **Squad & captain** | Your 15 with expected points, the captaincy call, and where it disagrees with you |
| **Recommendations** | Top 40 by xPts, filterable by position, price and availability |
| **Team news** | Availability changes since the last snapshot, prioritised, plus price moves |
| **Model accuracy** | Error per gameweek per model — scored only against predictions logged *before* the deadline |
| **My season** | Your own results, the benchmark the model has to beat |

Live countdowns are recomputed in the browser rather than read from the JSON:
`hours_to_deadline` was true when the pipeline ran, and a stale countdown is
worse than none. The page also warns when its own data is more than 14 hours
old, which means a scheduled run failed.

### Automation (built — `.github/workflows/fpl.yml`)

| Cron (UTC) | Why then |
|---|---|
| `0 6 * * *` | Catches overnight news; several hours' warning before a Saturday 11:30/12:30 deadline |
| `0 19 * * *` | Catches the afternoon press conferences, when most injury news actually lands |
| `workflow_dispatch` | Run by hand right before a deadline |

Email is conditional, not unconditional: `publish.py` writes a `should_email`
flag, and the workflow skips the send step when nothing is urgent. A daily
"nothing happened" email trains you to ignore the inbox, which defeats the
purpose of having an alerter at all.

Email fires when a **squad or watchlist player's availability changes**, or when
the **deadline is within 26 hours**.

### Deployment risks

1. ~~**Storage.**~~ **RESOLVED** — see "Storage management" below. Growth is now
   **0.098 MB per snapshot (~52 MB/season)**, measured, down from 1.7 MB.
2. ~~**Always-on cost.**~~ **RESOLVED** — no always-on service exists. Actions
   bills only for minutes used (~180 of 2,000 free, unmetered on a public repo)
   and Streamlit's free tier is unlimited for public apps.
3. **Database loss.** `fpl.db` lives in one release asset. `history` and
   `player_gw` are regenerable, but the **snapshot series is not** — a gameweek
   of injury-news history that was never captured cannot be recovered. The
   committed `site/data/*.json` is a partial hedge, not a backup.

---

## Design

The front end has a design system, `site/assets/pfl.css`, and it is not
decoration. Read its header before changing anything visual.

**Structure from the pitch. Colour from the Warriors kit.** Every FPL tool on
the internet clones the official purple-and-mint palette, and every football
site reaches for turf green. This one draws chalk-line pitch furniture, which
renders from flat colour and needs no images, in the Zimbabwe kit colours:
Warriors green, the gold of the Zimbabwe Bird, flag red held back for genuine
alarm. Competitive surfaces (captaincy, accuracy) go dark and near monochrome,
which is the pitch at kick-off. The personal ones (season record, photographs)
are warm and open.

It is the same method as [Bulawayo Chess Hub](https://github.com/percivalmahwaya/bcaChessHub),
whose `static/css/bch.css` carries the same five house rules, from the same
person:

- no em dashes in interface copy
- no emoji used as icons
- no gradients at all, meaning no blend between two colours
- no centred hero with a gradient background and two buttons
- no row of three feature cards each with an icon in a circle

**`qa_deploy.py` enforces the first three** under `DESIGN HOUSE RULES`. They
are taste rather than correctness, which is exactly why they need a check:
nothing breaks when an emoji creeps back into a heading, so nothing stops it.
Each one is verified to fail by injecting the violation.

### Photographs

```bash
python tools/optimise_photos.py ~/Pictures/some-folder   # 120 KB each, 4 max
python tools/make_mark.py                                # regenerate the tab icon
```

Drop images in `site/assets/photos/` and the "My season" tab shows them. The
caption is the filename. See that folder's README.

### Weight, honestly

Measured on a phone viewport, first load: **5171 KB, of which 4894 KB is
Streamlit's own JavaScript.** The stylesheet is 13 KB and loads no fonts, no
icon set and nothing from a CDN, but that is rearranging deckchairs. Bulawayo
Chess Hub went from 329 KB to 19 KB; nothing comparable is possible here. It is
acceptable only because this app has one reader on his own connection. For an
audience on a metered bundle the answer would not be a lighter stylesheet, it
would be a different host.

---

## Storage management

`players.raw` stores the complete FPL JSON for all 654 players on every run. Nothing
reads it; it exists only so a field nobody thought to store can still be recovered.
Left alone it was **94% of all database growth**, projecting ~3 GB/season against a
5 GB Railway volume.

`fpl_collect.py` now prunes and vacuums automatically on every run, keeping the blob
for the newest snapshot only. Run it manually with `python fpl_collect.py --prune`.

> **Pruning alone does not work, and this was measured rather than assumed.**
> `UPDATE ... SET raw = NULL` shrinks rows *in place*, leaving gaps inside pages
> rather than whole free pages — `PRAGMA freelist_count` stayed at **0**. New inserts
> cannot use those fragments, so the file kept extending by **+2.72 MB per snapshot,
> worse than doing nothing at all**. `VACUUM` rewrites the file and is what actually
> reclaims the space.

Measured steady state after the fix:

| Snapshot | Size | Growth |
|---|---|---|
| before | 33.68 MB | — |
| 1 | 23.24 MB | **−10.44 MB** (first vacuum) |
| 2 | 23.34 MB | +0.098 MB |
| 3 | 23.44 MB | +0.098 MB |
| 4 | 23.53 MB | +0.094 MB |

**≈52 MB/season.** The cost is ~5 s added to each collection run, twice a day.

---

## Local automation

Two Windows scheduled tasks, kept as a redundant local copy:

```
FPL Snapshot          09:00, 21:00   -> fpl_collect.py
FPL Team News Alert   21:15          -> alert.py --check >> alerts.log
```

These duplicate what GitHub Actions now does, and only capture snapshots while
this machine is on. They are harmless — snapshots are additive — but the cloud
pipeline is the source of truth. Remove them with
`Unregister-ScheduledTask -TaskName "<name>" -Confirm:$false`.

---

## Roadmap

- [x] ~~Drop the `raw` storage bloat~~ — done; growth 1.7 MB -> 0.098 MB per snapshot
- [ ] Score GW4 live — the **first real validation** of everything above
- [ ] **Opponent difficulty features — the model is currently fixture-blind.** All 47
      features describe the player; none describe who he is facing. It predicts the
      same score against Man City as against Hull City. Data is already available
      (`fixtures.difficulty_h/_a`, and for history both clubs' players share a
      `fixture` id so the opponent name is derivable). Likely higher value than
      position-specific models.
- [ ] Position-specific models (OpenFPL's approach; still outstanding)
- [ ] European fixture congestion — CL/EL fixtures are not in the FPL API
- [ ] Deploy to Railway with cron + email (est. 14–17 h)
- [ ] Champion/challenger promotion: only ship a model that wins a held-out backtest

---

## Prior art

Standing on rather than reinventing:

- **[OpenFPL](https://github.com/daniegr/OpenFPL)** ([paper](https://arxiv.org/html/2508.09992v1)) —
  position-specific ensembles matching commercial services. Its documented weakness is
  **expected minutes**, which is precisely the gap this project targets.
- **[vaastav/Fantasy-Premier-League](https://github.com/vaastav/Fantasy-Premier-League)** —
  the historical archive, with xG/xA already merged in.
- The FPL API itself — public, no auth, and richer than most people realise
  (28 fields per player per gameweek including xG, xA, ICT, BPS).

---

## Notes

- **Windows console encoding.** Printing player names crashes with
  `UnicodeEncodeError` on cp1252 (e.g. `ć`). Use `optimise.safe()` to fold accents.
- **FPL `web_name` is not unique.** There are two "Palmer" and two "Martinez". Match on
  `(web_name, position)`; matching on name alone silently picks the wrong player.
- **`chance_of_playing` is NULL for fit players**, not 0. Normalise to 100 or "cleared
  to play" reads as missing data.
- **SQLite does not reclaim space on UPDATE.** Nulling a large column leaves
  intra-page fragmentation, not free pages. Only `VACUUM` shrinks the file.
