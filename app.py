"""
Progeny FPL Engine, the web front end.

READ THIS BEFORE ADDING ANYTHING HEAVY HERE
===========================================
This app deliberately does no work. It opens no database, trains no model and
imports neither scikit-learn nor scipy. It reads a handful of small JSON files
that GitHub Actions produced on a schedule, and draws them.

That is not laziness, it is the constraint that makes free hosting viable:

  * Streamlit Community Cloud's filesystem is ephemeral, so a 25 MB fpl.db
    committed here would be wiped on every reboot anyway.
  * The free tier caps memory at 1 GB. Training the two-stage model needs far
    more headroom than that leaves.
  * Apps sleep after 12 quiet hours, so anything that only happens "when the
    page loads" happens roughly never.

Every decision this page shows was already made by the pipeline, before the
deadline, and logged. The page is a window onto that record, not a calculator.
If you find yourself wanting to compute something here, compute it in
publish.py instead and read the answer.

ON THE LOOK
===========
The design system is site/assets/pfl.css. Read its header before changing
anything visual. It carries five house rules from Percival that are not
negotiable, the ones most likely to be broken by accident being NO EMOJI USED
AS ICONS and NO EM DASHES IN INTERFACE COPY. Both were in this file before and
both are gone. qa_deploy.py now fails the build if they come back.

Run locally:   streamlit run app.py
"""

import html
import json
from datetime import datetime, timezone
from pathlib import Path

# altair is a direct dependency of streamlit, so importing it costs nothing at
# install time and nothing at runtime that streamlit was not already paying.
# It is here because st.line_chart cannot set an axis: gameweeks came out
# labelled 1.0, 1.5, 2.0, and forcing them to strings made Vega rotate the
# labels on their side. Both are worse than writing the chart out properly.
import altair as alt
import pandas as pd
import streamlit as st

ROOT = Path(__file__).parent
DATA = ROOT / "site" / "data"
ASSETS = ROOT / "site" / "assets"
PHOTOS = ASSETS / "photos"

# Measured, not guessed. Reseeding the model alone moves a pickable player's
# prediction by 0.169 expected points on average and by as much as 1.56, so any
# gap narrower than this carries no information. The engine used to report such
# gaps to two decimal places and call one player better, which is how a 0.08
# difference came to look like a disagreement worth having. See
# FINDINGS_2026-09-11.md on the fix/goalkeeper-training-gap branch.
NOISE_FLOOR = 0.17

# Chart series, in order. Streamlit draws its own default blue otherwise, which
# is the one colour on the page that belongs to no part of this design. Kept
# here rather than in pfl.css because charts are canvas, not DOM, so no
# stylesheet can reach them. These must stay in step with the CSS tokens.
SERIES = [
    "#1f7a3d",   # --pfl-green,  the model
    "#d8a222",   # --pfl-gold,   first baseline
    "#6a6a5f",   # --pfl-stone,  second baseline
    "#b3241a",   # --pfl-red,    anything beyond, and a hint it is unexpected
]

st.set_page_config(
    page_title="Progeny FPL Engine",
    page_icon=str(ASSETS / "mark.png") if (ASSETS / "mark.png").exists() else None,
    layout="wide",
)


# ------------------------------------------------------------------ chrome

def stylesheet():
    """The design system. Inlined because Streamlit serves no static assets.

    Deliberately NOT cached. It was, for an hour, and the effect was that
    every edit to pfl.css appeared to do nothing: the browser kept being
    served the stylesheet from before the change, which is a genuinely
    confusing way to lose half an hour. Re-reading 13 KB from local disk on
    each run is free next to the megabytes of JavaScript Streamlit ships.
    """
    css = ASSETS / "pfl.css"
    return css.read_text(encoding="utf-8") if css.exists() else ""


st.markdown(f"<style>{stylesheet()}</style>", unsafe_allow_html=True)


def block(markup):
    """Render one pre-built HTML block."""
    st.markdown(markup, unsafe_allow_html=True)


def e(value):
    """Escape anything that came out of the FPL API before it meets HTML."""
    return html.escape(str(value), quote=True)


def empty(title, body):
    block(f'<div class="pfl-empty"><strong>{e(title)}</strong>{e(body)}</div>')


def line(df, x, y, colour_by=None, y_title="", height=260, domain=None):
    """A line chart on the palette, with gameweeks labelled as gameweeks.

    Deliberately plain: no points, no area fill, no chart junk. Gridlines are
    horizontal only and barely there, because the reader is comparing heights
    and everything else is decoration competing with the data.
    """
    enc = {
        "x": alt.X(f"{x}:O", title="Gameweek",
                   axis=alt.Axis(labelAngle=0, labelExpr="'GW' + datum.value",
                                 grid=False, tickColor="#e4dfd0",
                                 domainColor="#e4dfd0", labelColor="#6a6a5f",
                                 titleColor="#6a6a5f")),
        "y": alt.Y(f"{y}:Q", title=y_title or y.title(),
                   scale=alt.Scale(zero=False, nice=True, domain=domain)
                   if domain else alt.Scale(zero=False, nice=True),
                   axis=alt.Axis(gridColor="#ece7da", domain=False,
                                 tickSize=0, labelColor="#6a6a5f",
                                 titleColor="#6a6a5f")),
    }
    if colour_by:
        order = ([c for c in df[colour_by].unique() if c == "two_stage_ml"]
                 + [c for c in df[colour_by].unique() if c != "two_stage_ml"])
        enc["color"] = alt.Color(
            f"{colour_by}:N", title=None,
            scale=alt.Scale(domain=order, range=SERIES[:len(order)]),
            legend=alt.Legend(orient="top", labelColor="#33332b", symbolType="stroke"))

    chart = alt.Chart(df).mark_line(strokeWidth=2.5, color=SERIES[0]).encode(**enc)
    st.altair_chart(chart.properties(height=height).configure_view(stroke=None),
                    use_container_width=True)


def caption(path):
    """Photo captions come from the filename, so adding one needs no code.

    `katowice-2022-the-flag.jpg` becomes "katowice 2022 the flag". A leading
    number is stripped so files can be ordered without the digits showing.
    """
    stem = path.stem.split("_", 1)[-1] if path.stem[:2].isdigit() else path.stem
    return stem.replace("-", " ").replace("_", " ").strip()


# ------------------------------------------------------------------ data

@st.cache_data(ttl=300)
def load(name):
    """Read one published JSON file. Missing files are not fatal."""
    path = DATA / f"{name}.json"
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def parse_iso(s):
    if not s:
        return None
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def humanise(hours):
    if hours is None:
        return "unknown"
    if hours < 0:
        return "passed"
    d, h = divmod(hours, 24)
    if d >= 1:
        return f"{int(d)}d {int(h)}h"
    m = (hours - int(hours)) * 60
    return f"{int(hours)}h {int(m)}m"


def status_of(player):
    """One short word instead of a warning triangle.

    A word survives a font fallback, reads correctly in a screen reader, and
    does not render as an empty box on the older Android handsets this is
    actually opened on. An emoji does none of those things.
    """
    if not player.get("flagged"):
        return ""
    chance = player.get("chance")
    if chance == 0:
        return "out"
    if chance is None:
        return "doubt"
    return f"{int(chance)}%"


meta = load("meta")

if meta is None:
    block(
        '<div class="pfl-head">'
        '<p class="kicker">Nothing published yet</p>'
        '<h1><span class="pfl-mark"></span>Progeny FPL Engine</h1>'
        '<p class="lede">The GitHub Actions workflow writes the JSON this page '
        'reads. Once it has run at least once, every panel below fills in by '
        'itself. Nothing here is computed in the browser.</p>'
        '</div><div class="pitch-rule"></div>'
    )
    st.stop()


# ---------------------------------------------------------------- masthead

deadline = parse_iso(meta.get("deadline"))
live_hours = (deadline - datetime.now(timezone.utc)).total_seconds() / 3600 if deadline else None
generated = parse_iso(meta.get("generated_at"))
age = (datetime.now(timezone.utc) - generated).total_seconds() / 3600 if generated else None

next_gw = meta.get("next_gw", "?")

# Urgency is earned, not decorated. Gold only inside the last day before a
# deadline, red only when the data is old enough to be wrong.
dl_class = "urgent" if (live_hours is not None and 0 <= live_hours < 24) else ""
age_class = "alarm" if (age is not None and age > 14) else ""

block(
    '<div class="pfl-head">'
    '<p class="kicker">Fantasy Premier League, decided on the record</p>'
    '<h1><span class="pfl-mark"></span>Progeny FPL Engine</h1>'
    '<p class="lede">Every prediction below was written before its deadline and '
    'has not been touched since. The engine gets no second guesses, and neither '
    'do I. Where the two of us disagree, the scoreboard settles it.</p>'
    '</div>'
    '<div class="pfl-stats">'
    f'<div class="pfl-stat"><span class="k">Next gameweek</span>'
    f'<span class="v">GW{e(next_gw)}</span></div>'
    f'<div class="pfl-stat"><span class="k">Deadline in</span>'
    f'<span class="v {dl_class}">{e(humanise(live_hours))}</span></div>'
    f'<div class="pfl-stat"><span class="k">Snapshot</span>'
    f'<span class="v">{e(meta.get("snapshot_id", "?"))}</span></div>'
    f'<div class="pfl-stat"><span class="k">Data age</span>'
    f'<span class="v {age_class}">{e(humanise(age) if age is not None else "unknown")}</span></div>'
    f'<div class="pfl-stat"><span class="k">Rows of history</span>'
    f'<span class="v">{meta["row_counts"]["history"]:,}</span></div>'
    '</div>'
    '<div class="pitch-rule"></div>'
)

if live_hours is not None and live_hours < 0:
    st.info(f"The GW{next_gw} deadline has passed. Results are scored once the "
            "gameweek finishes and the API confirms them, which can take a day "
            "after the last match.")
elif live_hours is not None and live_hours < 26:
    st.warning(f"GW{next_gw} deadline in {humanise(live_hours)}, at "
               f"{meta.get('deadline')}.")

if age is not None and age > 14:
    st.error(
        f"This data is {humanise(age)} old. The scheduled workflow runs twice "
        "daily, so anything beyond about 14 hours means a run failed. Check the "
        "Actions tab before trusting a single number on this page."
    )


(tab_squad, tab_transfers, tab_bench, tab_picks, tab_news,
 tab_model, tab_season) = st.tabs(
    ["Squad and captain", "Transfers", "Bench", "Recommendations",
     "Team news", "Model accuracy", "My season"]
)


# ------------------------------------------------------------- squad tab

with tab_squad:
    squad = load("squad")
    captain = load("captain")

    if not squad or not squad.get("squad"):
        empty("No squad recorded yet",
              "record_my_team.py writes the fifteen players you actually own. "
              "Until it runs, the engine cannot tell your players from anyone "
              "else's, and every recommendation below is generic.")
    else:
        left, right = st.columns([3, 2], gap="large")

        with left:
            st.markdown(f"##### Squad, GW{squad.get('gameweek')}")
            df = pd.DataFrame(squad["squad"])
            df["role"] = df.apply(
                lambda r: "C" if r["is_captain"] else ("V" if r["is_vice"] else ""),
                axis=1)
            df["status"] = df.apply(status_of, axis=1)

            st.dataframe(
                df[["name", "role", "position", "club", "price", "owned", "ep", "status"]]
                .rename(columns={"name": "player", "position": "pos", "price": "price",
                                 "owned": "owned %", "ep": "xPts"}),
                hide_index=True, width="stretch", height=560,
                column_config={
                    "xPts": st.column_config.NumberColumn(format="%.2f"),
                    "price": st.column_config.NumberColumn(format="%.1f"),
                    "owned %": st.column_config.NumberColumn(format="%.1f"),
                    "role": st.column_config.TextColumn(width="small"),
                    "status": st.column_config.TextColumn(
                        help="Blank means fit. A percentage is the reported "
                             "chance of playing."),
                },
            )
            st.caption(
                f"Squad value {squad.get('squad_value')}m across "
                f"{squad.get('count')} players. xPts is the two-stage model's "
                "expected points, logged before the deadline."
            )

        with right:
            if not captain or not captain.get("model_pick"):
                empty("No captaincy prediction",
                      "The model needs a published prediction for the upcoming "
                      "gameweek before it can rank captains.")
            else:
                mp = captain["model_pick"]
                ranked = captain.get("ranked", [])

                # The quiet surface. This is the pitch at kick-off: one number,
                # no decoration, and the model answering for itself.
                block(
                    '<div class="pfl-quiet">'
                    '<h3 style="margin:0 0 .75rem;font-size:.75rem;'
                    'text-transform:uppercase;letter-spacing:.1em;">'
                    f'Captain, GW{e(captain.get("gameweek", next_gw))}</h3>'
                    f'<span class="big">{e(mp["name"])}</span>'
                    f'<span class="sub">{mp["ep"]:.2f} expected points, '
                    f'doubling to {2 * mp["ep"]:.2f}</span>'
                    '</div>'
                )

                # Does the model actually have an opinion, or is it reporting
                # false precision? Measured, not asserted.
                if len(ranked) >= 2:
                    gap = ranked[0]["ep"] - ranked[1]["ep"]
                    if gap < NOISE_FLOOR:
                        block(
                            '<div class="pfl-verdict tied">'
                            f'<strong>Too close to call.</strong> {e(ranked[0]["name"])} '
                            f'leads {e(ranked[1]["name"])} by {gap:.2f} expected '
                            f'points, inside the model\'s measured noise floor of '
                            f'{NOISE_FLOOR:.2f}. Reseeding alone moves a prediction '
                            'by more than this, so the engine has no real '
                            'preference here. Use your own read.'
                            '</div>'
                        )
                    else:
                        block(
                            '<div class="pfl-verdict agree">'
                            f'<strong>A real preference.</strong> The gap of '
                            f'{gap:.2f} over {e(ranked[1]["name"])} clears the '
                            f'{NOISE_FLOOR:.2f} noise floor, so this is the model '
                            'saying something rather than rounding.'
                            '</div>'
                        )

                if captain.get("your_pick"):
                    if captain.get("agrees"):
                        block('<div class="pfl-verdict agree">'
                              f'You have <strong>{e(captain["your_pick"])}</strong> '
                              'and the model agrees.</div>')
                    else:
                        block('<div class="pfl-verdict differ">'
                              f'You have <strong>{e(captain["your_pick"])}</strong>. '
                              f'The model prefers <strong>{e(mp["name"])}</strong>. '
                              'Recorded either way, and scored afterwards.</div>')

                cdf = pd.DataFrame(ranked)
                if not cdf.empty:
                    cdf["status"] = cdf.apply(status_of, axis=1)
                    st.dataframe(
                        cdf[["name", "position", "ep", "status"]].rename(
                            columns={"name": "player", "position": "pos", "ep": "xPts"}),
                        hide_index=True, width="stretch",
                        column_config={
                            "xPts": st.column_config.NumberColumn(format="%.2f")},
                    )
                st.caption(
                    "Ranked by expected points, not by ceiling. A full season "
                    "simulation in captain.py found ceiling strategies lose: "
                    "captain points double linearly, so maximising the expected "
                    "value of twice the score is just maximising the score."
                )

        flagged = [p for p in squad["squad"] if p["flagged"]]
        if flagged:
            st.markdown("##### Doubts in your squad")
            for p in flagged:
                chance = "unknown" if p["chance"] is None else f"{p['chance']}%"
                st.error(f"{p['name']} ({p['position']}), chance of playing "
                         f"{chance}. {p['news']}")




# --------------------------------------------------------- transfers tab

with tab_transfers:
    tr = load("transfers")

    if not tr or not tr.get("available"):
        st.info(tr.get("reason", "No transfer suggestions yet.") if tr
                else "No transfer suggestions yet.")
    else:
        top = st.columns(3)
        top[0].metric("In the bank", f"{tr['bank']:.1f}m")
        top[1].metric("Free transfers", tr["free_transfers"])
        top[2].metric("Noise floor", f"{tr['noise_floor']:.2f} xP",
                      help="A gain smaller than this is inside the model's "
                           "own reseeding variance, so it is not a preference.")

        if tr["has_recommendation"]:
            st.success(f"**{tr['headline']}**")
        else:
            st.info(f"**{tr['headline']}**")

        st.caption(
            "Only players you can actually afford: the replacement must cost "
            "no more than the player leaving plus the bank. Hits are priced "
            "in, so a move gaining three points that costs four is shown as "
            "losing one."
        )

        for m in tr["moves"]:
            tag = "Free transfer" if m["free"] else f"Costs a {4} point hit"
            pen = " &nbsp;·&nbsp; **takes penalties**" if m["penalties"] else ""
            with st.expander(
                    f"{m['out']} to {m['in']}  ·  {m['net']:+.2f} net  ·  {tag}",
                    expanded=m["free"] and m["worth_it"]):
                st.markdown(
                    f"**Out** {m['out']} &nbsp; {m['out_price']:.1f}m &nbsp; "
                    f"{m['out_ep']:.2f} xP<br>"
                    f"**In** &nbsp;&nbsp;{m['in']} ({m['in_club']}) &nbsp; "
                    f"{m['in_price']:.1f}m &nbsp; {m['in_ep']:.2f} xP{pen}<br>"
                    f"**Money** {m['cost']:+.1f}m, leaving "
                    f"{m['bank_after']:.1f}m in the bank",
                    unsafe_allow_html=True)
                if not m["worth_it"]:
                    st.caption("Does not clear the noise floor, so this is "
                               "listed for completeness rather than advised.")
                for reason in m["reasons"]:
                    st.caption(reason)


# ------------------------------------------------------------- bench tab

with tab_bench:
    plan = load("bench")

    if not plan or not plan.get("available"):
        st.info(plan.get("reason", "No bench plan yet.") if plan
                else "No bench plan yet.")
    else:
        if plan.get("provisional"):
            st.warning(
                "**The order below is provisional.** These predictions were "
                "logged before the engine began storing both halves of the "
                "two-stage model, so this falls back to expected points, "
                "which is the wrong number for a bench. It corrects itself "
                "at the next deadline.")

        left, right = st.columns([3, 2])

        with left:
            st.markdown(f"#### Bench order, {plan['formation']}")
            st.caption(
                "Ordered by expected points **if the player appears**, not by "
                "expected points. Autosubs fall through anyone who did not "
                "play, so how likely they are to feature does not change the "
                "order, only how much cover you have."
            )
            for i, b in enumerate(plan["bench"]):
                slot = "GK" if b["position"] == "GKP" else str(i)
                flag = "  ·  flagged" if b["flagged"] else ""
                st.markdown(
                    f"**{slot}. {b['name']}** &nbsp; `{b['position']}` &nbsp; "
                    f"{b['if_played']:.2f} if he plays &nbsp;·&nbsp; "
                    f"{b['p_play'] * 100:.0f}% to feature{flag}",
                    unsafe_allow_html=True)

            if plan.get("autosub_value") is not None:
                st.metric("Expected points from autosubs",
                          f"{plan['autosub_value']:.2f}")

        with right:
            bb = plan["bench_boost"]
            st.markdown("#### Bench Boost")
            st.metric("Worth this week", f"{bb['value_now']:.1f} pts",
                      help="Every bench player scores with the chip active.")
            verdict = bb["verdict"]
            if verdict == "play it":
                st.success(f"**{verdict.upper()}**")
            elif verdict.startswith("hold"):
                st.info(f"**{verdict.upper()}**")
            else:
                st.warning(f"**{verdict.upper()}**")
            for line in bb["reasoning"]:
                st.caption(line)

        for note in plan.get("notes", []):
            st.caption(note)

        st.markdown("#### Starting XI")
        st.dataframe(
            [{"Player": x["name"], "Pos": x["position"],
              "xP": round(x["ep"], 2),
              "If he plays": round(x["if_played"], 2),
              "Plays": f"{x['p_play'] * 100:.0f}%"}
             for x in plan["starting_xi"]],
            hide_index=True, use_container_width=True)


# ------------------------------------------------------ recommendations

with tab_picks:
    recs = load("recommendations")
    if not recs or not recs.get("picks"):
        empty("No recommendations published yet",
              "recommend.py ranks every available player by expected points "
              "and writes the top of that list here before each deadline.")
    else:
        st.markdown(f"##### Ranked by expected points, GW{recs.get('gameweek')}")

        df = pd.DataFrame(recs["picks"])
        cols = st.columns(4)
        pos_filter = cols[0].multiselect(
            "Position", ["GKP", "DEF", "MID", "FWD"],
            default=["GKP", "DEF", "MID", "FWD"])
        max_price = cols[1].slider(
            "Max price", 3.5, 16.0,
            float(df["price"].max()) if df["price"].notna().any() else 16.0, 0.1)
        hide_flagged = cols[2].checkbox("Hide doubtful players", value=True)
        mine_only = cols[3].checkbox("Only players I own", value=False)

        f = df[df["position"].isin(pos_filter) & (df["price"] <= max_price)]
        if hide_flagged:
            f = f[f["news"].isna() | (f["news"] == "")]
        if mine_only:
            f = f[f["owned_by_me"]]

        f = f.copy()
        f["mine"] = f["owned_by_me"].map({True: "yours", False: ""})
        f["value"] = (f["ep"] / f["price"]).round(3)

        st.dataframe(
            f[["name", "mine", "position", "club", "price", "owned", "ep", "value", "news"]]
            .rename(columns={"name": "player", "position": "pos",
                             "owned": "owned %", "ep": "xPts",
                             "value": "xPts per m", "news": "status"}),
            hide_index=True, width="stretch", height=520,
            column_config={
                "xPts": st.column_config.NumberColumn(format="%.2f"),
                "price": st.column_config.NumberColumn(format="%.1f"),
                "owned %": st.column_config.NumberColumn(format="%.1f"),
                "xPts per m": st.column_config.NumberColumn(format="%.3f"),
                "mine": st.column_config.TextColumn(width="small"),
            },
        )
        st.caption(
            "Predictions were written to the database before the deadline and "
            "are never edited afterwards. Two players within "
            f"{NOISE_FLOOR:.2f} expected points of each other are tied, whatever "
            "order this table happens to show them in."
        )


# -------------------------------------------------------------- news tab

with tab_news:
    alerts = load("alerts")
    if not alerts or not alerts.get("available"):
        empty("Not enough snapshots to compare",
              "Team news is found by diffing two snapshots of the FPL API. "
              "The pipeline needs at least two before it can tell you what "
              "changed.")
    else:
        st.caption(
            f"Comparing snapshot {alerts.get('prev_snapshot')} to "
            f"{alerts.get('latest_snapshot')}, the last "
            f"{alerts.get('window_hours')} hours."
        )

        urgent = alerts.get("urgent", [])
        if urgent:
            st.markdown(f"##### {len(urgent)} change(s) affecting you")
            for a in urgent:
                st.warning(f"{a['severity'].upper()}, {a['name']} "
                           f"({a['position']}, {a['owned']:.1f}% owned). "
                           f"{a['message']}")
        else:
            st.success("Nothing changed for any player you own or watch.")

        rows = alerts.get("alerts", [])
        if rows:
            st.markdown(f"##### All availability changes ({len(rows)})")
            adf = pd.DataFrame(rows)
            adf["tag"] = adf["tag"].replace("-", "")
            st.dataframe(
                adf[["tag", "severity", "name", "position", "owned", "message"]]
                .rename(columns={"name": "player", "position": "pos",
                                 "owned": "owned %", "message": "what changed"}),
                hide_index=True, width="stretch", height=420,
                column_config={"owned %": st.column_config.NumberColumn(format="%.1f")},
            )

        moves = alerts.get("price_moves", [])
        if moves:
            st.markdown("##### Biggest price moves")
            mdf = pd.DataFrame(moves)
            mdf["change"] = (mdf["new"] - mdf["old"]).round(1)
            st.dataframe(
                mdf[["name", "old", "new", "change", "owned"]]
                .rename(columns={"name": "player", "owned": "owned %"}),
                hide_index=True, width="stretch",
                column_config={
                    "old": st.column_config.NumberColumn(format="%.1f"),
                    "new": st.column_config.NumberColumn(format="%.1f"),
                    "change": st.column_config.NumberColumn(format="%+.1f"),
                    "owned %": st.column_config.NumberColumn(format="%.1f"),
                },
            )


# ------------------------------------------------------------- model tab

with tab_model:
    acc = load("accuracy")

    # A model that changed mid-season is two models on one chart unless the
    # break is drawn. Shown above the numbers, not under them.
    for change in (acc or {}).get("model_changes", []):
        st.warning(
            f"**GW{change['gameweek']}: {change['change']}** "
            f"{change['detail']}")
    st.markdown("##### Is the model actually any good?")
    st.markdown(
        "Predictions are written **before** each deadline and never edited "
        "afterwards, so these errors cannot be flattered in hindsight. Lower is "
        "better. **The high return column matters more than overall MAE**, "
        "because being reliably right that a bench player scores one point is "
        "worth nothing. The players who decide a gameweek live in that column."
    )

    if not acc or not acc.get("scored"):
        pending = (acc or {}).get("pending", [])
        empty(
            "Nothing scored yet",
            "A gameweek needs both a logged prediction and a finished result."
            + (f" Waiting on GW{', GW'.join(str(g) for g in pending)}."
               if pending else "")
        )
    else:
        sdf = pd.DataFrame(acc["scored"])
        st.dataframe(
            sdf[["gameweek", "model", "n", "mae", "mae_high_return"]].rename(
                columns={"gameweek": "GW", "n": "players", "mae": "MAE",
                         "mae_high_return": "MAE, high return"}),
            hide_index=True, width="stretch",
            column_config={
                "MAE": st.column_config.NumberColumn(format="%.3f"),
                "MAE, high return": st.column_config.NumberColumn(format="%.3f"),
            },
        )

        # One gameweek is a data point, not a trend. Say so rather than drawing
        # a line chart through a single value and implying otherwise.
        if sdf["gameweek"].nunique() > 1:
            # The model under test is drawn in the kit colour and the baselines
            # it has to beat are drawn behind it, so which line matters is
            # visible before reading the legend. Ordering is handled in line().
            line(sdf, "gameweek", "mae", colour_by="model",
                 y_title="Mean absolute error", height=300)
        else:
            gw = int(sdf["gameweek"].iloc[0])
            st.caption(
                f"One scored gameweek so far, GW{gw}. A chart of a single point "
                "would suggest a trend that does not exist yet, so there is not "
                "one. It appears from the second scored gameweek."
            )

    with st.expander("How the model works, and what was rejected"):
        st.markdown("""
**Two-stage prediction.** Expected points are the probability of playing sixty
minutes, multiplied by expected points given that they did. Predicting points
directly makes the model hedge every rotation risk into every score. Splitting
the question keeps the availability problem and the performance problem apart.

**Validation is chronological, never random k-fold.** Three seasons train, the
following season tests. Rolling features are shifted with `groupby(...).shift(1)`
before any window is applied, so a row can only ever see matches that finished
before it.

**Leakage was tested, not assumed.** A negative control, the same pipeline with
the target shuffled, scored 1.551 MAE against 1.537 for simply predicting the
mean. A leaking pipeline would have beaten the mean comfortably.

**The model has a measured noise floor, and it is not small.** Changing nothing
but the random seed moves a pickable player's prediction by 0.169 expected
points on average, and by as much as 1.56. Any gap narrower than that is noise
wearing two decimal places. This page marks those as tied instead of ranking
them, which is the single most useful thing it does.

**A rejected feature is kept in the repo.** `opponent.py` adds opponent
strength. The underlying effect is real and large, 2.73 points against the
toughest fifth of opponents versus 4.33 against the leakiest, a swing of 1.59.
It still did not improve ranking: a paired bootstrap over 38 gameweeks gave
-0.125, 95% CI [-0.287, +0.036], probability better 0.06. It is applied as a
human overlay on captaincy instead, and the negative result is documented
rather than deleted.

**Position specific models were tested and rejected** at -0.229 points per pick,
the worst of anything tried. That is the second idea borrowed from published
prior art to fail on this dataset, after opponent features. Both are kept as
documented negative results, because a project that only records its wins is
not measuring anything.
        """)


# ------------------------------------------------------------ season tab

with tab_season:
    season = load("season")
    if not season or not season.get("gameweeks"):
        empty("No gameweeks recorded yet",
              "This is the benchmark the engine has to beat, so it is recorded "
              "by hand rather than inferred.")
    else:
        a, b, c = st.columns(3)
        a.metric("Total points", season.get("total_points", 0))
        b.metric("Gameweeks played", season.get("played", 0))
        c.metric("Average", f"{season['average']:.1f}" if season.get("average") else "none yet")

        sdf = pd.DataFrame(season["gameweeks"])
        st.dataframe(
            sdf[["gameweek", "points", "transfers", "formation",
                 "captain", "captain_points", "decided_by"]].rename(
                columns={"gameweek": "GW", "captain_points": "captain pts",
                         "decided_by": "decided by"}),
            hide_index=True, width="stretch",
        )

        # A recorded but unplayed gameweek carries points: null. pandas reads
        # that as NaN rather than making the column object dtype, so the chart
        # was never actually broken, but a NaN would still draw a gap. Dropped
        # explicitly so the line means "gameweeks with a score" and nothing else.
        played = sdf[sdf["played"] == True].copy()
        played["points"] = pd.to_numeric(played["points"], errors="coerce")
        played = played.dropna(subset=["points"])
        if len(played) > 1:
            line(played, "gameweek", "points", y_title="Points")

        st.caption(
            "This is the benchmark. GW1 to GW3 were picked on instinct with no "
            "model involved, so anything the engine produces has to beat these, "
            "not merely beat a naive baseline. The decided by column records "
            "who made each call, which is the only way the argument between "
            "instinct and model can ever be settled."
        )

        # The warm surface. Everything above this point is the pitch at
        # kick-off. This is the group photo afterwards, and it is the only
        # part of the page that is about a person rather than a number.
        shots = sorted(PHOTOS.glob("*.jpg")) + sorted(PHOTOS.glob("*.webp"))
        if shots:
            st.markdown("##### Why any of this matters")

            # st.image, not an <img> inside st.markdown. The first version
            # inlined each photo as a base64 data URI, on the assumption that
            # Streamlit serves no static files. Streamlit dropped the tag
            # silently: the heading rendered, the pictures did not, and nothing
            # was logged anywhere. st.image is the supported path and is also
            # the better one, because it serves each file from Streamlit's own
            # media endpoint as a real cacheable HTTP resource instead of
            # pushing the bytes through the HTML on every single page load.
            # Always four columns, however many photos there are. Sized to the
            # count instead, a single picture spans the full width and the 180px
            # crop takes a band out of the middle of it, which on a portrait
            # shot means the subject's head is above the frame.
            for col, p in zip(st.columns(4), shots[:4]):
                col.image(str(p), caption=caption(p), use_container_width=True)

            st.caption(
                "Katowice, playing for Zimbabwe. The engine exists because "
                "somebody who plays the game wanted to know whether a model "
                "could out-argue him about it."
            )


# ---------------------------------------------------------------- footer

block(
    '<div class="pfl-foot">'
    f'Snapshot {e(meta.get("snapshot_id"))}, generated '
    f'{e(meta.get("generated_at"))} by <code>{e(meta.get("published_by", "unknown"))}</code>. '
    f'{meta["row_counts"]["history"]:,} rows of historical player data behind '
    'every number above.<br>'
    'Built by GitHub Actions on a schedule, rendered by Streamlit, computed '
    'nowhere near your browser. Predictions are logged before each deadline and '
    'never edited afterwards.'
    '</div>'
)
