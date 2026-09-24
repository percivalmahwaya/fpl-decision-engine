/* ==========================================================================
   Progeny FPL Engine, the static front end.
   ==========================================================================

   THIS FILE COMPUTES NOTHING ABOUT FOOTBALL. It fetches the JSON that GitHub
   Actions wrote before the deadline and draws it. The one exception is the
   captaincy gap against the noise floor, which is a comparison rather than a
   prediction and is done here for the same reason app.py does it there: the
   number that matters is the DIFFERENCE between two published numbers, and
   publishing a third number derived from them invites the two to drift apart.

   NO FRAMEWORK AND NO CHART LIBRARY. The charts are two line charts of five
   points each. The smallest respectable charting library is around 200 KB,
   which is seven times the weight of every number on this page put together.

   THE NOISE FLOOR IS NOT DEFINED HERE.
   It arrives in transfers.json, measured by the pipeline. A copy of 0.17
   typed into this file would be a second source of truth that stays at 0.17
   forever after somebody remeasures it, and the whole argument of this
   project is that a number nobody recomputed is a number nobody should
   trust. If the pipeline has not published one, the page says so instead of
   guessing.
   ========================================================================== */

'use strict';

var DATA = 'data/';
var NOISE_FLOOR = null;          // set from transfers.json, never assumed

/* Chart series colours, in order, matching the CSS tokens. Charts are SVG
   drawn by this file, so no stylesheet can reach the stroke of a path that
   does not exist until runtime. Keep in step with :root in pfl.css. */
var SERIES = ['#1f7a3d', '#d8a222', '#6a6a5f', '#b3241a'];

/* ------------------------------------------------------------- utilities */

function el(tag, attrs, kids) {
  var node = document.createElement(tag);
  if (attrs) {
    Object.keys(attrs).forEach(function (k) {
      if (k === 'class') node.className = attrs[k];
      else if (k === 'text') node.textContent = attrs[k];
      else if (k === 'html') node.innerHTML = attrs[k];
      else if (attrs[k] !== null && attrs[k] !== undefined) node.setAttribute(k, attrs[k]);
    });
  }
  (kids || []).forEach(function (c) {
    if (c === null || c === undefined) return;
    node.appendChild(typeof c === 'string' ? document.createTextNode(c) : c);
  });
  return node;
}

/* Everything that reaches innerHTML goes through here first.

   The data is Percival's own and comes from the FPL API, so this is not
   defending against an attacker so much as against a player whose name
   contains an ampersand. But a page that escapes only where it expects
   trouble is a page that escapes nowhere, so it is done everywhere and
   without exception. */
function esc(v) {
  return String(v === null || v === undefined ? '' : v)
    .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;');
}

function num(v, dp) {
  if (v === null || v === undefined || v === '' || isNaN(v)) return '';
  return Number(v).toFixed(dp === undefined ? 2 : dp);
}

function signed(v, dp) {
  if (v === null || v === undefined || isNaN(v)) return '';
  var s = Number(v).toFixed(dp === undefined ? 2 : dp);
  return Number(v) > 0 ? '+' + s : s;
}

function pct(v) {
  if (v === null || v === undefined || isNaN(v)) return '';
  return Math.round(Number(v) * 100) + '%';
}

/* "2 days", "14 hours", "40 minutes". Matches humanise() in app.py, because
   two front ends describing the same instant differently is how somebody
   ends up mistrusting both. */
function humanise(hours) {
  if (hours === null || hours === undefined || isNaN(hours)) return 'unknown';
  var h = Math.abs(hours);
  var out;
  if (h < 1) out = Math.round(h * 60) + ' minutes';
  else if (h < 48) out = Math.round(h) + ' hour' + (Math.round(h) === 1 ? '' : 's');
  else out = Math.round(h / 24) + ' days';
  return hours < 0 ? out + ' ago' : out;
}

function parseIso(s) {
  if (!s) return null;
  var d = new Date(s);
  return isNaN(d.getTime()) ? null : d;
}

/* A word, never a symbol. An emoji renders as an empty box on the older
   Android handsets this is actually opened on, reads as nothing useful in a
   screen reader, and is a house rule violation besides. */
function statusOf(p) {
  if (!p || !p.flagged) return '';
  if (p.chance === 0) return 'out';
  if (p.chance === null || p.chance === undefined) return 'doubt';
  return Math.round(p.chance) + '%';
}

function note(kind, html) {
  return el('div', { class: 'note note-' + kind, html: html });
}

function caption(text) {
  return el('p', { class: 'caption', html: text });
}

function section(text) {
  return el('h2', { class: 'sect', text: text });
}

function empty(title, body) {
  return el('div', { class: 'pfl-empty' }, [
    el('h3', { text: title }), el('p', { text: body })
  ]);
}

/* ---------------------------------------------------------------- table */

/* columns: [{key, label, dp, cls, fmt}] */
function table(rows, columns, rowClass) {
  var head = el('tr', null, columns.map(function (c) {
    return el('th', { class: c.num ? 'num' : null, text: c.label });
  }));

  var body = rows.map(function (r) {
    return el('tr', { class: rowClass ? rowClass(r) : null }, columns.map(function (c) {
      var v = c.fmt ? c.fmt(r) : r[c.key];
      if (c.dp !== undefined && v !== '' && v !== null && v !== undefined) v = num(v, c.dp);
      return el('td', { class: (c.num ? 'num ' : '') + (c.cls || ''), text: v === null || v === undefined ? '' : String(v) });
    }));
  });

  return el('div', { class: 'tablewrap' }, [
    el('table', { class: 'data' }, [
      el('thead', null, [head]),
      el('tbody', null, body)
    ])
  ]);
}

function metrics(items) {
  return el('div', { class: 'metrics' }, items.map(function (m) {
    return el('div', { class: 'metric' }, [
      el('span', { class: 'k', text: m.k }),
      el('span', { class: 'v', text: m.v })
    ]);
  }));
}

/* ---------------------------------------------------------------- chart */

/* A line chart, in SVG, from scratch.

   viewBox with no width/height attribute, so the chart scales with its
   container and needs no resize listener. A chart that only redraws on load
   is a chart that is the wrong size on a phone rotated after the page
   opened. */
function lineChart(groups, opts) {
  opts = opts || {};
  var W = 720, H = opts.height || 280;
  var m = { t: 14, r: 14, b: 30, l: 46 };
  var iw = W - m.l - m.r, ih = H - m.t - m.b;

  var xs = [], ys = [];
  groups.forEach(function (g) {
    g.points.forEach(function (p) { xs.push(p.x); ys.push(p.y); });
  });
  if (!xs.length) return el('div');

  var x0 = Math.min.apply(null, xs), x1 = Math.max.apply(null, xs);
  var y0 = Math.min.apply(null, ys), y1 = Math.max.apply(null, ys);

  // Pad the value axis so the top line is not welded to the frame. Never
  // force zero: these are mean absolute errors around 1.1, and a zero
  // baseline would squash every real difference into one flat band.
  var pad = (y1 - y0) * 0.15 || (Math.abs(y1) * 0.1) || 1;
  y0 -= pad; y1 += pad;
  if (x1 === x0) { x0 -= 0.5; x1 += 0.5; }

  function sx(v) { return m.l + (v - x0) / (x1 - x0) * iw; }
  function sy(v) { return m.t + ih - (v - y0) / (y1 - y0) * ih; }

  var svgns = 'http://www.w3.org/2000/svg';
  function s(tag, attrs) {
    var n = document.createElementNS(svgns, tag);
    Object.keys(attrs || {}).forEach(function (k) { n.setAttribute(k, attrs[k]); });
    return n;
  }

  var svg = s('svg', {
    class: 'chart', viewBox: '0 0 ' + W + ' ' + H,
    preserveAspectRatio: 'xMidYMid meet', role: 'img',
    'aria-label': opts.label || 'chart'
  });

  // Value gridlines and labels, five of them.
  for (var i = 0; i <= 4; i++) {
    var v = y0 + (y1 - y0) * i / 4, y = sy(v);
    svg.appendChild(s('line', { class: 'grid', x1: m.l, x2: W - m.r, y1: y, y2: y }));
    var t = s('text', { class: 'tick', x: m.l - 6, y: y + 3, 'text-anchor': 'end' });
    t.textContent = v.toFixed(opts.ydp === undefined ? 2 : opts.ydp);
    svg.appendChild(t);
  }

  // One label per whole step on the gameweek axis, never a fractional one.
  // st.line_chart produced 1.0, 1.5, 2.0 and there is no half gameweek.
  var seen = {};
  xs.slice().sort(function (a, b) { return a - b; }).forEach(function (v) {
    if (seen[v]) return;
    seen[v] = 1;
    var t = s('text', { class: 'tick', x: sx(v), y: H - 10, 'text-anchor': 'middle' });
    t.textContent = opts.xprefix ? opts.xprefix + v : String(v);
    svg.appendChild(t);
  });

  svg.appendChild(s('line', { class: 'axis', x1: m.l, x2: W - m.r, y1: m.t + ih, y2: m.t + ih }));

  groups.forEach(function (g, gi) {
    var colour = SERIES[gi % SERIES.length];
    var pts = g.points.slice().sort(function (a, b) { return a.x - b.x; });
    var d = pts.map(function (p, i) {
      return (i ? 'L' : 'M') + sx(p.x).toFixed(1) + ' ' + sy(p.y).toFixed(1);
    }).join(' ');
    var path = s('path', { class: 'series', d: d, stroke: colour });
    svg.appendChild(path);
    pts.forEach(function (p) {
      svg.appendChild(s('circle', { class: 'dot', cx: sx(p.x), cy: sy(p.y), r: 3, fill: colour }));
    });
  });

  var wrap = el('div');
  wrap.appendChild(svg);
  if (groups.length > 1 || opts.alwaysLegend) {
    wrap.appendChild(el('div', { class: 'legend' }, groups.map(function (g, gi) {
      var i = el('i');
      i.style.background = SERIES[gi % SERIES.length];
      return el('span', null, [i, g.name]);
    })));
  }
  return wrap;
}

/* =========================================================== the panels */

function squadPanel(d) {
  var box = el('div');
  var squad = d.squad, captain = d.captain;

  if (!squad || !squad.squad || !squad.squad.length) {
    box.appendChild(empty('No squad recorded yet',
      'record_my_team.py writes the fifteen players you actually own. Until ' +
      'it runs, the engine cannot tell your players from anyone else’s, ' +
      'and every recommendation below is generic.'));
    return box;
  }

  box.appendChild(section('Squad, GW' + (squad.gameweek || '')));
  box.appendChild(table(squad.squad, [
    { key: 'name', label: 'player' },
    { label: 'role', fmt: function (r) { return r.is_captain ? 'C' : (r.is_vice ? 'V' : ''); } },
    { key: 'position', label: 'pos' },
    { key: 'club', label: 'club' },
    { key: 'price', label: 'price', dp: 1, num: true },
    { key: 'owned', label: 'owned %', dp: 1, num: true },
    { key: 'ep', label: 'xPts', dp: 2, num: true },
    { label: 'status', fmt: statusOf }
  ]));
  box.appendChild(caption(
    'Squad value ' + esc(squad.squad_value) + 'm across ' + esc(squad.count) +
    ' players. xPts is the two-stage model’s expected points, logged ' +
    'before the deadline.'));

  if (captain && captain.model_pick) {
    var mp = captain.model_pick;
    box.appendChild(section('Captain, GW' + (captain.gameweek || '')));
    box.appendChild(el('div', {
      class: 'pfl-quiet',
      html: '<span class="big">' + esc(mp.name) + '</span>' +
            '<span class="sub">' + num(mp.ep) + ' expected points, doubling to ' +
            num(2 * mp.ep) + '</span>'
    }));

    var ranked = captain.ranked || [];
    if (ranked.length >= 2 && NOISE_FLOOR !== null) {
      var gap = ranked[0].ep - ranked[1].ep;
      if (gap < NOISE_FLOOR) {
        box.appendChild(el('div', {
          class: 'pfl-verdict tied',
          html: '<strong>Too close to call.</strong> ' + esc(ranked[0].name) +
                ' leads ' + esc(ranked[1].name) + ' by ' + num(gap) +
                ' expected points, inside the model’s measured noise floor of ' +
                num(NOISE_FLOOR) + '. Reseeding alone moves a prediction by more ' +
                'than this, so the engine has no real preference here. Use your own read.'
        }));
      } else {
        box.appendChild(el('div', {
          class: 'pfl-verdict agree',
          html: '<strong>A real preference.</strong> The gap of ' + num(gap) +
                ' over ' + esc(ranked[1].name) + ' clears the ' + num(NOISE_FLOOR) +
                ' noise floor, so this is the model saying something rather than rounding.'
        }));
      }
    }

    if (captain.your_pick) {
      box.appendChild(el('div', {
        class: 'pfl-verdict ' + (captain.agrees ? 'agree' : 'differ'),
        html: captain.agrees
          ? 'You have <strong>' + esc(captain.your_pick) + '</strong> and the model agrees.'
          : 'You have <strong>' + esc(captain.your_pick) + '</strong>. The model prefers ' +
            '<strong>' + esc(mp.name) + '</strong>. Recorded either way, and scored afterwards.'
      }));
    }

    if (ranked.length) {
      box.appendChild(table(ranked, [
        { key: 'name', label: 'player' },
        { key: 'position', label: 'pos' },
        { key: 'ep', label: 'xPts', dp: 2, num: true },
        { label: 'status', fmt: statusOf }
      ]));
    }
    box.appendChild(caption(
      'Ranked by expected points, not by ceiling. A full season simulation in ' +
      'captain.py found ceiling strategies lose: captain points double linearly, ' +
      'so maximising the expected value of twice the score is just maximising the score.'));
  }

  var flagged = squad.squad.filter(function (p) { return p.flagged; });
  if (flagged.length) {
    box.appendChild(section('Doubts in your squad'));
    flagged.forEach(function (p) {
      box.appendChild(note('alarm',
        esc(p.name) + ' (' + esc(p.position) + '), chance of playing ' +
        (p.chance === null || p.chance === undefined ? 'unknown' : esc(p.chance) + '%') +
        '. ' + esc(p.news || '')));
    });
  }
  return box;
}

function transfersPanel(d) {
  var box = el('div'), tr = d.transfers;
  if (!tr || !tr.available) {
    box.appendChild(note('', esc((tr && tr.reason) || 'No transfer suggestions yet.')));
    return box;
  }

  box.appendChild(metrics([
    { k: 'In the bank', v: num(tr.bank, 1) + 'm' },
    { k: 'Free transfers', v: String(tr.free_transfers) },
    { k: 'Noise floor', v: num(tr.noise_floor) + ' xP' }
  ]));

  box.appendChild(note(tr.has_recommendation ? 'good' : '', '<strong>' + esc(tr.headline) + '</strong>'));
  box.appendChild(caption(
    'Only players you can actually afford: the replacement must cost no more ' +
    'than the player leaving plus the bank. Hits are priced in, so a move ' +
    'gaining three points that costs four is shown as losing one.'));

  (tr.moves || []).forEach(function (m) {
    var body = el('div', { class: 'body' });
    body.appendChild(el('p', {
      html: '<strong>Out</strong> ' + esc(m.out) + ' &nbsp; ' + num(m.out_price, 1) +
            'm &nbsp; ' + num(m.out_ep) + ' xP<br>' +
            '<strong>In</strong> &nbsp; ' + esc(m['in']) + ' (' + esc(m.in_club) + ') &nbsp; ' +
            num(m.in_price, 1) + 'm &nbsp; ' + num(m.in_ep) + ' xP' +
            (m.penalties ? ' &nbsp;·&nbsp; <strong>takes penalties</strong>' : '') + '<br>' +
            '<strong>Money</strong> ' + signed(m.cost, 1) + 'm, leaving ' +
            num(m.bank_after, 1) + 'm in the bank'
    }));
    if (!m.worth_it) {
      body.appendChild(caption(
        'Does not clear the noise floor, so this is listed for completeness ' +
        'rather than advised.'));
    }
    (m.reasons || []).forEach(function (r) { body.appendChild(caption(esc(r))); });

    var open = m.free && m.worth_it;
    var det = el('details', open ? { class: 'more', open: 'open' } : { class: 'more' }, [
      el('summary', {
        text: m.out + ' to ' + m['in'] + '  ·  ' + signed(m.net) + ' net  ·  ' +
              (m.free ? 'Free transfer' : 'Costs a 4 point hit')
      }),
      body
    ]);
    box.appendChild(det);
  });
  return box;
}

function benchPanel(d) {
  var box = el('div'), plan = d.bench;
  if (!plan || !plan.available) {
    box.appendChild(note('', esc((plan && plan.reason) || 'No bench plan yet.')));
    return box;
  }

  if (plan.provisional) {
    box.appendChild(note('warn',
      '<strong>The order below is provisional.</strong> These predictions were ' +
      'logged before the engine began storing both halves of the two-stage ' +
      'model, so this falls back to expected points, which is the wrong number ' +
      'for a bench. It corrects itself at the next deadline.'));
  }

  box.appendChild(section('Bench order, ' + esc(plan.formation)));
  box.appendChild(caption(
    'Ordered by expected points <strong>if the player appears</strong>, not by ' +
    'expected points. Autosubs fall through anyone who did not play, so how ' +
    'likely they are to feature does not change the order, only how much cover ' +
    'you have.'));

  box.appendChild(table(plan.bench, [
    {
      label: 'slot', fmt: function (b) {
        return b.position === 'GKP' ? 'GK' : String(plan.bench.indexOf(b));
      }
    },
    { key: 'name', label: 'player' },
    { key: 'position', label: 'pos' },
    { key: 'if_played', label: 'if he plays', dp: 2, num: true },
    { label: 'to feature', num: true, fmt: function (b) { return pct(b.p_play); } },
    { label: '', fmt: function (b) { return b.flagged ? 'flagged' : ''; } }
  ]));

  var bb = plan.bench_boost || {};
  box.appendChild(metrics([
    {
      k: 'Expected points from autosubs',
      v: plan.autosub_value === null || plan.autosub_value === undefined
        ? 'not yet' : num(plan.autosub_value)
    },
    { k: 'Bench Boost worth this week', v: num(bb.value_now, 1) + ' pts' }
  ]));

  if (bb.verdict) {
    var kind = bb.verdict === 'play it' ? 'good'
      : (bb.verdict.indexOf('hold') === 0 ? '' : 'warn');
    box.appendChild(note(kind, '<strong>' + esc(bb.verdict.toUpperCase()) + '</strong>'));
  }
  (bb.reasoning || []).forEach(function (r) { box.appendChild(caption(esc(r))); });
  (plan.notes || []).forEach(function (n) { box.appendChild(caption(esc(n))); });

  box.appendChild(section('Starting XI'));
  box.appendChild(table(plan.starting_xi, [
    { key: 'name', label: 'player' },
    { key: 'position', label: 'pos' },
    { key: 'ep', label: 'xP', dp: 2, num: true },
    { key: 'if_played', label: 'if he plays', dp: 2, num: true },
    { label: 'plays', num: true, fmt: function (x) { return pct(x.p_play); } }
  ]));
  return box;
}

function picksPanel(d) {
  var box = el('div'), recs = d.recommendations;
  if (!recs || !recs.picks || !recs.picks.length) {
    box.appendChild(empty('No recommendations published yet',
      'recommend.py ranks every available player by expected points and writes ' +
      'the top of that list here before each deadline.'));
    return box;
  }

  box.appendChild(section('Ranked by expected points, GW' + (recs.gameweek || '')));

  var state = {
    positions: { GKP: true, DEF: true, MID: true, FWD: true },
    maxPrice: Math.max.apply(null, recs.picks.map(function (p) { return p.price || 0; })),
    hideFlagged: true,
    mineOnly: false
  };

  var posWrap = el('div', { class: 'posgroup' });
  ['GKP', 'DEF', 'MID', 'FWD'].forEach(function (p) {
    var b = el('button', { type: 'button', 'aria-pressed': 'true', text: p });
    b.addEventListener('click', function () {
      state.positions[p] = !state.positions[p];
      b.setAttribute('aria-pressed', state.positions[p] ? 'true' : 'false');
      draw();
    });
    posWrap.appendChild(b);
  });

  var priceOut = el('span', { class: 'v', text: num(state.maxPrice, 1) + 'm' });
  var price = el('input', {
    type: 'range', min: '3.5', max: String(state.maxPrice),
    step: '0.1', value: String(state.maxPrice), id: 'maxprice'
  });
  price.addEventListener('input', function () {
    state.maxPrice = parseFloat(price.value);
    priceOut.textContent = num(state.maxPrice, 1) + 'm';
    draw();
  });

  var flagBox = el('input', { type: 'checkbox', id: 'hideflag', checked: 'checked' });
  flagBox.addEventListener('change', function () { state.hideFlagged = flagBox.checked; draw(); });
  var mineBox = el('input', { type: 'checkbox', id: 'mineonly' });
  mineBox.addEventListener('change', function () { state.mineOnly = mineBox.checked; draw(); });

  box.appendChild(el('div', { class: 'filters' }, [
    el('div', null, [el('label', { text: 'Position' }), posWrap]),
    el('div', null, [el('label', { for: 'maxprice', text: 'Max price' }), price, priceOut]),
    el('div', { class: 'check' }, [flagBox, el('label', { for: 'hideflag', text: 'Hide doubtful players' })]),
    el('div', { class: 'check' }, [mineBox, el('label', { for: 'mineonly', text: 'Only players I own' })])
  ]));

  var host = el('div');
  box.appendChild(host);
  box.appendChild(caption(
    'Predictions were written to the database before the deadline and are never ' +
    'edited afterwards. Two players within ' +
    (NOISE_FLOOR === null ? 'the noise floor' : num(NOISE_FLOOR)) +
    ' expected points of each other are tied, whatever order this table happens ' +
    'to show them in.'));

  function draw() {
    var rows = recs.picks.filter(function (p) {
      if (!state.positions[p.position]) return false;
      if (p.price > state.maxPrice) return false;
      if (state.hideFlagged && p.news) return false;
      if (state.mineOnly && !p.owned_by_me) return false;
      return true;
    });
    host.innerHTML = '';
    if (!rows.length) {
      host.appendChild(caption('No player matches those filters.'));
      return;
    }
    host.appendChild(table(rows, [
      { key: 'name', label: 'player' },
      { label: 'mine', fmt: function (r) { return r.owned_by_me ? 'yours' : ''; } },
      { key: 'position', label: 'pos' },
      { key: 'club', label: 'club' },
      { key: 'price', label: 'price', dp: 1, num: true },
      { key: 'owned', label: 'owned %', dp: 1, num: true },
      { key: 'ep', label: 'xPts', dp: 2, num: true },
      { label: 'xPts per m', num: true, dp: 3, fmt: function (r) { return r.price ? r.ep / r.price : null; } },
      { key: 'news', label: 'status' }
    ], function (r) { return r.owned_by_me ? 'rowmine' : null; }));
  }
  draw();
  return box;
}

function newsPanel(d) {
  var box = el('div'), a = d.alerts;
  if (!a || !a.available) {
    box.appendChild(empty('Not enough snapshots to compare',
      'Team news is found by diffing two snapshots of the FPL API. The pipeline ' +
      'needs at least two before it can tell you what changed.'));
    return box;
  }

  box.appendChild(caption(
    'Comparing snapshot ' + esc(a.prev_snapshot) + ' to ' + esc(a.latest_snapshot) +
    ', the last ' + esc(a.window_hours) + ' hours.'));

  var urgent = a.urgent || [];
  if (urgent.length) {
    box.appendChild(section(urgent.length + ' change(s) affecting you'));
    urgent.forEach(function (u) {
      box.appendChild(note('warn',
        esc(String(u.severity).toUpperCase()) + ', ' + esc(u.name) + ' (' +
        esc(u.position) + ', ' + num(u.owned, 1) + '% owned). ' + esc(u.message)));
    });
  } else {
    box.appendChild(note('good', 'Nothing changed for any player you own or watch.'));
  }

  if ((a.alerts || []).length) {
    box.appendChild(section('All availability changes (' + a.alerts.length + ')'));
    box.appendChild(table(a.alerts, [
      { label: 'tag', fmt: function (r) { return r.tag === '-' ? '' : r.tag; } },
      { key: 'severity', label: 'severity' },
      { key: 'name', label: 'player' },
      { key: 'position', label: 'pos' },
      { key: 'owned', label: 'owned %', dp: 1, num: true },
      { key: 'message', label: 'what changed' }
    ]));
  }

  if ((a.price_moves || []).length) {
    box.appendChild(section('Biggest price moves'));
    box.appendChild(table(a.price_moves, [
      { key: 'name', label: 'player' },
      { key: 'old', label: 'old', dp: 1, num: true },
      { key: 'new', label: 'new', dp: 1, num: true },
      { label: 'change', num: true, fmt: function (r) { return signed(r['new'] - r.old, 1); } },
      { key: 'owned', label: 'owned %', dp: 1, num: true }
    ]));
  }
  return box;
}

function modelPanel(d) {
  var box = el('div'), acc = d.accuracy || {};

  /* A model that changed mid-season is two models on one chart unless the
     break is drawn. Above the numbers, never under them. */
  (acc.model_changes || []).forEach(function (c) {
    box.appendChild(note('warn',
      '<strong>GW' + esc(c.gameweek) + ': ' + esc(c.change) + '</strong> ' + esc(c.detail)));
  });

  box.appendChild(section('Is the model actually any good?'));
  box.appendChild(el('p', {
    html: 'Predictions are written <strong>before</strong> each deadline and never ' +
          'edited afterwards, so these errors cannot be flattered in hindsight. ' +
          'Lower is better. <strong>The high return column matters more than ' +
          'overall MAE</strong>, because being reliably right that a bench player ' +
          'scores one point is worth nothing. The players who decide a gameweek ' +
          'live in that column.'
  }));

  var scored = acc.scored || [];
  if (!scored.length) {
    var pending = acc.pending || [];
    box.appendChild(empty('Nothing scored yet',
      'A gameweek needs both a logged prediction and a finished result.' +
      (pending.length ? ' Waiting on GW' + pending.join(', GW') + '.' : '')));
  } else {
    box.appendChild(table(scored, [
      { key: 'gameweek', label: 'GW', num: true },
      { key: 'model', label: 'model' },
      { key: 'n', label: 'players', num: true },
      { key: 'mae', label: 'MAE', dp: 3, num: true },
      { key: 'mae_high_return', label: 'MAE, high return', dp: 3, num: true }
    ]));

    var weeks = {};
    scored.forEach(function (r) { weeks[r.gameweek] = 1; });
    if (Object.keys(weeks).length > 1) {
      var byModel = {};
      scored.forEach(function (r) {
        (byModel[r.model] = byModel[r.model] || []).push({ x: r.gameweek, y: r.mae });
      });

      /* THE CHAMPION IS DRAWN FIRST, so it gets the kit colour and the
         baselines it has to beat sit behind it in gold and stone. Which line
         matters is then visible before reading the legend.

         Insertion order here is alphabetical, which put form_fdr in the kit
         colour and the model under test in grey: the chart said the opposite
         of what it meant, and looked entirely correct doing it. The champion
         is named in accuracy.json rather than typed here, because the day a
         challenger is promoted this has to follow it. */
      var champion = acc.champion;
      var names = Object.keys(byModel).sort(function (a, b) {
        if (a === champion) return -1;
        if (b === champion) return 1;
        return a < b ? -1 : 1;
      });
      box.appendChild(lineChart(names.map(function (k) {
        return { name: k, points: byModel[k] };
      }), { height: 300, ydp: 2, xprefix: 'GW', label: 'Mean absolute error by gameweek' }));
    } else {
      box.appendChild(caption(
        'One scored gameweek so far, GW' + Object.keys(weeks)[0] + '. A chart of a ' +
        'single point would suggest a trend that does not exist yet, so there is ' +
        'not one. It appears from the second scored gameweek.'));
    }
  }

  box.appendChild(el('details', { class: 'more' }, [
    el('summary', { text: 'How the model works, and what was rejected' }),
    el('div', {
      class: 'body',
      html:
        '<p><strong>Two-stage prediction.</strong> Expected points are the ' +
        'probability of playing sixty minutes, multiplied by expected points given ' +
        'that they did. Predicting points directly makes the model hedge every ' +
        'rotation risk into every score. Splitting the question keeps the ' +
        'availability problem and the performance problem apart.</p>' +
        '<p><strong>Validation is chronological, never random k-fold.</strong> Three ' +
        'seasons train, the following season tests. Rolling features are shifted ' +
        'with <code>groupby(...).shift(1)</code> before any window is applied, so a ' +
        'row can only ever see matches that finished before it.</p>' +
        '<p><strong>Leakage was tested, not assumed.</strong> A negative control, the ' +
        'same pipeline with the target shuffled, scored 1.551 MAE against 1.537 for ' +
        'simply predicting the mean. A leaking pipeline would have beaten the mean ' +
        'comfortably.</p>' +
        '<p><strong>The model has a measured noise floor, and it is not small.</strong> ' +
        'Changing nothing but the random seed moves a pickable player’s ' +
        'prediction by 0.169 expected points on average, and by as much as 1.56. Any ' +
        'gap narrower than that is noise wearing two decimal places. This page marks ' +
        'those as tied instead of ranking them, which is the single most useful thing ' +
        'it does.</p>' +
        '<p><strong>A rejected feature is kept in the repo.</strong> <code>opponent.py</code> ' +
        'adds opponent strength. The underlying effect is real and large, 2.73 points ' +
        'against the toughest fifth of opponents versus 4.33 against the leakiest, a ' +
        'swing of 1.59. It still did not improve ranking, and the negative result is ' +
        'documented rather than deleted. It now runs as a registered challenger, ' +
        'predicting every gameweek under its own name and driving nothing.</p>' +
        '<p><strong>Position specific models were tested and rejected</strong> at -0.229 ' +
        'points per pick, the worst of anything tried. That is the second idea ' +
        'borrowed from published prior art to fail on this dataset. Both are kept as ' +
        'documented negative results, because a project that only records its wins is ' +
        'not measuring anything.</p>'
    })
  ]));
  return box;
}

function seasonPanel(d) {
  var box = el('div'), season = d.season;
  if (!season || !season.gameweeks || !season.gameweeks.length) {
    box.appendChild(empty('No gameweeks recorded yet',
      'This is the benchmark the engine has to beat, so it is recorded rather ' +
      'than inferred.'));
    return box;
  }

  box.appendChild(metrics([
    { k: 'Total points', v: String(season.total_points || 0) },
    { k: 'Gameweeks played', v: String(season.played || 0) },
    { k: 'Average', v: season.average ? num(season.average, 1) : 'none yet' }
  ]));

  box.appendChild(table(season.gameweeks, [
    { key: 'gameweek', label: 'GW', num: true },
    { key: 'points', label: 'points', num: true },
    { key: 'transfers', label: 'transfers', num: true },
    { key: 'formation', label: 'formation' },
    { key: 'captain', label: 'captain' },
    { key: 'captain_points', label: 'captain pts', num: true },
    { key: 'decided_by', label: 'decided by' }
  ]));

  /* An unplayed gameweek carries points: null. Dropped explicitly, so the
     line means "gameweeks with a score" and nothing else. */
  var played = season.gameweeks.filter(function (g) {
    return g.played && g.points !== null && g.points !== undefined;
  });
  if (played.length > 1) {
    box.appendChild(lineChart([{
      name: 'Points',
      points: played.map(function (g) { return { x: g.gameweek, y: g.points }; })
    }], { height: 260, ydp: 0, xprefix: 'GW', label: 'Points by gameweek' }));
  }

  box.appendChild(caption(
    'This is the benchmark. GW1 to GW3 were picked on instinct with no model ' +
    'involved, so anything the engine produces has to beat these, not merely ' +
    'beat a naive baseline. The decided by column records who made each call, ' +
    'which is the only way the argument between instinct and model can ever be settled.'));

  /* The warm surface. Everything above is the pitch at kick-off; this is the
     group photo afterwards, and the only part of the page about a person
     rather than a number. The list arrives as photos.json because a static
     host has no directory listing to ask. */
  var photos = d.photos || [];
  if (photos.length) {
    box.appendChild(section('Why any of this matters'));
    box.appendChild(el('div', { class: 'shots' }, photos.map(function (p) {
      return el('figure', null, [
        el('img', { src: 'assets/photos/' + p.file, alt: p.caption, loading: 'lazy' }),
        el('figcaption', { text: p.caption })
      ]);
    })));
    box.appendChild(caption(
      'Katowice, playing for Zimbabwe. The engine exists because somebody who ' +
      'plays the game wanted to know whether a model could out-argue him about it.'));
  }
  return box;
}

/* ============================================================ page build */

var TABS = [
  { id: 'squad', label: 'Squad and captain', build: squadPanel },
  { id: 'transfers', label: 'Transfers', build: transfersPanel },
  { id: 'bench', label: 'Bench', build: benchPanel },
  { id: 'picks', label: 'Recommendations', build: picksPanel },
  { id: 'news', label: 'Team news', build: newsPanel },
  { id: 'model', label: 'Model accuracy', build: modelPanel },
  { id: 'season', label: 'My season', build: seasonPanel }
];

var FILES = ['meta', 'squad', 'captain', 'transfers', 'bench',
             'recommendations', 'alerts', 'accuracy', 'season', 'photos'];

function fetchJson(name) {
  return fetch(DATA + name + '.json', { cache: 'no-cache' })
    .then(function (r) { return r.ok ? r.json() : null; })
    .catch(function () { return null; });
}

function masthead(meta) {
  var deadline = parseIso(meta.deadline);
  var generated = parseIso(meta.generated_at);
  var now = new Date();
  var liveHours = deadline ? (deadline - now) / 3600000 : null;
  var age = generated ? (now - generated) / 3600000 : null;

  /* Urgency is earned, not decorated. Gold only inside the last day before a
     deadline, red only when the data is old enough to be wrong. */
  var dlClass = (liveHours !== null && liveHours >= 0 && liveHours < 24) ? 'urgent' : '';
  var ageClass = (age !== null && age > 14) ? 'alarm' : '';

  document.getElementById('stats').innerHTML =
    '<div class="pfl-stat"><span class="k">Next gameweek</span>' +
    '<span class="v">GW' + esc(meta.next_gw) + '</span></div>' +
    '<div class="pfl-stat"><span class="k">Deadline in</span>' +
    '<span class="v ' + dlClass + '">' + esc(humanise(liveHours)) + '</span></div>' +
    '<div class="pfl-stat"><span class="k">Snapshot</span>' +
    '<span class="v">' + esc(meta.snapshot_id) + '</span></div>' +
    '<div class="pfl-stat"><span class="k">Data age</span>' +
    '<span class="v ' + ageClass + '">' + esc(age === null ? 'unknown' : humanise(age)) + '</span></div>' +
    '<div class="pfl-stat"><span class="k">Rows of history</span>' +
    '<span class="v">' + Number(meta.row_counts.history).toLocaleString('en-GB') + '</span></div>';

  var banners = document.getElementById('banners');
  if (liveHours !== null && liveHours < 0) {
    banners.appendChild(note('',
      'The GW' + esc(meta.next_gw) + ' deadline has passed. Results are scored ' +
      'once the gameweek finishes and the API confirms them, which can take a ' +
      'day after the last match.'));
  } else if (liveHours !== null && liveHours < 26) {
    banners.appendChild(note('warn',
      'GW' + esc(meta.next_gw) + ' deadline in ' + esc(humanise(liveHours)) +
      ', at ' + esc(meta.deadline) + '.'));
  }
  if (age !== null && age > 14) {
    banners.appendChild(note('alarm',
      'This data is ' + esc(humanise(age)) + ' old. The scheduled workflow runs ' +
      'twice daily, so anything beyond about 14 hours means a run failed. Check ' +
      'the Actions tab before trusting a single number on this page.'));
  }

  document.getElementById('foot').innerHTML =
    'Snapshot ' + esc(meta.snapshot_id) + ', generated ' + esc(meta.generated_at) +
    ' by <code>' + esc(meta.published_by || 'unknown') + '</code>. ' +
    Number(meta.row_counts.history).toLocaleString('en-GB') +
    ' rows of historical player data behind every number above.<br>' +
    'Built by GitHub Actions on a schedule, served as static files, computed ' +
    'nowhere near your browser. Predictions are logged before each deadline and ' +
    'never edited afterwards.';
}

function buildTabs(data) {
  var strip = document.getElementById('tabs');
  var panels = document.getElementById('panels');

  TABS.forEach(function (t, i) {
    var btn = el('button', {
      type: 'button', role: 'tab', id: 'tab-' + t.id,
      'aria-controls': 'panel-' + t.id,
      'aria-selected': i === 0 ? 'true' : 'false',
      text: t.label
    });
    var panel = el('div', {
      class: 'panel', role: 'tabpanel', id: 'panel-' + t.id,
      'aria-labelledby': 'tab-' + t.id, tabindex: '0'
    });
    if (i !== 0) panel.hidden = true;

    /* Built once, on first view, and kept. Building all seven up front
       renders six panels nobody has asked for; rebuilding on every click
       throws away the filter state on the recommendations tab. */
    var built = false;
    function show() {
      TABS.forEach(function (o) {
        document.getElementById('tab-' + o.id).setAttribute('aria-selected', o === t ? 'true' : 'false');
        document.getElementById('panel-' + o.id).hidden = (o !== t);
      });
      if (!built) { panel.appendChild(t.build(data)); built = true; }
    }
    btn.addEventListener('click', show);
    if (i === 0) { panel.appendChild(t.build(data)); built = true; }

    strip.appendChild(btn);
    panels.appendChild(panel);
  });
}

function boot() {
  Promise.all(FILES.map(fetchJson)).then(function (loaded) {
    var data = {};
    FILES.forEach(function (name, i) { data[name] = loaded[i]; });

    if (!data.meta) {
      document.getElementById('panels').appendChild(empty(
        'Nothing published yet',
        'The GitHub Actions workflow writes the JSON this page reads. Once it ' +
        'has run at least once, every panel fills in by itself. Nothing here is ' +
        'computed in the browser.'));
      return;
    }

    if (data.transfers && typeof data.transfers.noise_floor === 'number') {
      NOISE_FLOOR = data.transfers.noise_floor;
    }

    masthead(data.meta);
    buildTabs(data);
  });
}

if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', boot);
} else {
  boot();
}
