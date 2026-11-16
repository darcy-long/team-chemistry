"""
Look at the data: our Extra Bases Taken stat against Baseball Savant's published runner values.

Joins our runner-seasons to Savant's leaderboard (same runner, same season), then
    1. prints the season-by-season comparison and the runners we disagree about most,
    2. saves the merged table (one row per runner-season) so you can open it and sort it,
    3. writes a self-contained web page with a season filter, four charts and the tables.

The charts:
    A. every runner-season, our net bases against Savant's Runner Runs (how closely they line up)
    B. the runners whose rank differs most between the two lists
    C. agreement by season (correlation and rank correlation)
    D. our totals divided by Savant's, by season (chances, attempts, thrown out)

Units differ: ours are bases and Savant's are runs, so the comparisons are about order and
spread, not the size of the numbers. "Rank gap" = Savant's rank minus ours within the season
(rank 1 = best), so a positive gap means we rate the runner higher than Savant does.

Input:  data/processed/extra_bases_stat/extra_bases_runners.csv   (--variant picks another version;
        --without-batter-for-3 uses the without_batter_for_3 folder instead)
        data/raw/savant/base_running_YYYY.csv                      (Savant's download)
Output: data/processed/extra_bases_stat/savant_comparison/savant_vs_ours_runners.csv
        data/processed/extra_bases_stat/savant_comparison/savant_vs_ours_report.html

Usage (from the repo root):
    python src/extra_bases_stat/explore_vs_savant.py
    python src/extra_bases_stat/explore_vs_savant.py --open                  # open the page when done
    python src/extra_bases_stat/explore_vs_savant.py --variant _regularseason
    python src/extra_bases_stat/explore_vs_savant.py --min-chances 100
"""

import argparse
import json
import webbrowser
from pathlib import Path

import pandas as pd

from compare_to_savant import SAVANT_DIR, STAT_DIR, load_savant, rank_corr, stat_dir

OUT_DIR = STAT_DIR / "savant_comparison"


def load_merged(suffix="", min_chances=50, without_batter3=False):
    """One row per runner-season that is in both our file and Savant's (Savant with min_chances+)."""
    ours = pd.read_csv(stat_dir(suffix, without_batter3) / f"extra_bases_runners{suffix}.csv")
    savant_years = sorted(int(p.stem.split("_")[-1]) for p in SAVANT_DIR.glob("base_running_*.csv"))
    frames = []
    for year in sorted(set(ours["season"]) & set(savant_years)):
        savant = load_savant(year)
        savant = savant[savant["n_opp_xb"] >= min_chances]
        m = ours[ours["season"] == year].merge(savant, on="mlbam_id", suffixes=("", "_savant"))
        frames.append(m)
    df = pd.concat(frames, ignore_index=True)

    df = df.rename(columns={
        "n_opportunities": "chances", "n_attempts": "attempts", "n_thrown_out": "thrown_out",
        "net_bases_above_expected": "our_bases",
        "n_opp_xb": "sav_chances", "n_att_xb": "sav_attempts", "n_out": "sav_thrown_out",
        "runner_runs": "sav_runs",
    })
    df["our_rank"] = df.groupby("season")["our_bases"].rank(ascending=False, method="min").astype(int)
    df["sav_rank"] = df.groupby("season")["sav_runs"].rank(ascending=False, method="min").astype(int)
    df["rank_gap"] = df["sav_rank"] - df["our_rank"]       # positive: we rate him higher than Savant does
    df["chances_diff"] = df["chances"] - df["sav_chances"]
    df["attempts_diff"] = df["attempts"] - df["sav_attempts"]
    df["thrown_out_diff"] = df["thrown_out"] - df["sav_thrown_out"]
    keep = ["season", "name", "team", "mlbam_id", "our_bases", "sav_runs", "our_rank", "sav_rank", "rank_gap",
            "chances", "sav_chances", "chances_diff", "attempts", "sav_attempts", "attempts_diff",
            "thrown_out", "sav_thrown_out", "thrown_out_diff"]
    return df[keep].sort_values(["season", "our_rank"]).reset_index(drop=True)


def season_summary(df):
    rows = []
    for season, g in df.groupby("season"):
        rows.append({
            "season": int(season), "runners": len(g),
            "pearson": g["our_bases"].corr(g["sav_runs"]), "rank_corr": rank_corr(g["our_bases"], g["sav_runs"]),
            "chances": int(g["chances"].sum()), "sav_chances": int(g["sav_chances"].sum()),
            "attempts": int(g["attempts"].sum()), "sav_attempts": int(g["sav_attempts"].sum()),
            "thrown_out": int(g["thrown_out"].sum()), "sav_thrown_out": int(g["sav_thrown_out"].sum()),
        })
    out = pd.DataFrame(rows)
    out["chances_ratio"] = out["chances"] / out["sav_chances"]
    out["attempts_ratio"] = out["attempts"] / out["sav_attempts"]
    out["thrown_out_ratio"] = out["thrown_out"] / out["sav_thrown_out"]
    return out


def print_report(df, summary, n=8):
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 30)
    print(f"\n{len(df):,} runner-seasons in both lists, {summary['season'].nunique()} seasons")
    show = summary.rename(columns={"pearson": "r", "rank_corr": "rank r", "chances_ratio": "chances x",
                                   "attempts_ratio": "attempts x", "thrown_out_ratio": "thrown out x"})
    print("\nBy season (r = correlation of our net bases with Savant Runner Runs; 'x' = ours divided by Savant's):")
    print(show[["season", "runners", "r", "rank r", "chances x", "attempts x", "thrown out x"]].round(3).to_string(index=False))
    print(f"\nAverage over seasons: r {summary['pearson'].mean():.3f}, rank r {summary['rank_corr'].mean():.3f}")
    cols = ["season", "name", "our_rank", "sav_rank", "rank_gap", "our_bases", "sav_runs"]
    print(f"\nWe rate him MUCH higher than Savant does (top {n}):")
    print(df.nlargest(n, "rank_gap")[cols].round(2).to_string(index=False))
    print(f"\nSavant rates him MUCH higher than we do (top {n}):")
    print(df.nsmallest(n, "rank_gap")[cols].round(2).to_string(index=False))


TEMPLATE = r"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Extra Bases Taken: ours vs Savant</title>
<style>
:root { color-scheme: light dark;
  --page:#f9f9f7; --surface:#fcfcfb; --ink1:#0b0b0b; --ink2:#52514e; --muted:#898781;
  --grid:#e1e0d9; --axis:#c3c2b7; --hair:rgba(11,11,11,0.10);
  --s1:#2a78d6; --s2:#eb6834; --s3:#1baf7a; --pos:#2a78d6; --neg:#e34948; }
@media (prefers-color-scheme: dark) { :root { --page:#0d0d0d; --surface:#1a1a19; --ink1:#f0efec; --ink2:#c3c2b7;
  --muted:#898781; --grid:#2c2c2a; --axis:#383835; --hair:rgba(255,255,255,0.10);
  --s1:#3987e5; --s2:#d95926; --s3:#199e70; --pos:#3987e5; --neg:#e66767; } }
* { box-sizing:border-box; }
body { margin:0; background:var(--page); color:var(--ink1); font:15px/1.5 system-ui,-apple-system,"Segoe UI",sans-serif; }
main { max-width:1080px; margin:0 auto; padding:28px 20px 60px; }
h1 { font-size:24px; margin:0 0 4px; font-weight:600; }
.sub { color:var(--ink2); margin:0 0 18px; max-width:760px; }
.filters { display:flex; gap:12px; align-items:center; margin:0 0 18px; }
.filters label { color:var(--ink2); font-size:13px; }
select { font:inherit; padding:6px 10px; border:1px solid var(--axis); border-radius:8px; background:var(--surface); color:var(--ink1); }
.tiles { display:grid; grid-template-columns:repeat(auto-fit,minmax(200px,1fr)); gap:12px; margin-bottom:18px; }
.tile, .card { background:var(--surface); border:1px solid var(--hair); border-radius:12px; padding:16px 18px; }
.tile .label { color:var(--ink2); font-size:13px; }
.tile .value { font-size:30px; font-weight:600; letter-spacing:-0.01em; margin:2px 0; }
.tile .note { color:var(--muted); font-size:12px; }
.card { margin-bottom:18px; }
.card h2 { font-size:16px; margin:0 0 2px; font-weight:600; }
.card .why { color:var(--ink2); font-size:13px; margin:0 0 8px; }
.card .read { color:var(--muted); font-size:12px; margin:6px 0 0; }
.keyrow { display:flex; gap:16px; flex-wrap:wrap; font-size:13px; color:var(--ink2); margin:2px 0 6px; }
.keyrow span::before { content:""; display:inline-block; width:14px; height:3px; border-radius:2px; margin-right:6px; vertical-align:middle; background:var(--c); }
table { border-collapse:collapse; width:100%; font-size:13px; }
th, td { text-align:right; padding:5px 8px; border-bottom:1px solid var(--grid); }
th { color:var(--ink2); font-weight:600; } th:first-child, td:first-child { text-align:left; }
td.num { font-variant-numeric:tabular-nums; }
details summary { cursor:pointer; color:var(--ink2); font-size:14px; margin:6px 0 10px; }
.foot { color:var(--muted); font-size:12px; margin-top:20px; }
</style></head><body><main>
<h1>Extra Bases Taken: our stat against Baseball Savant</h1>
<p class="sub">Each dot is one runner in one season. Our number is net bases gained over what a typical runner would
have gained on the same plays; Savant's is Runner Runs. The two are in different units, so the question is whether the
<em>order</em> and <em>spread</em> agree, not whether the numbers match. Version: __VARIANT__</p>

<div class="filters"><label for="season">Season</label><select id="season"></select></div>
<div class="tiles" id="tiles"></div>

<section class="card"><h2>Do the two lists line up?</h2>
<p class="why">Our net bases (up) against Savant Runner Runs (right), one dot per runner-season.</p>
<div id="chartScatter" role="img" aria-label="Scatter of our net bases against Savant Runner Runs"></div>
<p class="read">The line is the best straight-line fit. Dots above it are runners we rate higher than Savant does, below it lower. Hover a dot for the runner.</p></section>

<section class="card"><h2>Where we disagree most</h2>
<p class="why">Ranks within the season (1 = best). The bar is how many places apart the two lists put the runner.</p>
<div id="chartGap" role="img" aria-label="Bar chart of the runners with the largest rank differences"></div>
<p class="read">Blue: we rate him higher than Savant. Red: Savant rates him higher than we do.</p></section>

<section class="card"><h2>Agreement by season</h2>
<p class="why">How closely the two lists match, season by season (1 = identical order). The axis starts at 0.7.</p>
<div id="chartAgree" role="img" aria-label="Line chart of correlation by season"></div></section>

<section class="card"><h2>How our counts compare with Savant's</h2>
<p class="why">Our total divided by Savant's, for the same runners. 1.0 means we count the same number.</p>
<div id="chartRatio" role="img" aria-label="Line chart of ours divided by Savant by season"></div></section>

<section class="card"><h2>Table view</h2>
<details open><summary>Season by season</summary><div id="tblSeason"></div></details>
<details open><summary>Biggest rank gaps (selected season)</summary><div id="tblGap"></div></details></section>

<p class="foot">Every runner-season is in <code>savant_vs_ours_runners.csv</code> next to this file. Runners with
fewer than the minimum Savant chances are left out, since Savant does not list them.</p>
</main>

<script>__PLOTLY__</script>
<script id="data" type="application/json">__DATA__</script>
<script>
const D = JSON.parse(document.getElementById('data').textContent);
const rows = D.rows, summary = D.summary;
let current = 'all';
const css = n => getComputedStyle(document.documentElement).getPropertyValue(n).trim();
const tk = () => ({surface:css('--surface'), ink1:css('--ink1'), ink2:css('--ink2'), muted:css('--muted'), grid:css('--grid'),
  axis:css('--axis'), s1:css('--s1'), s2:css('--s2'), s3:css('--s3'), pos:css('--pos'), neg:css('--neg')});
const esc = s => String(s).replace(/[<>&]/g, c => ({'<':'&lt;','>':'&gt;','&':'&amp;'}[c]));
const sel = () => current === 'all' ? rows : rows.filter(r => r.season === +current);
const f = (x, d=2) => (x >= 0 ? '+' : '−') + Math.abs(x).toFixed(d);
const cfg = {responsive:true, displayModeBar:false};

function base(t, extra) {
  return Object.assign({paper_bgcolor:t.surface, plot_bgcolor:t.surface, margin:{l:64,r:30,t:10,b:54},
    font:{family:'system-ui,-apple-system,Segoe UI,sans-serif', size:12, color:t.ink2},
    hoverlabel:{bgcolor:t.surface, bordercolor:t.grid, font:{color:t.ink1, size:12}}, hovermode:'closest', hoverdistance:24}, extra);
}
const axis = (t, title, extra) => Object.assign({title:{text:title, font:{color:t.ink2}}, gridcolor:t.grid, gridwidth:1,
  linecolor:t.axis, zerolinecolor:t.axis, zerolinewidth:1, tickfont:{color:t.muted}, showline:false}, extra||{});

function linfit(xs, ys) {
  const n = xs.length, mx = xs.reduce((a,b)=>a+b,0)/n, my = ys.reduce((a,b)=>a+b,0)/n;
  let sxx = 0, sxy = 0; for (let i=0;i<n;i++){ sxx += (xs[i]-mx)**2; sxy += (xs[i]-mx)*(ys[i]-my); }
  const b = sxy/sxx; return {b, a:my-b*mx};
}

function drawTiles() {
  const row = current === 'all' ? null : summary.find(s => s.season === +current);
  const pick = k => row ? row[k] : summary.reduce((a,s)=>a+s[k],0)/summary.length;
  const tiles = [
    ['Agreement (correlation)', pick('pearson').toFixed(2), current==='all' ? 'average of the seasons' : 'this season', 'Closer to 1 means the lists move together'],
    ['Same ordering (rank correlation)', pick('rank_corr').toFixed(2), current==='all' ? 'average of the seasons' : 'this season', 'Looks only at who ranks above whom'],
    ['Chances: ours vs Savant', pick('chances_ratio').toFixed(2) + '×', 'same runners', 'Above 1 means we count more chances'],
    ['Thrown out: ours vs Savant', pick('thrown_out_ratio').toFixed(2) + '×', 'same runners', 'Above 1 means we count more outs']];
  const box = document.getElementById('tiles'); box.textContent = '';
  tiles.forEach(([l,v,n,h]) => { const d = document.createElement('div'); d.className = 'tile';
    [['label',l],['value',v],['note',n + ' · ' + h]].forEach(([c,txt]) => { const e = document.createElement('div'); e.className = c; e.textContent = txt; d.appendChild(e); });
    box.appendChild(d); });
}

function drawScatter() {
  const t = tk(), d = sel(), xs = d.map(r=>r.sav_runs), ys = d.map(r=>r.our_bases);
  const fit = linfit(xs, ys), x0 = Math.min(...xs), x1 = Math.max(...xs);
  const resid = d.map((r,i) => ys[i] - (fit.a + fit.b*xs[i]));
  const order = idx => idx.sort((i,j)=>resid[j]-resid[i]);
  const idx = d.map((_,i)=>i);
  const lab = new Set([...idx.slice().sort((i,j)=>xs[j]-xs[i]).slice(0,3), ...idx.slice().sort((i,j)=>xs[i]-xs[j]).slice(0,2),
                       ...order(idx.slice()).slice(0,2), ...order(idx.slice()).slice(-2)]);
  const L = [...lab];
  const traces = [
    {type:'scatter', mode:'markers', x:xs, y:ys, showlegend:false,
     customdata:d.map(r => [esc(r.name), r.season, r.our_rank, r.sav_rank, r.chances]),
     hovertemplate:'<b>%{y:+.1f} bases</b> · Savant %{x:+.1f} runs<br>%{customdata[0]}, %{customdata[1]}<br>Our rank %{customdata[2]} · Savant rank %{customdata[3]}<extra></extra>',
     marker:{size:8, color:t.s1, opacity:0.55, line:{color:t.surface, width:1.5}}},
    {type:'scatter', mode:'lines', x:[x0,x1], y:[fit.a+fit.b*x0, fit.a+fit.b*x1], showlegend:false, hoverinfo:'skip',
     line:{color:t.muted, width:2}},
    {type:'scatter', mode:'text', x:L.map(i=>xs[i]), y:L.map(i=>ys[i]), text:L.map(i=>esc(d[i].name) + (current==='all' ? ' ' + d[i].season : '')),
     textposition:L.map(i => xs[i] > x0 + 0.72*(x1-x0) ? 'top left' : 'top right'),   // labels near the right edge go left
     textfont:{color:t.ink2, size:11}, showlegend:false, hoverinfo:'skip', cliponaxis:false}];
  const layout = base(t, {height:460, xaxis:axis(t,'Savant Runner Runs (runs)'), yaxis:axis(t,'Our net bases above expected (bases)'),
    annotations:[{x:x0+0.62*(x1-x0), y:fit.a+fit.b*(x0+0.62*(x1-x0)), xanchor:'left', yanchor:'top', xshift:6, yshift:-6, text:'best-fit line', showarrow:false, font:{color:t.muted, size:11}}]});
  Plotly.react('chartScatter', traces, layout, cfg);
}

function drawGap() {
  const t = tk(), d = sel().slice().sort((a,b)=>b.rank_gap-a.rank_gap), n = Math.min(10, Math.floor(d.length/2));
  const list = d.slice(0,n).concat(d.slice(-n));
  const lbl = r => r.name + (current==='all' ? ' (' + r.season + ')' : '');
  const mk = (rs, name, color) => ({type:'bar', orientation:'h', name, x:rs.map(r=>r.rank_gap), y:rs.map(lbl),
    marker:{color, cornerradius:4}, text:rs.map(r=>(r.rank_gap>0?'+':'−') + Math.abs(r.rank_gap)), textposition:'outside', cliponaxis:false,
    textfont:{color:t.ink2, size:11},
    customdata:rs.map(r => [r.our_rank, r.sav_rank, r.our_bases.toFixed(1), r.sav_runs.toFixed(1)]),
    hovertemplate:'<b>%{x:+d} places</b><br>%{y}<br>Our rank %{customdata[0]} · Savant rank %{customdata[1]}<br>Ours %{customdata[2]} bases · Savant %{customdata[3]} runs<extra></extra>'});
  const traces = [mk(list.filter(r=>r.rank_gap>0), 'We rate him higher', t.pos), mk(list.filter(r=>r.rank_gap<=0), 'Savant rates him higher', t.neg)];
  const m = Math.max(...list.map(r=>Math.abs(r.rank_gap)), 10) * 1.2;
  const layout = base(t, {height: 34*list.length + 110, bargap:0.35, margin:{l:190,r:40,t:30,b:50},
    xaxis:axis(t,'Places apart in the rank (1 = best)', {range:[-m,m], zeroline:true}),
    yaxis:{categoryorder:'array', categoryarray:list.map(lbl), autorange:'reversed', tickfont:{color:t.ink1, size:12}, gridcolor:'rgba(0,0,0,0)', linecolor:t.axis},
    legend:{orientation:'h', x:0, y:1.08, font:{color:t.ink2}}});
  Plotly.react('chartGap', traces, layout, cfg);
}

function lineTrace(t, name, x, y, color, fmt) {
  return {type:'scatter', mode:'lines+markers', name, x, y, line:{color, width:2, shape:'linear'},
    marker:{size:8, color, line:{color:t.surface, width:2}},
    hovertemplate:'<b>' + fmt + '</b> · ' + name + '<br>%{x}<extra></extra>'};
}
function endLabels(t, items) {
  return items.map(([text, y, color]) => ({xref:'paper', x:1.0, xanchor:'left', y, yanchor:'middle', text:esc(text), showarrow:false,
    font:{color:t.ink2, size:12}, xshift:8}));
}
function drawAgree() {
  const t = tk(), x = summary.map(s=>String(s.season));
  const traces = [lineTrace(t,'Correlation', x, summary.map(s=>s.pearson), t.s1, '%{y:.2f}'),
                  lineTrace(t,'Rank correlation', x, summary.map(s=>s.rank_corr), t.s2, '%{y:.2f}')];
  const last = summary[summary.length-1];
  const layout = base(t, {height:320, margin:{l:64,r:130,t:10,b:54}, xaxis:axis(t,'Season',{type:'category'}),
    yaxis:axis(t,'Agreement with Savant',{range:[0.7,1.0], tickformat:'.2f'}), legend:{orientation:'h', x:0, y:1.14, font:{color:t.ink2}},
    annotations:endLabels(t, [['Correlation', last.pearson, t.s1], ['Rank correlation', last.rank_corr, t.s2]])});
  Plotly.react('chartAgree', traces, layout, cfg);
}
function drawRatio() {
  const t = tk(), x = summary.map(s=>String(s.season)), last = summary[summary.length-1];
  const traces = [lineTrace(t,'Chances', x, summary.map(s=>s.chances_ratio), t.s1, '%{y:.2f}×'),
                  lineTrace(t,'Attempts', x, summary.map(s=>s.attempts_ratio), t.s2, '%{y:.2f}×'),
                  lineTrace(t,'Thrown out', x, summary.map(s=>s.thrown_out_ratio), t.s3, '%{y:.2f}×')];
  const layout = base(t, {height:340, margin:{l:64,r:130,t:10,b:54}, xaxis:axis(t,'Season',{type:'category'}),
    yaxis:axis(t,'Ours divided by Savant',{range:[0.8,1.4], tickformat:'.1f'}), legend:{orientation:'h', x:0, y:1.14, font:{color:t.ink2}},
    shapes:[{type:'line', xref:'paper', x0:0, x1:1, y0:1, y1:1, line:{color:t.axis, width:1}}],
    annotations:endLabels(t, [['Chances', last.chances_ratio, t.s1], ['Attempts', last.attempts_ratio, t.s2], ['Thrown out', last.thrown_out_ratio, t.s3]])
      .concat([{xref:'paper', x:0.0, yref:'y', y:1.0, xanchor:'left', yanchor:'top', text:'same as Savant', showarrow:false, yshift:-4, font:{color:t.muted, size:11}}])});
  Plotly.react('chartRatio', traces, layout, cfg);
}

function table(el, head, body) {
  el.textContent = ''; const tb = document.createElement('table'), th = document.createElement('thead'), tr = document.createElement('tr');
  head.forEach(h => { const c = document.createElement('th'); c.textContent = h; tr.appendChild(c); }); th.appendChild(tr); tb.appendChild(th);
  const bd = document.createElement('tbody');
  body.forEach(r => { const row = document.createElement('tr'); r.forEach((v,i) => { const c = document.createElement('td'); if (i>0) c.className='num'; c.textContent = v; row.appendChild(c); }); bd.appendChild(row); });
  tb.appendChild(bd); el.appendChild(tb);
}
function drawTables() {
  table(document.getElementById('tblSeason'),
    ['Season','Runners','Correlation','Rank corr.','Chances ours','Savant','Attempts ours','Savant','Thrown out ours','Savant'],
    summary.map(s => [s.season, s.runners, s.pearson.toFixed(3), s.rank_corr.toFixed(3), s.chances, s.sav_chances, s.attempts, s.sav_attempts, s.thrown_out, s.sav_thrown_out]));
  const d = sel().slice().sort((a,b)=>Math.abs(b.rank_gap)-Math.abs(a.rank_gap)).slice(0,20);
  table(document.getElementById('tblGap'), ['Runner','Season','Our rank','Savant rank','Gap (places)','Our bases','Savant runs'],
    d.map(r => [r.name, r.season, r.our_rank, r.sav_rank, (r.rank_gap>0?'+':'') + r.rank_gap, f(r.our_bases,1), f(r.sav_runs,1)]));
}

function drawAll() { drawTiles(); drawScatter(); drawGap(); drawAgree(); drawRatio(); drawTables(); }
const selEl = document.getElementById('season');
[['all','All seasons'], ...D.seasons.map(s => [String(s), String(s)])].forEach(([v,l]) => { const o = document.createElement('option'); o.value = v; o.textContent = l; selEl.appendChild(o); });
selEl.addEventListener('change', e => { current = e.target.value; drawAll(); });
window.matchMedia('(prefers-color-scheme: dark)').addEventListener('change', drawAll);
drawAll();
</script></body></html>
"""


def plotly_js():
    """Plotly's JavaScript, read from the installed package so the page works offline."""
    import plotly
    path = Path(plotly.__file__).parent / "package_data" / "plotly.min.js"
    return path.read_text(encoding="utf-8")


def write_html(df, summary, path, variant_label):
    rows = df[["season", "name", "our_bases", "sav_runs", "our_rank", "sav_rank", "rank_gap", "chances"]].copy()
    rows["our_bases"] = rows["our_bases"].round(3)
    rows["sav_runs"] = rows["sav_runs"].round(3)
    payload = {"rows": rows.to_dict("records"),
               "summary": summary.round(5).to_dict("records"),
               "seasons": [int(s) for s in summary["season"]]}
    data = json.dumps(payload).replace("</", "<\\/")      # a name can never close the script tag
    html = (TEMPLATE.replace("__PLOTLY__", plotly_js()).replace("__DATA__", data)
            .replace("__VARIANT__", variant_label))
    path.write_text(html, encoding="utf-8")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Compare our Extra Bases Taken stat with Savant's, with charts.")
    ap.add_argument("--variant", default="", help='which of our outputs to use, e.g. "_regularseason" (default: the main one)')
    ap.add_argument("--without-batter-for-3", action="store_true",
                    help="use the outputs that leave out batter_for_3 (the without_batter_for_3 folder)")
    ap.add_argument("--min-chances", type=int, default=50, help="minimum Savant chances to include a runner")
    ap.add_argument("--open", action="store_true", help="open the page in the browser when done")
    args = ap.parse_args()

    merged = load_merged(args.variant, args.min_chances, args.without_batter_for_3)
    summary = season_summary(merged)
    print_report(merged, summary)

    out_dir = (STAT_DIR / "without_batter_for_3" if args.without_batter_for_3 else STAT_DIR) / "savant_comparison"
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "savant_vs_ours_runners.csv"
    html_path = out_dir / "savant_vs_ours_report.html"
    merged.to_csv(csv_path, index=False)
    write_html(merged, summary, html_path, (args.variant.strip("_") or "main (speed-neutral, results, postseason included)")
                + (", without batter_for_3" if args.without_batter_for_3 else ""))
    print(f"\nSaved {csv_path}\nSaved {html_path}")
    if args.open:
        webbrowser.open(html_path.as_uri())
