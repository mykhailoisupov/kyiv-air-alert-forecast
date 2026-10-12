"""One self-contained HTML page: state now, onset chances, the last 7 days, and a replay of the test weeks.

The page is plain HTML with inline CSS, inline SVG and a few lines of inline script for
hover read-outs. It loads nothing from the network and needs no server.
"""

from __future__ import annotations

import html
import json
from pathlib import Path

import numpy as np
import pandas as pd

from airalerts import data
from airalerts.grid import HOUR, epoch_seconds
from airalerts.model import FINAL, HORIZONS
from airalerts.now import SIMPLE_NAMES, minutes_text

WIDTH, GUTTER, PLOT = 1000, 96, 864  # shared SVG geometry: total width, left gutter, plot width
STRIP_DAYS, STRIP_TOP, STRIP_ROW, STRIP_BAR = 7, 8, 26, 14
WEEK_HOURS, REPLAY_TOP, REPLAY_PLOT, REPLAY_BAND, REPLAY_ROW = 168, 14, 56, 6, 82
THREAT_CLASS = {"drone": "drone", "missile": "missile"}
THREAT_LABEL = {"drone": "drone alert (yellow level)", "missile": "missile-type alert (red level)"}
BINS = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)

CSS = """
:root { color-scheme: light; --page: #f9f9f7; --surface: #fcfcfb; --ink: #0b0b0b; --ink2: #52514e;
  --muted: #898781; --grid: #e1e0d9; --axis: #c3c2b7; --border: rgba(11,11,11,0.10);
  --series: #2a78d6; --track: #cde2fb; --simple: #898781;
  --drone: #fab219; --missile: #d03b3b; --plain: #898781; --good: #0ca30c; }
@media (prefers-color-scheme: dark) {
  :root:not([data-theme=light]) { color-scheme: dark; --page: #0d0d0d; --surface: #1a1a19; --ink: #f0efec;
    --ink2: #c3c2b7; --muted: #898781; --grid: #2c2c2a; --axis: #383835; --border: rgba(255,255,255,0.10);
    --series: #3987e5; --track: #184f95; --simple: #898781; } }
* { box-sizing: border-box; }
body { margin: 0; background: var(--page); color: var(--ink);
  font: 15px/1.45 system-ui, -apple-system, "Segoe UI", sans-serif; }
main { max-width: 1040px; margin: 0 auto; padding: 24px 16px 48px; }
h1 { font-size: 24px; margin: 0 0 2px; }
h2 { font-size: 16px; margin: 0 0 2px; }
p { margin: 0; }
.sub { color: var(--ink2); font-size: 13px; }
.card { background: var(--surface); border: 1px solid var(--border); border-radius: 12px;
  padding: 16px 20px; margin-top: 16px; }
.state { font-size: 18px; margin-top: 12px; display: flex; gap: 10px; align-items: center; }
.dot { width: 12px; height: 12px; border-radius: 50%; flex: none; }
.dot.quiet { background: var(--good); } .dot.alert { background: var(--missile); }
.tiles { display: grid; grid-template-columns: repeat(auto-fit, minmax(210px, 1fr)); gap: 12px;
  margin-top: 12px; }
.tile { background: var(--surface); border: 1px solid var(--border); border-radius: 10px; padding: 12px 14px; }
.tile .label { color: var(--ink2); font-size: 13px; }
.tile .value { font-size: 40px; font-weight: 600; line-height: 1.15; }
.meter { width: 100%; height: 12px; display: block; margin: 4px 0 6px; }
.scroll { overflow-x: auto; margin-top: 10px; }
.scroll svg { display: block; width: 100%; min-width: 720px; height: auto; }
svg text { font: 11px system-ui, -apple-system, "Segoe UI", sans-serif; fill: var(--ink2); }
svg text.quiet { fill: var(--muted); }
.gridline { stroke: var(--grid); stroke-width: 1; } .axis { stroke: var(--axis); stroke-width: 1; }
.track { fill: var(--track); } .fill { fill: var(--series); } .tick { fill: var(--ink); }
.drone { fill: var(--drone); } .missile { fill: var(--missile); } .plain { fill: var(--plain); }
i.drone { background: var(--drone); } i.missile { background: var(--missile); }
i.plain { background: var(--plain); }
.model { fill: none; stroke: var(--series); stroke-width: 2; stroke-linejoin: round; stroke-linecap: round; }
.simple { fill: none; stroke: var(--simple); stroke-width: 1.5; stroke-linejoin: round;
  stroke-linecap: round; }
.hit { fill: transparent; cursor: default; } .hit:focus { outline: 2px solid var(--series); }
.xhair { stroke: var(--ink2); stroke-width: 1; visibility: hidden; }
.legend { display: flex; flex-wrap: wrap; gap: 6px 18px; margin-top: 10px; color: var(--ink2);
  font-size: 13px; }
.legend span { display: inline-flex; align-items: center; gap: 6px; }
.key { width: 14px; height: 10px; border-radius: 2px; display: inline-block; }
.key.line { height: 0; border-top: 2px solid var(--series); border-radius: 0; }
.key.line.grey { border-top-color: var(--simple); }
.tabs { display: flex; gap: 6px; margin-top: 10px; }
.tabs label { border: 1px solid var(--border); border-radius: 8px; padding: 4px 12px; cursor: pointer;
  color: var(--ink2); font-size: 13px; }
input.tab { position: absolute; opacity: 0; pointer-events: none; }
.panel { display: none; }
#h1:checked ~ .panels .p1, #h3:checked ~ .panels .p3, #h6:checked ~ .panels .p6 { display: block; }
#h1:checked ~ .tabs label[for=h1], #h3:checked ~ .tabs label[for=h3],
#h6:checked ~ .tabs label[for=h6] { color: var(--ink); border-color: var(--ink2); font-weight: 600; }
#h1:focus-visible ~ .tabs label[for=h1], #h3:focus-visible ~ .tabs label[for=h3],
#h6:focus-visible ~ .tabs label[for=h6] { outline: 2px solid var(--series); }
table { border-collapse: collapse; font-size: 13px; margin-top: 8px; }
th, td { text-align: right; padding: 3px 12px 3px 0; font-variant-numeric: tabular-nums; }
th { color: var(--ink2); font-weight: 500; } th:first-child, td:first-child { text-align: left; }
details { margin-top: 10px; } summary { cursor: pointer; color: var(--ink2); font-size: 13px; }
#tip { position: fixed; z-index: 5; pointer-events: none; display: none; background: var(--surface);
  color: var(--ink); border: 1px solid var(--border); border-radius: 8px; padding: 6px 10px;
  font-size: 12px; line-height: 1.4; box-shadow: 0 4px 14px rgba(0,0,0,0.14); max-width: 260px; }
#tip b { font-size: 13px; } #tip div + div { color: var(--ink2); }
footer { color: var(--ink2); font-size: 13px; margin-top: 20px; }
"""

SCRIPT = """
const tip = document.getElementById('tip');
function show(lines, x, y) {
  tip.replaceChildren();
  lines.forEach((text, i) => {
    const row = document.createElement('div');
    const node = i === 0 ? document.createElement('b') : document.createTextNode(text);
    if (i === 0) node.textContent = text;
    row.appendChild(node);
    tip.appendChild(row);
  });
  tip.style.display = 'block';
  const box = tip.getBoundingClientRect();
  tip.style.left = Math.max(8, Math.min(x + 14, window.innerWidth - box.width - 8)) + 'px';
  tip.style.top = Math.max(8, y - box.height - 12) + 'px';
}
function hide() { tip.style.display = 'none'; }
document.querySelectorAll('[data-tip]').forEach((mark) => {
  const lines = mark.dataset.tip.split('|');
  mark.addEventListener('pointermove', (e) => show(lines, e.clientX, e.clientY));
  mark.addEventListener('pointerleave', hide);
  mark.addEventListener('focus', () => {
    const box = mark.getBoundingClientRect();
    show(lines, box.left + box.width / 2, box.top);
  });
  mark.addEventListener('blur', hide);
});
const replay = JSON.parse(document.getElementById('replay-data').textContent);
const clock = new Intl.DateTimeFormat('en-GB', { timeZone: 'Europe/Kyiv', weekday: 'short',
  day: 'numeric', month: 'short', hour: '2-digit', minute: '2-digit' });
const pct = (p) => Math.round(100 * p) + '%';
document.querySelectorAll('svg[data-horizon]').forEach((svg) => {
  const h = svg.dataset.horizon, g = replay.geometry, rows = replay.horizons[h];
  const hair = svg.querySelector('.xhair'), zone = svg.querySelector('.zone');
  zone.addEventListener('pointermove', (e) => {
    const p = new DOMPoint(e.clientX, e.clientY).matrixTransform(svg.getScreenCTM().inverse());
    const week = Math.max(0, Math.min(g.weeks - 1, Math.floor((p.y - g.top) / g.row)));
    const hour = Math.max(0, Math.min(g.hours - 1, Math.round((p.x - g.gutter) / g.plot * g.hours)));
    const index = week * g.hours + hour, row = rows[index];
    if (row === undefined) { hide(); return; }
    const x = g.gutter + g.plot * hour / g.hours, y = g.top + week * g.row;
    hair.setAttribute('x1', x); hair.setAttribute('x2', x);
    hair.setAttribute('y1', y); hair.setAttribute('y2', y + g.band);
    hair.style.visibility = 'visible';
    const lines = [clock.format(new Date(replay.start + index * 3600000))];
    if (row === 0) lines.push('An alert was running, so there is no forecast.');
    else lines.push(pct(row[0]) + ' model', pct(row[1]) + ' best simple forecast',
      (row[2] ? 'A new alert started' : 'No new alert') + ' within ' + h + ' h');
    show(lines, e.clientX, e.clientY);
  });
  zone.addEventListener('pointerleave', () => { hide(); hair.style.visibility = 'hidden'; });
});
"""


def esc(text: object) -> str:
    return html.escape(str(text), quote=True)


def tag(element: str, body: object = None, /, **attrs: object) -> str:
    """One element. `cls` becomes class, underscores become hyphens, floats get one decimal."""
    pairs = []
    for key, value in attrs.items():
        key = "class" if key == "cls" else key.replace("_", "-")
        value = f"{value:.1f}" if isinstance(value, float) else str(value)
        pairs.append(f' {key}="{esc(value)}"')
    opening = f"<{element}{''.join(pairs)}"
    return f"{opening}/>" if body is None else f"{opening}>{body}</{element}>"


def vline(css: str, x: float, y1: float, y2: float) -> str:
    return tag("line", cls=css, x1=x, x2=x, y1=y1, y2=y2)


def hline(css: str, y: float) -> str:
    return tag("line", cls=css, x1=GUTTER, x2=GUTTER + PLOT, y1=y, y2=y)


def kyiv(t: pd.Timestamp) -> pd.Timestamp:
    return t.tz_convert(data.KYIV_TZ)


def wrap(parts: list[str], per_line: int) -> str:
    """Join short fragments into lines, so the generated file stays readable in a diff."""
    return "\n".join(" ".join(parts[i : i + per_line]) for i in range(0, len(parts), per_line))


def reflow(markup: str, limit: int = 118) -> str:
    """Put each element on its own line and break long lines at spaces outside quoted attribute values.

    HTML treats a line break like a space there, so only the file's layout changes. Used for
    markup only; the style, script and data blocks are written with short lines already.
    """
    out = []
    for line in markup.replace("><", ">\n<").split("\n"):
        in_tag = False
        while len(line) > limit:
            cut, cut_in_tag, quoted, state = 0, in_tag, False, in_tag
            for i, char in enumerate(line[: limit + 1]):
                if char == '"' and state:
                    quoted = not quoted
                elif char == "<" and not quoted:
                    state = True
                elif char == ">" and not quoted:
                    state = False
                elif char == " " and not quoted and i:
                    cut, cut_in_tag = i, state
            if not cut:
                break  # one unbreakable token; leave the line as it is
            out.append(line[:cut])
            line, in_tag = line[cut + 1 :], cut_in_tag
        out.append(line)
    return "\n".join(out)


def state_block(live: dict) -> str:
    if not live["active"]:
        quiet = minutes_text(live["minutes_since_end"])
        text = f"No alert is active. The last one ended {quiet} ago."
        return tag("p", tag("span", "", cls="dot quiet") + text, cls="state")
    text = f"An alert has been running for {minutes_text(live['minutes_since_start'])}."
    left = live.get("remaining")
    if left:
        low, high = minutes_text(left["low_min"]), minutes_text(left["high_min"])
        text += f" Likely to end in about {minutes_text(left['median_min'])} (80% interval: {low} to {high})."
    return tag("p", tag("span", "", cls="dot alert") + esc(text), cls="state")


def tiles_block(live: dict) -> str:
    if live["active"]:
        return tag("p", "The chance of a new alert is forecast only while no alert is active.", cls="sub")
    tiles = []
    for h, row in live["horizons"].items():
        label = "1 hour" if h == 1 else f"{h} hours"
        model, simple = 100 * row["model"], 100 * row["simple"]
        meter = tag("rect", cls="track", width=100, height=12, rx=3)
        meter += tag("rect", cls="fill", width=model, height=12, rx=3)
        meter += tag("rect", cls="tick", x=simple - 0.4, width=0.8, height=12)
        described = f"Model {model:.0f}%, simple forecast {simple:.0f}%"
        note = f"Simple forecast {simple:.0f}% (the tick on the bar): {SIMPLE_NAMES[row['simple_name']]}"
        tiles.append(
            tag(
                "div",
                tag("div", f"New alert within {label}", cls="label")
                + tag("div", f"{model:.0f}%", cls="value")
                + tag("svg", meter, cls="meter", viewBox="0 0 100 12", preserveAspectRatio="none", role="img",
                      aria_label=described)
                + tag("div", esc(note), cls="sub"),
                cls="tile",
            )
        )
    return tag("div", "\n".join(tiles), cls="tiles")


def alert_spans(events: pd.DataFrame, as_of: pd.Timestamp) -> pd.DataFrame:
    """Alerts as known at `as_of`; one still running is drawn up to `as_of` and flagged."""
    known = data.truncate(events, as_of)
    spans = known[["start", "threat"]].copy()
    spans["running"] = known["end"].isna()
    spans["end"] = known["end"].fillna(as_of)
    return spans


def span_text(row) -> tuple[str, str, str, str]:
    """Start, end, length and kind of one alert, in Kyiv time."""
    length = minutes_text((row.end - row.start).total_seconds() / 60)
    ended = "still running" if row.running else f"{kyiv(row.end):%H:%M}"
    return f"{kyiv(row.start):%a %d %b %H:%M}", ended, length, THREAT_LABEL.get(row.threat, "alert")


def strip_block(events: pd.DataFrame, as_of: pd.Timestamp) -> str:
    """Seven Kyiv calendar days, one row each, with every alert drawn as a bar."""
    spans = alert_spans(events, as_of)
    now_local = kyiv(as_of)
    first = now_local.normalize() - pd.Timedelta(days=STRIP_DAYS - 1)
    days = pd.date_range(first, periods=STRIP_DAYS + 1, freq="D")
    height = STRIP_TOP + STRIP_DAYS * STRIP_ROW + 22

    def x_of(t: pd.Timestamp, day: pd.Timestamp) -> float:
        return GUTTER + PLOT * (t - day).total_seconds() / (24 * HOUR)

    parts = []
    for hour in range(0, 25, 6):
        x = GUTTER + PLOT * hour / 24
        parts.append(vline("gridline", x, 6, height - 20))
        parts.append(tag("text", f"{hour:02d}:00", x=x, y=height - 6, text_anchor="middle"))
    for i, day in enumerate(days[:-1]):
        y = STRIP_TOP + i * STRIP_ROW
        parts.append(tag("text", f"{day:%a %d %b}", x=GUTTER - 10, y=y + STRIP_BAR - 2, text_anchor="end"))
        day_end = min(days[i + 1], now_local)
        for row in spans[(spans["end"] > day) & (spans["start"] < day_end)].itertuples():
            x1, x2 = x_of(max(row.start, day), day), x_of(min(row.end, day_end), day)
            start, ended, length, kind = span_text(row)
            tip = f"{start}, {ended}|{length}|{kind}" if row.running else f"{start} to {ended}|{length}|{kind}"
            css = THREAT_CLASS.get(row.threat, "plain")
            parts.append(tag("rect", cls=css, x=x1, y=y, width=max(x2 - x1, 1.5), height=STRIP_BAR, rx=3))
            wide = max(x2 - x1, 24.0)  # the hover and focus target is wider than a short bar
            parts.append(
                tag("rect", tag("title", esc(tip.replace("|", ", "))), cls="hit", tabindex=0, data_tip=tip,
                    x=(x1 + x2 - wide) / 2, y=y - 5, width=wide, height=STRIP_BAR + 10)
            )
    x_now, y_now = x_of(now_local, days[-2]), STRIP_TOP + (STRIP_DAYS - 1) * STRIP_ROW
    parts.append(vline("axis", x_now, y_now - 4, y_now + STRIP_BAR + 4))
    parts.append(tag("text", "now", cls="quiet", x=x_now + 5, y=y_now + STRIP_BAR - 2))
    label = "Alerts in the last 7 days"
    return tag("svg", "\n".join(parts), viewBox=f"0 0 {WIDTH} {height}", role="img", aria_label=label)


def strip_table(events: pd.DataFrame, as_of: pd.Timestamp) -> str:
    spans = alert_spans(events, as_of)
    spans = spans[spans["end"] > as_of - pd.Timedelta(days=STRIP_DAYS)]
    head = "".join(tag("th", name) for name in ("Start (Kyiv time)", "End", "Length", "Type"))
    rows = [tag("tr", head)]
    for row in spans.itertuples():
        rows.append(tag("tr", "".join(tag("td", esc(cell)) for cell in span_text(row))))
    return tag("table", "\n".join(rows))


def path_data(xs: np.ndarray, ys: np.ndarray, lift: np.ndarray) -> str:
    """SVG path data through the points, lifting the pen wherever `lift` is true.

    A point with no neighbour on either side gets a zero-length stroke, which a round line cap draws as a dot.
    """
    alone = lift & np.append(lift[1:], True)
    commands = []
    for x, y, up, single in zip(xs, ys, lift, alone):
        commands.append(f"{'M' if up else 'L'}{x:.1f},{y:.1f}")
        if single:
            commands.append(f"L{x:.1f},{y:.1f}")
    return wrap(commands, 8)


def replay_svg(h: int, frame: pd.DataFrame, spans: pd.DataFrame, start: pd.Timestamp, weeks: int, simple: str) -> str:
    """Forecast lines for one horizon, one row per test week, over a band showing the alerts."""
    height = REPLAY_TOP + weeks * REPLAY_ROW + 14
    hours = ((frame.index - start) / pd.Timedelta(hours=1)).to_numpy()
    week = (hours // WEEK_HOURS).astype(int)
    xs = GUTTER + PLOT * (hours % WEEK_HOURS) / WEEK_HOURS
    row_top = REPLAY_TOP + week * REPLAY_ROW
    # Lift the pen at the start of a week row and after any gap, which is where an alert was running.
    lift = np.concatenate([[True], (np.diff(hours) > 1) | (np.diff(week) != 0)])
    week_seconds = WEEK_HOURS * HOUR

    def x_of(t: pd.Timestamp, week_start: pd.Timestamp) -> float:
        return GUTTER + PLOT * (t - week_start).total_seconds() / week_seconds

    parts = []
    first_midnight = kyiv(start).normalize() + pd.Timedelta(days=1)
    for day in range(7):
        midnight = first_midnight + pd.Timedelta(days=day)
        x = x_of(midnight, start)
        parts.append(vline("gridline", x, REPLAY_TOP, height - 16))
        parts.append(tag("text", f"{midnight:%a}", x=x + 4, y=height - 3))
    for w in range(weeks):
        y0 = REPLAY_TOP + w * REPLAY_ROW
        week_start = start + pd.Timedelta(hours=w * WEEK_HOURS)
        week_end = week_start + pd.Timedelta(hours=WEEK_HOURS)
        parts.append(hline("gridline", y0))
        parts.append(hline("axis", y0 + REPLAY_PLOT))
        label = f"from {kyiv(week_start):%d %b}"
        parts.append(tag("text", label, x=GUTTER - 10, y=y0 + REPLAY_PLOT // 2 + 4, text_anchor="end"))
        if w == 0:
            parts.append(tag("text", "100%", cls="quiet", x=GUTTER + PLOT + 4, y=y0 + 4))
            parts.append(tag("text", "0%", cls="quiet", x=GUTTER + PLOT + 4, y=y0 + REPLAY_PLOT + 3))
        for row in spans[(spans["end"] > week_start) & (spans["start"] < week_end)].itertuples():
            x1, x2 = x_of(max(row.start, week_start), week_start), x_of(min(row.end, week_end), week_start)
            css = THREAT_CLASS.get(row.threat, "plain")
            band_y = y0 + REPLAY_PLOT + 3
            parts.append(tag("rect", cls=css, x=x1, y=band_y, width=max(x2 - x1, 1.0), height=REPLAY_BAND))
    for css, column in (("simple", simple), ("model", FINAL.name)):
        ys = row_top + REPLAY_PLOT * (1 - frame[column].to_numpy())
        parts.append(f'<path class="{css}" d="{path_data(xs, ys, lift)}"/>')
    parts.append(tag("line", cls="xhair", x1=0, x2=0, y1=0, y2=0))
    parts.append(tag("rect", cls="hit zone", x=GUTTER, y=REPLAY_TOP, width=PLOT, height=weeks * REPLAY_ROW))
    label = f"Forecast and alerts, {h} h horizon"
    return tag("svg", "\n".join(parts), viewBox=f"0 0 {WIDTH} {height}", data_horizon=h, role="img", aria_label=label)


def reliability_table(frame: pd.DataFrame, h: int) -> str:
    """When the model said X%, how often did a new alert follow?"""
    bucket = pd.cut(frame[FINAL.name], BINS, include_lowest=True)
    grouped = frame.groupby(bucket, observed=False)["y"].agg(["size", "mean"])
    head = "".join(tag("th", name) for name in ("Model forecast", "Origins", f"New alert within {h} h"))
    rows = [tag("tr", head)]
    for interval, row in grouped.iterrows():
        share = "none" if row["size"] == 0 else f"{100 * row['mean']:.0f}%"
        label = f"{100 * max(interval.left, 0):.0f}% to {100 * interval.right:.0f}%"
        rows.append(tag("tr", tag("td", label) + tag("td", int(row["size"])) + tag("td", share)))
    return tag("table", "\n".join(rows))


def replay_data(report: dict, predictions: dict[int, pd.DataFrame], start: pd.Timestamp, hours: int) -> str:
    """Per-hour read-out values for the hover layer, as JSON with short lines."""
    blocks = []
    for h, frame in predictions.items():
        simple = report["horizons"][str(h)]["test_period"]["best_simple_forecast"]
        index = ((frame.index - start) / pd.Timedelta(hours=1)).to_numpy().astype(int)
        cells = ["0"] * hours  # 0 marks an hour with an alert running, when no forecast is made
        for i, p, s, y in zip(index, frame[FINAL.name], frame[simple], frame["y"]):
            cells[i] = f"[{p:.2f},{s:.2f},{int(y)}]"
        blocks.append(f'"{h}": [\n{wrap([c + "," for c in cells[:-1]] + cells[-1:], 8)}\n]')
    geometry = {"gutter": GUTTER, "plot": PLOT, "top": REPLAY_TOP, "row": REPLAY_ROW}
    geometry.update({"band": REPLAY_PLOT + REPLAY_BAND + 3, "weeks": hours // WEEK_HOURS, "hours": WEEK_HOURS})
    head = f'"start": {int(epoch_seconds([start])[0]) * 1000},\n"geometry": {json.dumps(geometry)},'
    return "{\n" + head + '\n"horizons": {\n' + ",\n".join(blocks) + "\n}\n}"


def replay_block(events: pd.DataFrame, as_of: pd.Timestamp, report: dict, predictions: dict) -> tuple[str, str]:
    """Markup for the replay, and the JSON its hover layer reads."""
    start, end = pd.Timestamp(report["test_start_utc"]), pd.Timestamp(report["test_end_utc"])
    hours = int((end - start) / pd.Timedelta(hours=1))
    spans = alert_spans(events, as_of)
    tabs, labels, panels = "", "", []
    for h in HORIZONS:
        checked = {"checked": "checked"} if h == HORIZONS[0] else {}
        tabs += tag("input", cls="tab", type="radio", name="h", id=f"h{h}", **checked)
        labels += tag("label", f"Within {h} h", **{"for": f"h{h}"})
        block = report["horizons"][str(h)]["test_period"]
        simple = block["best_simple_forecast"]
        scores = block["forecasts"]
        summary = (
            f"Brier score {scores[FINAL.name]['brier']:.3f} for the model against {scores[simple]['brier']:.3f} "
            f"for the best simple forecast on these weeks ({SIMPLE_NAMES[simple]}). Lower is better."
        )
        legend = tag("span", tag("i", "", cls="key line") + "Model forecast")
        legend += tag("span", tag("i", "", cls="key line grey") + "Best simple forecast")
        chart = replay_svg(h, predictions[h], spans, start, hours // WEEK_HOURS, simple)
        question = "When the model said X%, how often did a new alert follow?"
        panels.append(
            tag(
                "div",
                tag("p", esc(summary), cls="sub")
                + tag("div", legend, cls="legend")
                + tag("div", chart, cls="scroll")
                + tag("details", tag("summary", question) + reliability_table(predictions[h], h)),
                cls=f"panel p{h}",
            )
        )
    markup = tabs + tag("div", labels, cls="tabs") + tag("div", "\n".join(panels), cls="panels")
    return markup, replay_data(report, predictions, start, hours)


def render(events: pd.DataFrame, as_of: pd.Timestamp, live: dict, report: dict, predictions: dict) -> str:
    stamp = kyiv(as_of)
    start, end = (kyiv(pd.Timestamp(report[key])) for key in ("test_start_utc", "test_end_utc"))
    alert_legend = tag(
        "div",
        tag("span", tag("i", "", cls="key drone") + "Drone alert (yellow level)")
        + tag("span", tag("i", "", cls="key missile") + "Missile-type alert (red level)")
        + tag("span", tag("i", "", cls="key plain") + "Alert before levels existed (7 Sep 2026)"),
        cls="legend",
    )
    replay, replay_values = replay_block(events, as_of, report, predictions)
    body = f"""<main>
<h1>Kyiv air raid alerts</h1>
<p class="sub">Snapshot for {stamp:%A %d %B %Y, %H:%M} Kyiv time. Built from alert history only.</p>
{state_block(live)}
{tiles_block(live)}
<section class="card">
<h2>The last 7 days</h2>
<p class="sub">Each bar is one alert, by Kyiv clock time. Hover or tab to a bar for its times.</p>
<div class="scroll">
{strip_block(events, as_of)}
</div>
{alert_legend}
<details><summary>Show these alerts as a table</summary>
{strip_table(events, as_of)}
</details>
</section>
<section class="card">
<h2>Replay of the test weeks: the forecast against what happened</h2>
<p class="sub">Chance of a new alert, forecast at the top of each quiet hour from {start:%d %B} to
{end:%d %B %Y}. Each row is one week and runs from 0% at its baseline to 100% at its top line. The band
under a row shows the alerts that followed; the line breaks while an alert was running.</p>
{replay}
{alert_legend}
</section>
<footer>Generated by <code>python -m airalerts.run</code>. The model is a logistic regression refitted
weekly on the previous {FINAL.train_weeks} weeks; every forecast uses only what was known at its hour.</footer>
</main>
<div id="tip" role="status"></div>"""
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Kyiv alert forecast</title>
<style>{CSS}</style>
</head>
<body>
{reflow(body)}
<script type="application/json" id="replay-data">
{replay_values}
</script>
<script>{SCRIPT}</script>
</body>
</html>
"""


def write(path: Path, events: pd.DataFrame, as_of: pd.Timestamp, live: dict, report: dict, predictions: dict) -> None:
    path.write_text(render(events, as_of, live, report, predictions), encoding="utf-8")
