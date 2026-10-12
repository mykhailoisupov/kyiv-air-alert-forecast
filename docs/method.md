# Method

## Data

- **Source.** The [Kyiv Digital open API](https://kyiv.digital/open-api/air-alert/history) returns every
  alert start and all-clear for Kyiv City since February 2022 and is current to the minute. It needs no
  key. It answers in JSON or XML regardless of the request header, so both are accepted.
- **Cross-check.** On every run the alert table built from the API stream is compared with the city's
  [weekly open-data file][open-data]. All 2,641 published alerts match on start and end time.
- **Scope.** Kyiv City only. It is one alert unit for the whole period, so the 2025 switch from
  oblast-level to raion-level alerts elsewhere in Ukraine does not break the series.
- **Time.** The API gives Kyiv wall-clock times with no offset. They are converted to UTC, with the
  repeated autumn hour resolved by order and the skipped spring hour moved forward. One record fell in a
  repeated hour.
- **Repairs.** In 2022 the stream has 47 repeated starts and 6 repeated all-clears. An all-clear closes
  the start immediately before it, which is the rule the published file follows. The series starts on
  28 February 2022.
- **Causes.** Since the reform of 7 September 2026 each alert lists causes (`drone`, `missile`,
  `ballistic`, `massive-drone`). The list can grow while the alert is running, so it is only used for
  alerts that have already ended.

Raw files are downloaded at run time and are not stored in the repository.

[open-data]: https://data.kyivcity.gov.ua/dataset/statystyka-povitrianykh-tryvoh-u-misti-kyievi-dep-municipal

## Onset forecast

**Target.** At the top of each hour, if no alert is active in Kyiv: will a new alert start within the
next 1, 3 or 6 hours?

Forecasting whether an alert will be *active* is mostly answered by one that is already running: in the
test period, 57% of the hours with an alert at some point in the following hour already had one active.
Conditioning on a quiet city removes that.

**Inputs at forecast time t.** Counts of alert starts in the last 1 hour to 28 days, minutes under alert
in the last 24 hours and 7 days, time since the last start and the last all-clear, the duration and
threat type of the last finished alert, clock hour, and the trailing hour-of-day rate.

**Simple forecasts.** All use only origins whose outcome was known at t:

- Flat 28-day rate.
- Hour-of-day rate over 28 days, shrunk towards the flat rate. This was fixed in advance as the
  reference for skill.
- Hour-of-day rate over all history.
- Recent-onset rate: the 28-day rate conditional on whether an alert started in the previous H hours.

**Model.** Logistic regression, refitted every week on the previous 16 weeks.

**Protocol.** The test period is the latest eight weeks, 17 August to 12 October 2026: 940 quiet hourly
origins over 57 days. Each weekly fold trains only on origins whose label window closed before the fold
starts. Scores are the Brier score and Brier skill, with 95% intervals from resampling whole days
1,000 times.

## Leakage checks

Each check is a test in `tests/`.

- **A feature looks past t.** Features are recomputed with every later event removed and must not
  change, on synthetic and real data (`test_features_do_not_change_when_the_future_is_hidden`).
- **The check itself is blind.** A deliberately leaky feature is added and the check must flag exactly
  that one (`test_a_feature_that_peeks_ahead_is_caught`).
- **A training label overlaps the test week.** `test_training_labels_close_before_each_test_fold`.
- **A simple forecast uses an outcome not yet known.** Every still-open label is flipped and the
  forecast must stay the same (`test_reference_forecasts_ignore_labels_that_have_not_closed`).
- **A running alert's end or final cause list leaks.** `truncate` blanks both, and the tests above run
  through it.
- **The live forecast differs from the backtest.** For a past instant the two must give the same number
  (`test_live_forecast_equals_the_backtest_forecast_for_the_same_instant`).
- **Text, the remaining-time estimate and the page.** Each has its own hide-the-future test in
  `test_text.py`, `test_remaining.py` and `test_page.py`.
- **Tuning on test data.** The scaler is fitted on training rows only. The window, inputs and reference
  were fixed before the first test run, and each later experiment was designed first and scored once.

## Live forecast

`python -m airalerts.now` fits the same model on every quiet hour of the last 16 weeks whose outcome is
already known, then forecasts from the current minute. The best simple forecast shown beside it is the
simple rule with the lowest Brier score over the last 8 weeks of known outcomes.

## Time left in a running alert

Fixed before scoring:

- At instant t, for an alert that has run e minutes: take the alerts that had ended by t and started in
  the previous 16 weeks, keep those that lasted longer than e, and read the 10th, 50th and 90th
  percentiles of what was left of them. The middle value is the estimate; the outer two are the 80%
  interval.
- If fewer than 30 such alerts exist, use every alert that had ended by t.
- Baseline: the same percentiles of full durations, ignoring minutes elapsed.
- Backtest: every 15 minutes of the test period at which an alert was running.

## Text experiment

- **Source.** 24,800 posts from the Ukrainian Air Force channel [@kpszsu](https://t.me/s/kpszsu),
  26 April to 12 October 2026, read from the public web preview.
- **Features at t.** Counts of posts published strictly before t: all posts in the last 60 and 180
  minutes, posts mentioning Kyiv or "the capital" in the last 15, 30, 60 and 180 minutes, posts with
  missile or strategic-aviation keywords in the last 60 and 180 minutes, and minutes since the last
  Kyiv mention. The keyword groups and windows were fixed before any result was produced.
- **Edited posts.** 2% of posts are marked edited. They keep their timestamp but their text is ignored,
  because the text shown now may have been written after publication.
- **Test.** The final model with and without these features, on the same folds, at hourly and at
  15-minute origins.

## Status page

`airalerts.run` writes `outputs/page.html`: inline CSS and SVG, a short inline script for hover
read-outs, light and dark themes, and no network requests. `docs/page.png` is a browser screenshot of it;
`docs/hero.png` and `docs/replay.png` are crops of that screenshot.

## Layout

```
src/airalerts/
  data.py       download, clean, cross-check -> alert table in UTC
  grid.py       as-of features and labels on a time grid
  baselines.py  simple reference forecasts
  model.py      the final model and the challengers
  evaluate.py   walk-forward folds, scores, day-block bootstrap
  now.py        live forecast for one instant
  remaining.py  time left in a running alert
  page.py       the status page
  describe.py   descriptive tables and figures
  report.py     evaluation figures
  telegram.py   backfill of a public Telegram channel
  text.py       text features and the with/without-text test
  run.py        entry point for the whole pipeline
tests/          data cleaning and leakage tests
outputs/        metrics, tables, figures and the page for the snapshot of 12 October 2026, 18:00 UTC
docs/           method, results, and the page screenshots
```

## Reproduce

```
uv sync
uv run pytest
uv run python -m airalerts.run --as-of 2026-10-12T18:00:00Z --sensitivity
```

`--as-of` discards everything after the given UTC time, so this reproduces every number in
[results.md](results.md) from a fresh download. Without it, the run uses all data available at download
time.

The text experiment needs the channel backfill first, which takes about half an hour at one request per
second. Posts can be edited or deleted later, so a new backfill may differ slightly.

```
uv run python -m airalerts.telegram kpszsu --since 2026-04-27T00:00:00Z
uv run python -m airalerts.text --as-of 2026-10-12T18:00:00Z
```
