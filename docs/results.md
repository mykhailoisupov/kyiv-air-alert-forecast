# Results

Everything here is for the snapshot of 12 October 2026, 18:00 UTC. The numbers come from
`outputs/metrics.json` and `outputs/tables/`.

## The alerts

![Alerts and hours under alert per month](../outputs/figures/monthly.png)

![Alert starts and time under alert by clock hour](../outputs/figures/hour_profile.png)

![Alert durations and quiet gaps](../outputs/figures/durations_gaps.png)

| Period | Days | Alerts | Per day | Under alert | Median length | 90th pct | Median gap | Night | Burst |
|---|---|---|---|---|---|---|---|---|---|
| 2022 (from 28 Feb) | 307 | 592 | 1.93 | 8.4% | 49 min | 127 min | 5.3 h | 29% | 0.19 |
| 2023 to 2025 | 1,096 | 1,313 | 1.20 | 6.4% | 37 min | 188 min | 11.2 h | 40% | 0.11 |
| Jan to Jul 2026 | 212 | 336 | 1.59 | 5.8% | 28 min | 113 min | 7.0 h | 37% | 0.08 |
| Aug 2026 onward | 73 | 404 | 5.54 | 24.4% | 35 min | 149 min | 1.4 h | 34% | 0.19 |

"Under alert" is the share of time with an alert active. "Median gap" is the quiet time between one
alert ending and the next starting. "Night" is the share of alerts starting between 23:00 and 07:00.
"Burst" is the Goh-Barabási coefficient of the times between alert starts: 0 is what a constant-rate
random process gives, and values towards 1 mean alerts arrive in clusters. Kyiv's alerts are mildly
clustered in every period.

- September 2026 had 190 alert starts and 243 hours under alert, against 190 and 213 in March 2022.
- From 2023 to July 2026, an hour around 01:00 to 03:00 was under alert about 11% to 15% of the time and
  an afternoon hour 2% to 5%. Since August 2026 every clock hour sits between 12% and 33%.
- Since the reform of 7 September 2026, 220 finished alerts were drone-only (yellow level), with a
  median of 34 minutes, and 25 involved a missile, ballistic or massive-drone cause (red level), with a
  median of 47 minutes.

## Onset forecast on the latest eight weeks

Brier score, with skill over the hour-of-day reference in brackets. Lower Brier is better. The base rate
of a new alert was 33% within 1 h, 60% within 3 h and 75% within 6 h.

| Forecast | 1 h | 3 h | 6 h |
|---|---|---|---|
| Flat 28-day rate | 0.225 (+1.9%) | 0.244 (+2.1%) | 0.191 (+1.0%) |
| Hour-of-day rate, 28 days (reference) | 0.230 | 0.250 | 0.193 |
| Hour-of-day rate, all history | 0.292 (-27.4%) | 0.435 (-74.2%) | 0.410 (-112.4%) |
| Recent-onset rate, 28 days | 0.225 (+1.8%) | 0.225 (+10.1%) | 0.165 (+14.4%) |
| Logistic, all history (first attempt) | 0.210 (+8.4%) | 0.215 (+14.0%) | 0.154 (+20.2%) |
| **Logistic, last 16 weeks (final)** | 0.218 (+5.0%) | 0.217 (+13.1%) | 0.157 (+18.7%) |
| Boosted trees, last 16 weeks | 0.250 (-9.0%) | 0.277 (-11.1%) | 0.236 (-22.3%) |
| Boosted trees in a sigmoid calibration wrapper (first run) | 0.270 (-17.8%) | 0.362 (-45.1%) | 0.358 (-85.5%) |

Final model, with 95% intervals:

| Period and yardstick | 1 h | 3 h | 6 h |
|---|---|---|---|
| 8 weeks, vs hour-of-day reference | +5.0% (-0.5 to +10.4) | +13.1% (+5.6 to +20.9) | +18.7% (+10.3 to +28.2) |
| 8 weeks, vs best simple forecast | +3.2% (-2.5 to +8.5) | +3.3% (-4.7 to +11.1) | +5.1% (-4.6 to +13.6) |
| Since reform, vs hour-of-day reference | +4.6% (-2.1 to +11.2) | +11.8% (+1.4 to +24.6) | +24.4% (+10.4 to +41.4) |
| Since reform, vs best simple forecast | +2.3% (-4.0 to +8.7) | +10.2% (+0.2 to +22.9) | +11.8% (+1.2 to +19.9) |

The best simple forecast is whichever simple rule scored best on that period, chosen after the fact, so
it is a deliberately hard bar.

![Brier skill over the best simple forecast](../outputs/figures/skill.png)

![Reliability of the final and first models](../outputs/figures/reliability.png)

- The hour-of-day reference turned out weaker than a flat rate, which fits the vanished day and night
  pattern. Skill over it overstates what the model adds.
- Over the full eight weeks the first attempt, trained on all history, scores slightly better than the
  final model at every horizon. Since the reform it does not: at 6 h it falls 6% behind the recent-onset
  rule, while the final model stays 12% ahead.
- The first attempt forecasts too low. Since the reform its average 6 h forecast is 74% against an
  observed 87%; the final model averages 82%.
- Calibration bin by bin is rough for both models, and the final model's highest 1 h forecasts were too
  high.

### Correction: boosted trees

An earlier version of this project said boosted trees "failed outright" because they "cannot extrapolate
beyond the alert rates seen in training". That explanation was never tested, and a review showed it was
mostly wrong.

- The trees were first run inside a sigmoid calibration wrapper (`CalibratedClassifierCV`, three folds).
  At 1 h that version scored a Brier of 0.270 with no ranking ability (ROC-AUC 0.51) and an average
  forecast of 12% against an observed 33%.
- The same trees without the wrapper score 0.250, rank origins about as well as the logistic model
  (ROC-AUC 0.63 for both) and average 27%.
- Without the wrapper the trees still trail the hour-of-day reference by 9% to 22%, with intervals at
  1 h and 3 h that include zero, and they still forecast too low.

Both versions stay in the table. The trees were not tuned, so this says little about boosted trees in
general.

### Sensitivity to the training window

Run after the headline numbers were fixed. Skill over the best simple forecast:

| Training window | 8 wk, 1 h | 8 wk, 3 h | 8 wk, 6 h | Reform, 1 h | Reform, 3 h | Reform, 6 h |
|---|---|---|---|---|---|---|
| Last 8 weeks | +3.1% | +0.9% | -6.8% | +0.4% | +6.6% | +4.6% |
| Last 16 weeks | +3.4% | +3.6% | +5.8% | +2.7% | +9.8% | +12.5% |
| Last 16 weeks + hour-of-day rate (final) | +3.2% | +3.3% | +5.1% | +2.3% | +10.2% | +11.8% |
| Last 32 weeks | +5.8% | +3.6% | +1.4% | +4.8% | +8.8% | +2.6% |
| All history | +6.6% | +4.3% | +6.9% | +1.7% | +0.8% | -6.1% |

The 16- and 32-week windows stay positive in every cell; the shortest and the longest do not. Feeding in
the hour-of-day rate makes no real difference at 16 weeks and hurts at 8 weeks and on all history (full
table in `outputs/tables/sensitivity.csv`).

## Time left in a running alert

Scored once, at every 15 minutes of the test period with an alert running: 1,620 origins inside 331
alerts. Since the reform: 1,134 origins inside 219 alerts.

| Period | Estimate | 80% interval covered | Mean error of the middle value | Mean width |
|---|---|---|---|---|
| 8 weeks | Uses minutes elapsed | 74.8% (69.7 to 79.4) | 70.9 min (53.9 to 92.1) | 181 min |
| 8 weeks | Ignores minutes elapsed | 62.4% (57.0 to 67.6) | 75.1 min (55.7 to 98.3) | 113 min |
| Since reform | Uses minutes elapsed | 74.1% (67.5 to 80.7) | 78.4 min (53.9 to 103.9) | 189 min |
| Since reform | Ignores minutes elapsed | 63.2% (55.9 to 70.8) | 83.8 min (55.6 to 112.2) | 124 min |

- Using minutes elapsed brings the 80% interval closer to its target: it covered 75% of true remaining
  times, against 62% for the baseline. Its intervals are also wider, about three hours on average
  against about two.
- The middle value is off by about 71 minutes on average, 5.6% better than the baseline (95% interval
  2.0% to 8.2%). That is a small gain on a large error.
- Part of why the error looks large next to a 35-minute median alert is the sampling. Origins taken
  every 15 minutes weight alerts by their length: a three-hour alert contributes twelve origins, a
  ten-minute one at most one.

## Text experiment

Brier skill of the final model with text features over the same model without them:

| Origins | Horizon | Latest 8 weeks | Since 7 Sep reform |
|---|---|---|---|
| Hourly | 1 h | +4.0% (+0.4 to +7.3) | +1.7% (-3.5 to +6.3) |
| Hourly | 3 h | +3.0% (-0.3 to +6.5) | -1.2% (-4.4 to +2.3) |
| Hourly | 6 h | +1.8% (-2.2 to +5.7) | -3.4% (-9.2 to +3.4) |
| Every 15 minutes | 1 h | +4.2% (+1.1 to +6.7) | +1.7% (-2.4 to +5.6) |

Text adds about 4% at one hour over the eight weeks, and that interval excludes zero in both designs.
Nothing is measurable at 3 h or 6 h, and since the reform no interval excludes zero. The gain is real but
small, so the headline model stays on alert history alone.

Two caveats. The posts were collected after the fact, so posts deleted since are missing. And keyword
matching is crude: a post about a town near Kyiv that does not name Kyiv is not counted. Structured
extraction of threat type, direction and target from each post, for example with an LLM used only as a
parser, is the natural next step and was not run here.

## Limits

- One city, eight weeks, and a test period that sits inside a surge. The intervals are wide and most
  margins over the best simple forecast include zero.
- The alert-level reform is five weeks old; any conclusion about yellow and red alerts rests on 25
  red-type alerts.
- Alerts are a response to an adversary. A pattern that holds for eight weeks can end overnight, and the
  model only adapts after the change shows up in the data.
- Hourly origins hide anything faster than an hour, including how alerts spread between regions.
- The headline forecast uses alert history only. Text was tested with keyword counts from one channel;
  launch reports and weather were not used.
- Boosted trees and the remaining-time estimate were each run with one untuned design.
