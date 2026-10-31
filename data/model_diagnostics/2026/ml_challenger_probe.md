# ML challenger probe

17 races, 374 driver rows per replay root. Each race is predicted from earlier races only, at the last checkpoint before qualifying. Scored from round 5, mean over 2 replay seeds.

| Target | Model | Rank corr | MAE | Top 3 |
|---|---|---|---|---|
| Qualifying | Live model | 0.887 | 2.12 | 56% |
| Qualifying | Ridge, pace features only | 0.887 | 2.03 | 49% |
| Qualifying | Boosting, pace features only | 0.890 | 2.10 | 44% |
| Qualifying | Ridge, pace + live forecast | 0.874 | 2.08 | 54% |
| Qualifying | Boosting, pace + live forecast | 0.886 | 2.15 | 47% |
| Qualifying | Ridge, 7 simple features | 0.899 | 1.98 | 51% |
| Qualifying | 50/50 blend live + ridge | 0.894 | 1.98 | 58% |
| Race | Live model | 0.642 | 3.69 | 45% |
| Race | Ridge, pace features only | 0.660 | 3.64 | 49% |
| Race | Boosting, pace features only | 0.592 | 4.20 | 38% |
| Race | Ridge, pace + live forecast | 0.658 | 3.62 | 54% |
| Race | Boosting, pace + live forecast | 0.585 | 4.21 | 36% |
| Race | Ridge, 7 simple features | 0.659 | 3.63 | 45% |
| Race | 50/50 blend live + ridge | 0.662 | 3.51 | 50% |

Simple ridge vs live model, MAE change per race (negative = better):

| Target and seed | MAE change | 95% CI | Races better / worse |
|---|---|---|---|
| Qualifying C35_adjust_off_s42 | -0.167 | [-0.326, -0.018] | 8 / 4 |
| Race C35_adjust_off_s42 | -0.057 | [-0.198, +0.077] | 7 / 6 |
| Qualifying C35_adjust_off_s43 | -0.161 | [-0.311, -0.014] | 7 / 3 |
| Race C35_adjust_off_s43 | -0.062 | [-0.232, +0.095] | 7 / 4 |

Feature group share of importance by round (pace-only ridge) is in `ml_challenger_probe.json` and the chart `ml_challenger_probe.html`.

Qualifying, last round (Singapore Grand Prix): Recent results (driver and team form) 47%, Practice car traits (corners, top speed, deg) 25%, Measured pace history (race pace, quali gap) 15%, Practice pace (short and long run) 7%, Practice teammate split 3%, Track and conditions 1%.

Race, last round (Singapore Grand Prix): Recent results (driver and team form) 44%, Practice car traits (corners, top speed, deg) 25%, Practice pace (short and long run) 15%, Measured pace history (race pace, quali gap) 11%, Practice teammate split 3%, Car traits x track 2%.
