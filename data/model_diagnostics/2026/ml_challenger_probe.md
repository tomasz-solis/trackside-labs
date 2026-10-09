# ML challenger probe

14 races, 308 driver rows per replay root. Each race is predicted from earlier races only, at the last checkpoint before qualifying. Scored from round 5, mean over 2 replay seeds.

| Target | Model | Rank corr | MAE | Top 3 |
|---|---|---|---|---|
| Qualifying | Live model | 0.930 | 1.76 | 53% |
| Qualifying | Ridge, pace features only | 0.914 | 1.75 | 47% |
| Qualifying | Boosting, pace features only | 0.914 | 1.90 | 43% |
| Qualifying | Ridge, pace + live forecast | 0.911 | 1.74 | 57% |
| Qualifying | Boosting, pace + live forecast | 0.915 | 1.95 | 47% |
| Qualifying | Ridge, 7 simple features | 0.935 | 1.64 | 57% |
| Qualifying | 50/50 blend live + ridge | 0.929 | 1.68 | 57% |
| Race | Live model | 0.721 | 3.07 | 60% |
| Race | Ridge, pace features only | 0.651 | 3.62 | 50% |
| Race | Boosting, pace features only | 0.566 | 4.26 | 33% |
| Race | Ridge, pace + live forecast | 0.674 | 3.50 | 58% |
| Race | Boosting, pace + live forecast | 0.588 | 4.16 | 45% |
| Race | Ridge, 7 simple features | 0.664 | 3.50 | 50% |
| Race | 50/50 blend live + ridge | 0.725 | 3.04 | 62% |

Simple ridge vs live model, MAE change per race (negative = better):

| Target and seed | MAE change | 95% CI | Races better / worse |
|---|---|---|---|
| Qualifying Z_base_32_s42 | -0.168 | [-0.384, +0.050] | 6 / 4 |
| Race Z_base_32_s42 | +0.446 | [+0.218, +0.691] | 0 / 9 |
| Qualifying Z_base_32_s43 | -0.141 | [-0.372, +0.087] | 6 / 4 |
| Race Z_base_32_s43 | +0.447 | [+0.165, +0.721] | 1 / 9 |

Feature group share of importance by round (pace-only ridge) is in `ml_challenger_probe.json` and the chart `ml_challenger_probe.html`.

Qualifying, last round (Spanish Grand Prix): Recent results (driver and team form) 47%, Practice car traits (corners, top speed, deg) 25%, Measured pace history (race pace, quali gap) 14%, Practice pace (short and long run) 9%, Practice teammate split 3%, Car traits x track 1%.

Race, last round (Spanish Grand Prix): Recent results (driver and team form) 49%, Practice car traits (corners, top speed, deg) 20%, Practice pace (short and long run) 15%, Measured pace history (race pace, quali gap) 11%, Practice teammate split 3%, Car traits x track 2%.
