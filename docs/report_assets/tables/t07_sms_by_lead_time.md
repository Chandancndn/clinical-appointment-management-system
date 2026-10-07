**Table 7 (E4). SMS reminder against no-show rate within each lead-time band (Kaggle training part)**

| Lead time (days) | SMS received | Appointments | No-show rate (95% Wilson interval) | Share of band with SMS | Difference in no-show rate, SMS minus none (percentage points) |
|---:|---|---:|---:|---:|---:|
| 0 (same day) | No | 30,982 | 4.8% (4.6% to 5.0%) | 0.0% | n/a |
| 1-3 | No | 10,923 | 23.3% (22.5% to 24.1%) | 4.5% | -2.0 |
| 1-3 | Yes | 517 | 21.3% (18.0% to 25.0%) | 4.5% | -2.0 |
| 4-7 | No | 6,379 | 27.3% (26.2% to 28.4%) | 55.6% | -2.7 |
| 4-7 | Yes | 7,985 | 24.5% (23.6% to 25.5%) | 55.6% | -2.7 |
| 8-14 | No | 4,689 | 33.8% (32.5% to 35.2%) | 53.6% | -5.4 |
| 8-14 | Yes | 5,409 | 28.4% (27.2% to 29.6%) | 53.6% | -5.4 |
| 15-30 | No | 6,089 | 37.2% (36.0% to 38.5%) | 54.6% | -6.2 |
| 15-30 | Yes | 7,311 | 31.1% (30.0% to 32.1%) | 54.6% | -6.2 |
| 31+ | No | 3,645 | 37.8% (36.2% to 39.3%) | 55.8% | -8.1 |
| 31+ | Yes | 4,598 | 29.6% (28.3% to 31.0%) | 55.8% | -8.1 |

Source: `results/e4_sms_crosstab.csv`. An association only: reminders were not sent at random, so this does not show that the SMS causes the difference.
