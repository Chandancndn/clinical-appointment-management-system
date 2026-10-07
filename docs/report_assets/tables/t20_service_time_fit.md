**Table 20 (S1). Consultation-time distributions fitted to the Hangu data (minutes)**

| Distribution | Parameters | Log-likelihood | AIC | KS statistic | Fitted mean (minutes) | Chosen by AIC |
|---|---|---:|---:|---:|---:|---|
| Lognormal | mu = 2.496, sigma = 0.438 | -20,511.8 | 41,027.6 | 0.0072 | 13.36 | Yes |
| Gamma | shape = 5.339, scale = 2.503 | -20,632.2 | 41,268.5 | 0.0347 | 13.37 | No |

Source: `results/sim_service_time_fit.csv`. Observed mean 13.37 minutes over 6,637 consultations; median 12.1 minutes, so the default slot is 15 minutes and the session 240 minutes.
