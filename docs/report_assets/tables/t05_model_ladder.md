**Table 5 (E3). Does each step up the ladder beat the one before? Paired differences on the holdout**

| Step | Change in AUC-ROC | Change in average precision | Change in sensitivity at 40% precision | Clearly better? |
|---|---:|---:|---:|---|
| Earlier no-show rate rule → Logistic regression | +0.200 (+0.190 to +0.210) | +0.118 (+0.110 to +0.128) | +0.029 (+0.001 to +0.103) | Yes |
| Logistic regression → Random forest | +0.012 (+0.007 to +0.018) | +0.019 (+0.008 to +0.029) | +0.074 (-0.006 to +0.196) | Yes |
| Random forest → Gradient boosting | +0.002 (-0.001 to +0.004) | +0.006 (+0.001 to +0.011) | +0.015 (-0.056 to +0.079) | No |

Source: `results/e3_ladder.csv`. Clearly better means the intervals for both AUC-ROC and average precision lie above zero.
