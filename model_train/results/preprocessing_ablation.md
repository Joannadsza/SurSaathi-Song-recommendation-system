| Experiment | Accuracy | Macro-F1 | Weighted-F1 |
|---|---|---|---|
| R0 Roman: raw (lowercase only) | 0.597 ± 0.038 | 0.363 ± 0.054 | 0.555 ± 0.042 |
| R1 Roman: cleaned (current baseline) | 0.593 ± 0.035 | 0.372 ± 0.046 | 0.554 ± 0.037 |
| R2 R1 + spelling variants (rules+dict+mined) | 0.590 ± 0.030 | 0.376 ± 0.040 | 0.551 ± 0.032 |
| R3 R2 + suffix stemming | 0.592 ± 0.029 | 0.378 ± 0.046 | 0.553 ± 0.033 |
| R4 R1 + repetition handling (dedupe lines/runs) | 0.587 ± 0.040 | 0.353 ± 0.073 | 0.541 ± 0.048 |
| R5 R1 + data-driven stopwords (max_df=0.3) | 0.596 ± 0.033 | 0.377 ± 0.049 | 0.556 ± 0.037 |
| R6 all Roman steps combined | 0.573 ± 0.033 | 0.354 ± 0.058 | 0.531 ± 0.036 |
| D0 Devanagari: raw | 0.594 ± 0.027 | 0.359 ± 0.044 | 0.551 ± 0.034 |
| D1 Devanagari: cleaned | 0.593 ± 0.029 | 0.375 ± 0.045 | 0.553 ± 0.035 |
| D2 D1 + suffix stemming | 0.598 ± 0.028 | 0.378 ± 0.046 | 0.558 ± 0.033 |
| D3 D2 + repetition handling | 0.587 ± 0.032 | 0.353 ± 0.062 | 0.542 ± 0.040 |
| X1 Roman(all) + Devanagari(stem) | 0.599 ± 0.032 | 0.380 ± 0.048 | 0.557 ± 0.036 |
| R1 + metadata | 0.652 ± 0.035 | 0.526 ± 0.029 | 0.642 ± 0.032 |
| R6 + metadata | 0.637 ± 0.027 | 0.518 ± 0.037 | 0.626 ± 0.025 |
| X1 + metadata | 0.659 ± 0.022 | 0.533 ± 0.028 | 0.647 ± 0.020 |
