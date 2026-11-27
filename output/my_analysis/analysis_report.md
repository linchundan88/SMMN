# Musician vs nonmusician: multi-lab analysis

Input: `\\192.168.32.14\code\music_ensemble\Data and analyses scripts\Python_code\output\data\clean\dat-clean.csv`
Participants with recognised group and lab: 1,200; laboratories: 33; musicians: 600; nonmusicians: 600.

## Primary hypothesis tests

For each outcome, laboratory-specific Hedges' g values were combined using a random-effects meta-analysis estimated by REML. The pooled two-sided z test evaluates H₀: mean Hedges' g = 0 against H₁: mean Hedges' g ≠ 0. Positive values favour musicians. Holm-adjusted p values control family-wise error across the three primary outcomes.

| Outcome | Labs | Hedges' g | 95% CI | z | p | Holm p | I² |
|---|---:|---:|---:|---:|---:|---:|---:|
| Music short-term memory | 33 | 1.080 | [0.940, 1.220] | 15.11 | 1.39e-51 | 4.16e-51 | 28.1% |
| Verbal short-term memory | 33 | 0.161 | [0.019, 0.303] | 2.21 | 0.0268 | 0.0268 | 39.7% |
| Visuospatial short-term memory | 33 | 0.282 | [0.154, 0.410] | 4.32 | 1.53e-05 | 3.06e-05 | 17.0% |

## Laboratory-level comparisons

`lab_comparisons.csv` contains each lab's sample sizes, means, SDs, raw mean difference (musician minus nonmusician), Welch independent-samples t test, and Hedges' g. Lab-level tests are exploratory and their p values are not multiplicity-adjusted. The primary cross-lab inference is the pooled REML result above.

## Files

- `lab_comparisons.csv`: laboratory-by-outcome descriptive and Welch test results.
- `pooled_meta_analysis.csv`: REML estimates, hypothesis tests, and heterogeneity.
