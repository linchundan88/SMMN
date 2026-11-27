# Overall musician vs nonmusician analysis

Input: `\\192.168.32.14\code\music_ensemble\Data and analyses scripts\Python_code\output\data\clean\dat-clean.csv`
Combined participants: 1,200; musicians: 600; nonmusicians: 600.

All eligible participant rows were pooled before analysis. Laboratory membership was not used in grouping, weighting, adjustment, or testing. The primary test for each outcome is a two-sided Welch independent-samples t test of H₀: the overall group means are equal. Holm adjustment controls family-wise error across the three primary outcomes. Positive mean differences and positive Hedges' g values favour musicians.

| Outcome | Musicians n, mean (SD) | Nonmusicians n, mean (SD) | Mean difference (95% CI) | t(df) | p | Holm p | Cohen's d | Hedges' g (approx. 95% CI) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Music short-term memory | 600, 27.62 (5.98) | 600, 21.48 (5.11) | 6.14 [5.51, 6.77] | 19.13 (1169.6) | 3.63e-71 | 1.09e-70 | 1.104 | 1.104 [0.982, 1.225] (large) |
| Verbal short-term memory | 600, 10.19 (2.29) | 600, 9.84 (2.05) | 0.36 [0.11, 0.60] | 2.85 (1183.3) | 0.00442 | 0.00442 | 0.165 | 0.165 [0.051, 0.278] (negligible) |
| Visuospatial short-term memory | 600, 9.07 (2.05) | 600, 8.53 (1.72) | 0.54 [0.33, 0.75] | 4.94 (1162.5) | 9.09e-07 | 1.82e-06 | 0.285 | 0.285 [0.171, 0.399] (small) |

The confidence interval for the raw mean difference uses the Welch-Satterthwaite degrees of freedom. Cohen's d is the pooled-SD standardized mean difference; Hedges' g applies the small-sample correction. The displayed standardized-effect confidence intervals are approximate Wald intervals. Magnitude labels use conventional thresholds as a rough guide, not as a significance test.

## Output files

- `combined_data.csv`: pooled participant rows used by the analysis, with a normalized `analysis_group` column.
- `group_descriptives.csv`: overall sample size, mean, and SD by outcome and group.
- `overall_tests.csv`: pooled Welch tests, confidence intervals, Holm-adjusted p values, and effect sizes.
- `effect_sizes.csv`: Cohen's d and Hedges' g with approximate confidence intervals.
