# This repository contains the source code for the article “A Comparative Study of Short-Term Memory in Musicians and Nonmusicians.”

 

## Requirements

Python 3.10 or newer. Install the listed libraries from the workspace root:

    python -m pip install -r Python_code/requirements.txt

## Run

Analyse the existing clean data:

    python Python_code/run_analysis.py

Rebuild the corrected and scored data from the raw exports, then analyse them:

    python Python_code/run_analysis.py --rebuild-data

The default output directory is Python_code/output. It is separate from the R project's data, plots, tables, and reports directories. To reduce run time, use --simulations 0 to skip Monte Carlo post-hoc power analysis or --skip-mixed-models to skip the mixed-effects regressions. --input-clean and --input-full can point to alternate CSV files.

## Modules

- preprocessing.py reads JATOS/LimeSurvey exports and the participant ID workbook, standardises IDs/demographics, applies the dictionary's kept columns, scores memory tasks and questionnaires, applies participant rules, and writes full and main clean CSVs.
- analysis.py produces the primary and sensitivity REML meta-analyses, exploratory meta-analyses, within-group correlations, outcome plots, a-priori and simulated power tables, and the supplementary figure based on the MAT file.
- models.py fits full-sample and group-specific linear mixed-effects models with statsmodels.

## Analysis notes

- Positive Hedges' g means higher scores for musicians. Site effects use the musician and nonmusician means, sample SDs, and sample sizes; pooled estimates use REML.
- The main clean dataset retains all matched participants. The separate sensitivity analysis uses the to_keep flag, including the rule that a participant's matched partner is also excluded when either member fails an eligibility criterion.
- MixedLM is the closest standard Python counterpart to lme4::lmer. Its optimizer, p-values, random-effect handling, and AIC-based blockwise backward selection are not identical to lmerTest::step; exact model coefficients and selected terms can therefore differ. The Python model tables report Wald z tests and identify failed or skipped fits.
- The Monte Carlo power table uses the observed number of participants per group at each site and an REML pooled test. Set a seed with --seed for repeatable simulation output.

The pipeline reads the existing Excel workbooks but writes generated data as CSV. Tables are CSV (including dataset_counts.csv), figures are PNG/PDF/SVG, and the concise analysis report is Markdown.


