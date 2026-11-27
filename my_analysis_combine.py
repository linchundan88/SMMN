"""Pooled musician vs nonmusician analysis across all participants.

This script combines eligible participant rows from the clean CSV and runs
one overall comparison per primary memory outcome. Laboratory membership is
not used to stratify, adjust, weight, or model the results.

The hypothesis test is a two-sided Welch independent-samples t test:
H0: the overall musician and nonmusician population means are equal.

Run from the workspace root:
    python Python_code/my_analysis_combine.py
    python Python_code/my_analysis_combine.py --input path/to/dat-clean.csv
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import pandas as pd
from scipy.stats import t


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT = SCRIPT_DIR / "output" / "data" / "clean" / "dat-clean.csv"
DEFAULT_OUTPUT = SCRIPT_DIR / "output" / "my_analysis_combine"

PRIMARY_OUTCOMES = {
    "melody_span": "Music short-term memory",
    "digit_span": "Verbal short-term memory",
    "spatial_span": "Visuospatial short-term memory",
}
GROUPS = ("musician", "nonmusician")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Pool all participants and compare musicians with nonmusicians "
            "without laboratory-level adjustment or meta-analysis."
        )
    )
    parser.add_argument(
        "--input", type=Path, default=DEFAULT_INPUT,
        help=f"Clean participant-level CSV (default: {DEFAULT_INPUT}).",
    )
    parser.add_argument(
        "--output", type=Path, default=DEFAULT_OUTPUT,
        help=f"Directory for pooled data, tables, and report (default: {DEFAULT_OUTPUT}).",
    )
    return parser.parse_args()


def _normalise_group(value: object) -> str | None:
    if pd.isna(value):
        return None
    key = "".join(ch for ch in str(value).casefold().strip() if ch.isalnum())
    if key in {"musician", "musicians"}:
        return "musician"
    if key in {"nonmusician", "nonmusicians"}:
        return "nonmusician"
    return None


def _load_and_combine(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(
            f"Input CSV not found: {path}\n"
            "Run `python Python_code/run_analysis.py --rebuild-data` first, "
            "or supply a clean CSV with --input."
        )
    data = pd.read_csv(path, low_memory=False)
    required = {"group", *PRIMARY_OUTCOMES}
    missing = sorted(required.difference(data.columns))
    if missing:
        raise ValueError("Input CSV is missing required columns: " + ", ".join(missing))

    data = data.copy()
    data["analysis_group"] = data["group"].map(_normalise_group)
    data = data.loc[data["analysis_group"].isin(GROUPS)].copy()
    if data.empty:
        raise ValueError("No rows have a recognised musician/nonmusician group.")
    for outcome in PRIMARY_OUTCOMES:
        data[outcome] = pd.to_numeric(data[outcome], errors="coerce")
    return data.reset_index(drop=True)


def _welch_comparison(musician: pd.Series, nonmusician: pd.Series) -> dict[str, float | int]:
    """Welch test and confidence interval for the pooled raw mean difference."""
    n1, n0 = len(musician), len(nonmusician)
    if min(n1, n0) < 2:
        raise ValueError("Each group needs at least two non-missing scores per outcome.")

    mean1, mean0 = float(musician.mean()), float(nonmusician.mean())
    difference = mean1 - mean0
    variance_term1 = float(musician.var(ddof=1)) / n1
    variance_term0 = float(nonmusician.var(ddof=1)) / n0
    standard_error = math.sqrt(variance_term1 + variance_term0)
    denominator = variance_term1**2 / (n1 - 1) + variance_term0**2 / (n0 - 1)
    degrees_freedom = (variance_term1 + variance_term0) ** 2 / denominator if denominator > 0 else math.nan
    if standard_error > 0 and math.isfinite(degrees_freedom):
        statistic = difference / standard_error
        p_value = float(2 * t.sf(abs(statistic), degrees_freedom))
        critical_value = float(t.ppf(0.975, degrees_freedom))
        margin = critical_value * standard_error
        difference_ci_low, difference_ci_high = difference - margin, difference + margin
    else:
        statistic = p_value = difference_ci_low = difference_ci_high = math.nan

    pooled_sd = math.sqrt(
        ((n1 - 1) * float(musician.var(ddof=1))
         + (n0 - 1) * float(nonmusician.var(ddof=1)))
        / (n1 + n0 - 2)
    )
    if pooled_sd > 0:
        cohens_d = difference / pooled_sd
        variance_d = (n1 + n0) / (n1 * n0) + cohens_d**2 / (2 * (n1 + n0))
        se_d = math.sqrt(variance_d)
        correction = 1 - 3 / (4 * (n1 + n0) - 9)
        hedges_g = correction * cohens_d
        variance_g = (n1 + n0) / (n1 * n0) + hedges_g**2 / (2 * (n1 + n0))
        se_g = math.sqrt(variance_g)
        d_ci_low, d_ci_high = cohens_d - 1.96 * se_d, cohens_d + 1.96 * se_d
        g_ci_low, g_ci_high = hedges_g - 1.96 * se_g, hedges_g + 1.96 * se_g
        magnitude = _effect_magnitude(abs(hedges_g))
    else:
        cohens_d = d_ci_low = d_ci_high = math.nan
        hedges_g = variance_g = g_ci_low = g_ci_high = math.nan
        magnitude = "not estimable"

    return {
        "n_musician": n1,
        "mean_musician": mean1,
        "sd_musician": float(musician.std(ddof=1)),
        "n_nonmusician": n0,
        "mean_nonmusician": mean0,
        "sd_nonmusician": float(nonmusician.std(ddof=1)),
        "mean_difference_musician_minus_nonmusician": difference,
        "difference_ci_low": difference_ci_low,
        "difference_ci_high": difference_ci_high,
        "welch_t": statistic,
        "welch_df": degrees_freedom,
        "p_value": p_value,
        "cohens_d": cohens_d,
        "cohens_d_ci_low_approx": d_ci_low,
        "cohens_d_ci_high_approx": d_ci_high,
        "hedges_g": hedges_g,
        "hedges_g_ci_low_approx": g_ci_low,
        "hedges_g_ci_high_approx": g_ci_high,
        "effect_magnitude_heuristic": magnitude,
    }


def _effect_magnitude(abs_g: float) -> str:
    """Conventional rough labels for standardized mean differences."""
    if abs_g < 0.2:
        return "negligible"
    if abs_g < 0.5:
        return "small"
    if abs_g < 0.8:
        return "medium"
    return "large"


def _holm_adjust(p_values: list[float]) -> list[float]:
    """Holm family-wise adjustment, preserving the input order."""
    adjusted = [math.nan] * len(p_values)
    valid = [i for i, value in enumerate(p_values) if math.isfinite(value)]
    ordered = sorted(valid, key=lambda i: p_values[i])
    running_max = 0.0
    total = len(ordered)
    for rank, index in enumerate(ordered):
        candidate = min(1.0, (total - rank) * p_values[index])
        running_max = max(running_max, candidate)
        adjusted[index] = running_max
    return adjusted


def _analyse(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    descriptive_rows: list[dict[str, object]] = []
    test_rows: list[dict[str, object]] = []
    for outcome, label in PRIMARY_OUTCOMES.items():
        valid = data.loc[data[outcome].notna(), ["analysis_group", outcome]]
        musician = valid.loc[valid["analysis_group"].eq("musician"), outcome]
        nonmusician = valid.loc[valid["analysis_group"].eq("nonmusician"), outcome]
        for group, scores in (("musician", musician), ("nonmusician", nonmusician)):
            descriptive_rows.append({
                "outcome": outcome,
                "label": label,
                "group": group,
                "n": len(scores),
                "mean": float(scores.mean()) if len(scores) else math.nan,
                "sd": float(scores.std(ddof=1)) if len(scores) > 1 else math.nan,
            })
        test_rows.append({
            "outcome": outcome,
            "label": label,
            **_welch_comparison(musician, nonmusician),
        })

    tests = pd.DataFrame(test_rows)
    tests["p_holm"] = _holm_adjust(tests["p_value"].astype(float).tolist())
    return pd.DataFrame(descriptive_rows), tests


def _make_report(data: pd.DataFrame, descriptives: pd.DataFrame,
                 tests: pd.DataFrame, input_path: Path) -> str:
    counts = data["analysis_group"].value_counts()
    lines = [
        "# Overall musician vs nonmusician analysis",
        "",
        f"Input: `{input_path}`",
        f"Combined participants: {len(data):,}; "
        f"musicians: {int(counts.get('musician', 0)):,}; "
        f"nonmusicians: {int(counts.get('nonmusician', 0)):,}.",
        "",
        "All eligible participant rows were pooled before analysis. Laboratory "
        "membership was not used in grouping, weighting, adjustment, or testing. "
        "The primary test for each outcome is a two-sided Welch independent-samples "
        "t test of H₀: the overall group means are equal. Holm adjustment controls "
        "family-wise error across the three primary outcomes. Positive mean "
        "differences and positive Hedges' g values favour musicians.",
        "",
        "| Outcome | Musicians n, mean (SD) | Nonmusicians n, mean (SD) | Mean difference (95% CI) | t(df) | p | Holm p | Cohen's d | Hedges' g (approx. 95% CI) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    desc = descriptives.set_index(["outcome", "group"])
    for row in tests.to_dict("records"):
        musician = desc.loc[(row["outcome"], "musician")]
        nonmusician = desc.loc[(row["outcome"], "nonmusician")]
        lines.append(
            f"| {row['label']} | {int(musician['n'])}, {musician['mean']:.2f} ({musician['sd']:.2f}) | "
            f"{int(nonmusician['n'])}, {nonmusician['mean']:.2f} ({nonmusician['sd']:.2f}) | "
            f"{row['mean_difference_musician_minus_nonmusician']:.2f} "
            f"[{row['difference_ci_low']:.2f}, {row['difference_ci_high']:.2f}] | "
            f"{row['welch_t']:.2f} ({row['welch_df']:.1f}) | {row['p_value']:.3g} | "
            f"{row['p_holm']:.3g} | {row['cohens_d']:.3f} | "
            f"{row['hedges_g']:.3f} [{row['hedges_g_ci_low_approx']:.3f}, "
            f"{row['hedges_g_ci_high_approx']:.3f}] ({row['effect_magnitude_heuristic']}) |"
        )
    lines.extend([
        "",
        "The confidence interval for the raw mean difference uses the Welch-Satterthwaite degrees of freedom. "
        "Cohen's d is the pooled-SD standardized mean difference; Hedges' g applies the small-sample correction. "
        "The displayed standardized-effect confidence intervals are approximate Wald intervals. Magnitude labels "
        "use conventional thresholds as a rough guide, not as a significance test.",
        "",
        "## Output files",
        "",
        "- `combined_data.csv`: pooled participant rows used by the analysis, with a normalized `analysis_group` column.",
        "- `group_descriptives.csv`: overall sample size, mean, and SD by outcome and group.",
        "- `overall_tests.csv`: pooled Welch tests, confidence intervals, Holm-adjusted p values, and effect sizes.",
        "- `effect_sizes.csv`: Cohen's d and Hedges' g with approximate confidence intervals.",
    ])
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    input_path = args.input.expanduser().resolve()
    output_dir = args.output.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    combined = _load_and_combine(input_path)
    descriptives, tests = _analyse(combined)
    combined.to_csv(output_dir / "combined_data.csv", index=False)
    descriptives.to_csv(output_dir / "group_descriptives.csv", index=False)
    tests.to_csv(output_dir / "overall_tests.csv", index=False)
    effect_columns = [
        "outcome", "label", "n_musician", "n_nonmusician",
        "cohens_d", "cohens_d_ci_low_approx", "cohens_d_ci_high_approx",
        "hedges_g", "hedges_g_ci_low_approx", "hedges_g_ci_high_approx",
        "effect_magnitude_heuristic",
    ]
    tests[effect_columns].to_csv(output_dir / "effect_sizes.csv", index=False)
    (output_dir / "analysis_report.md").write_text(
        _make_report(combined, descriptives, tests, input_path), encoding="utf-8"
    )
    print(f"Overall pooled analysis complete. Results saved to: {output_dir}")


if __name__ == "__main__":
    main()
