"""Compare musicians and nonmusicians across laboratories.

The script reads the participant-level clean CSV produced by run_analysis.py.
It reports lab-level descriptive statistics and exploratory Welch t tests, then
combines lab-level Hedges' g values with a REML random-effects meta-analysis.
The primary pooled hypothesis test is two-sided: H0: mean Hedges' g = 0.

Run from the workspace root:
    python Python_code/my_analysis.py
    python Python_code/my_analysis.py --input path/to/dat-clean.csv
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.stats import chi2, norm, t


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT = SCRIPT_DIR / "output" / "data" / "clean" / "dat-clean.csv"
DEFAULT_OUTPUT = SCRIPT_DIR / "output" / "my_analysis"

PRIMARY_OUTCOMES = {
    "melody_span": "Music short-term memory",
    "digit_span": "Verbal short-term memory",
    "spatial_span": "Visuospatial short-term memory",
}
GROUPS = ("musician", "nonmusician")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare musician and nonmusician memory scores across labs, "
            "with lab-level Welch tests and pooled REML meta-analysis."
        )
    )
    parser.add_argument(
        "--input", type=Path, default=DEFAULT_INPUT,
        help=f"Participant-level clean CSV (default: {DEFAULT_INPUT}).",
    )
    parser.add_argument(
        "--output", type=Path, default=DEFAULT_OUTPUT,
        help=f"Directory for CSV tables and the Markdown report (default: {DEFAULT_OUTPUT}).",
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


def _read_input(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(
            f"Input CSV not found: {path}\n"
            "Run `python Python_code/run_analysis.py --rebuild-data` first, "
            "or supply a clean CSV with --input."
        )
    data = pd.read_csv(path, low_memory=False)
    required = {"group", "unit_short", *PRIMARY_OUTCOMES}
    missing = sorted(required.difference(data.columns))
    if missing:
        raise ValueError("Input CSV is missing required columns: " + ", ".join(missing))

    data = data.copy()
    data["_group"] = data["group"].map(_normalise_group)
    data["_lab"] = data["unit_short"].astype("string").str.strip()
    data.loc[data["_lab"].isin(["", "<NA>"]), "_lab"] = pd.NA
    data = data.loc[data["_group"].isin(GROUPS) & data["_lab"].notna()].copy()
    if data.empty:
        raise ValueError("No rows with recognised musician/nonmusician groups and lab names.")
    for outcome in PRIMARY_OUTCOMES:
        data[outcome] = pd.to_numeric(data[outcome], errors="coerce")
    return data


def _hedges_g(musician: pd.Series, nonmusician: pd.Series) -> tuple[float, float]:
    """Return small-sample corrected standardized difference and its variance."""
    n1, n0 = len(musician), len(nonmusician)
    if min(n1, n0) < 2:
        return math.nan, math.nan
    pooled_df = n1 + n0 - 2
    pooled_var = (
        (n1 - 1) * musician.var(ddof=1)
        + (n0 - 1) * nonmusician.var(ddof=1)
    ) / pooled_df
    pooled_sd = math.sqrt(pooled_var) if pooled_var >= 0 else math.nan
    if not math.isfinite(pooled_sd) or pooled_sd <= 0:
        return math.nan, math.nan
    correction = 1 - 3 / (4 * (n1 + n0) - 9)
    effect = correction * (musician.mean() - nonmusician.mean()) / pooled_sd
    variance = (n1 + n0) / (n1 * n0) + effect**2 / (2 * (n1 + n0))
    return float(effect), float(variance)


def _welch_test(musician: pd.Series, nonmusician: pd.Series) -> dict[str, float]:
    """Welch independent-samples t test and 95% CI for the raw mean difference."""
    n1, n0 = len(musician), len(nonmusician)
    mean_difference = float(musician.mean() - nonmusician.mean())
    if min(n1, n0) < 2:
        return {
            "mean_difference": mean_difference, "welch_t": math.nan,
            "welch_df": math.nan, "p_welch": math.nan,
            "diff_ci_low": math.nan, "diff_ci_high": math.nan,
        }

    v1 = float(musician.var(ddof=1)) / n1
    v0 = float(nonmusician.var(ddof=1)) / n0
    se = math.sqrt(v1 + v0)
    denom = v1**2 / (n1 - 1) + v0**2 / (n0 - 1)
    df = (v1 + v0) ** 2 / denom if denom > 0 else math.nan
    if se == 0 or not math.isfinite(se) or not math.isfinite(df):
        statistic, p_value = math.nan, math.nan
        ci_low = ci_high = math.nan
    else:
        statistic = mean_difference / se
        p_value = float(2 * t.sf(abs(statistic), df))
        margin = float(t.ppf(0.975, df) * se)
        ci_low, ci_high = mean_difference - margin, mean_difference + margin
    return {
        "mean_difference": mean_difference, "welch_t": float(statistic),
        "welch_df": float(df), "p_welch": p_value,
        "diff_ci_low": float(ci_low), "diff_ci_high": float(ci_high),
    }


def _lab_comparisons(data: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for outcome, label in PRIMARY_OUTCOMES.items():
        for lab, lab_data in data.groupby("_lab", sort=True, dropna=True):
            group = lab_data["_group"]
            musician = lab_data.loc[group.eq("musician"), outcome].dropna()
            nonmusician = lab_data.loc[group.eq("nonmusician"), outcome].dropna()
            enough_data = min(len(musician), len(nonmusician)) >= 2
            test = _welch_test(musician, nonmusician)
            effect, variance = _hedges_g(musician, nonmusician)
            rows.append({
                "lab": str(lab), "outcome": outcome, "label": label,
                "n_musician": len(musician),
                "mean_musician": float(musician.mean()) if len(musician) else math.nan,
                "sd_musician": float(musician.std(ddof=1)) if len(musician) > 1 else math.nan,
                "n_nonmusician": len(nonmusician),
                "mean_nonmusician": float(nonmusician.mean()) if len(nonmusician) else math.nan,
                "sd_nonmusician": float(nonmusician.std(ddof=1)) if len(nonmusician) > 1 else math.nan,
                **test,
                "hedges_g": effect,
                "variance_g": variance,
                "status": "analysed" if enough_data else "insufficient_n",
            })
    return pd.DataFrame(rows)


def _reml_objective(tau2: float, effects: np.ndarray, variances: np.ndarray) -> float:
    total_variance = variances + tau2
    weights = 1 / total_variance
    pooled = float(np.sum(weights * effects) / np.sum(weights))
    residual_q = float(np.sum(weights * (effects - pooled) ** 2))
    return 0.5 * (
        float(np.log(total_variance).sum())
        + math.log(float(weights.sum()))
        + residual_q
    )


def _fit_reml(effects: np.ndarray, variances: np.ndarray) -> dict[str, float | int]:
    valid = np.isfinite(effects) & np.isfinite(variances) & (variances > 0)
    effects, variances = effects[valid], variances[valid]
    k = len(effects)
    if k < 2:
        raise ValueError(f"At least two labs with valid effect estimates are required; found {k}.")

    spread = float(np.var(effects, ddof=1))
    upper = max(1.0, (spread + float(np.max(variances))) * 100)
    optimum = minimize_scalar(
        _reml_objective, args=(effects, variances),
        bounds=(0.0, upper), method="bounded", options={"xatol": 1e-12},
    )
    tau2_candidates = [(0.0, _reml_objective(0.0, effects, variances))]
    if optimum.success:
        tau2_candidates.append((float(optimum.x), float(optimum.fun)))
    tau2 = min(tau2_candidates, key=lambda pair: pair[1])[0]

    weights = 1 / (variances + tau2)
    estimate = float(np.sum(weights * effects) / np.sum(weights))
    se = float(math.sqrt(1 / np.sum(weights)))
    z_value = estimate / se
    q_weights = 1 / variances
    fixed_mean = float(np.sum(q_weights * effects) / np.sum(q_weights))
    q_stat = float(np.sum(q_weights * (effects - fixed_mean) ** 2))
    q_df = k - 1
    return {
        "k_labs": k,
        "hedges_g_pooled": estimate,
        "se": se,
        "z": z_value,
        "p_two_sided": float(2 * norm.sf(abs(z_value))),
        "ci_low": estimate - 1.96 * se,
        "ci_high": estimate + 1.96 * se,
        "tau2": tau2,
        "tau": math.sqrt(tau2),
        "Q": q_stat,
        "Q_df": q_df,
        "Q_p": float(chi2.sf(q_stat, q_df)),
        "I2_percent": max(0.0, (q_stat - q_df) / q_stat) * 100 if q_stat > 0 else 0.0,
    }


def _holm_adjust(p_values: list[float]) -> list[float]:
    """Holm family-wise adjustment, preserving the input order."""
    adjusted = [math.nan] * len(p_values)
    valid_indices = [i for i, value in enumerate(p_values) if math.isfinite(value)]
    ordered = sorted(valid_indices, key=lambda i: p_values[i])
    running_max = 0.0
    total = len(ordered)
    for rank, index in enumerate(ordered):
        candidate = min(1.0, (total - rank) * p_values[index])
        running_max = max(running_max, candidate)
        adjusted[index] = running_max
    return adjusted


def _pooled_results(lab_results: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for outcome, label in PRIMARY_OUTCOMES.items():
        studies = lab_results.loc[
            lab_results["outcome"].eq(outcome)
            & np.isfinite(lab_results["hedges_g"])
            & np.isfinite(lab_results["variance_g"])
        ]
        fit = _fit_reml(
            studies["hedges_g"].to_numpy(dtype=float),
            studies["variance_g"].to_numpy(dtype=float),
        )
        rows.append({"outcome": outcome, "label": label, **fit})
    result = pd.DataFrame(rows)
    result["p_holm"] = _holm_adjust(result["p_two_sided"].astype(float).tolist())
    return result


def _markdown_report(data: pd.DataFrame, pooled: pd.DataFrame, input_path: Path) -> str:
    recognised = data["_group"].value_counts()
    lines = [
        "# Musician vs nonmusician: multi-lab analysis",
        "",
        f"Input: `{input_path}`",
        f"Participants with recognised group and lab: {len(data):,}; "
        f"laboratories: {data['_lab'].nunique()}; "
        f"musicians: {int(recognised.get('musician', 0)):,}; "
        f"nonmusicians: {int(recognised.get('nonmusician', 0)):,}.",
        "",
        "## Primary hypothesis tests",
        "",
        "For each outcome, laboratory-specific Hedges' g values were combined "
        "using a random-effects meta-analysis estimated by REML. The pooled "
        "two-sided z test evaluates H₀: mean Hedges' g = 0 against H₁: mean "
        "Hedges' g ≠ 0. Positive values favour musicians. Holm-adjusted p values "
        "control family-wise error across the three primary outcomes.",
        "",
        "| Outcome | Labs | Hedges' g | 95% CI | z | p | Holm p | I² |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in pooled.to_dict("records"):
        lines.append(
            f"| {row['label']} | {row['k_labs']} | {row['hedges_g_pooled']:.3f} | "
            f"[{row['ci_low']:.3f}, {row['ci_high']:.3f}] | {row['z']:.2f} | "
            f"{row['p_two_sided']:.3g} | {row['p_holm']:.3g} | {row['I2_percent']:.1f}% |"
        )
    lines.extend([
        "",
        "## Laboratory-level comparisons",
        "",
        "`lab_comparisons.csv` contains each lab's sample sizes, means, SDs, "
        "raw mean difference (musician minus nonmusician), Welch independent-"
        "samples t test, and Hedges' g. Lab-level tests are exploratory and their "
        "p values are not multiplicity-adjusted. The primary cross-lab inference "
        "is the pooled REML result above.",
        "",
        "## Files",
        "",
        "- `lab_comparisons.csv`: laboratory-by-outcome descriptive and Welch test results.",
        "- `pooled_meta_analysis.csv`: REML estimates, hypothesis tests, and heterogeneity.",
    ])
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    input_path = args.input.expanduser().resolve()
    output_dir = args.output.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    data = _read_input(input_path)
    lab_results = _lab_comparisons(data)
    pooled = _pooled_results(lab_results)

    lab_results.to_csv(output_dir / "lab_comparisons.csv", index=False)
    pooled.to_csv(output_dir / "pooled_meta_analysis.csv", index=False)
    (output_dir / "analysis_report.md").write_text(
        _markdown_report(data, pooled, input_path), encoding="utf-8"
    )
    print(f"Analysis complete. Results saved to: {output_dir}")


if __name__ == "__main__":
    main()
