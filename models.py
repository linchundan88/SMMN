"""Approximate Python counterparts of the R mixed-effects models.

Uses statsmodels MixedLM and AIC-based backward selection. The R project uses
lme4/lmerTest stepwise selection, so denominators and selected fixed effects
may differ slightly between the two implementations.
"""
from __future__ import annotations

import logging
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

LOG = logging.getLogger(__name__)

BASE_COVARIATES = [
    "FLUID_INTELLIGENCE_RAVEN",
    "CRYSTALLIZED_INTELLIGENCE",
    "EXECUTIVE_FUNCTIONS_N_BACK",
    "BIG5_OPENNESS",
    "EBMRQ",
    "SOCIOECONOMIC_STATUS",
    "YEARS_OF_EDUCATION",
    "AGE",
]
SUBGROUP_COVARIATES = BASE_COVARIATES + ["PROMS", "GOLD_MSI"]
MEMORY = {
    "MUSIC_MEMORY": "melody_span",
    "VERBAL_MEMORY": "digit_span",
    "VISUOSPATIAL_MEMORY": "spatial_span",
}
SOURCE_TO_MODEL = {
    "raven": "FLUID_INTELLIGENCE_RAVEN",
    "waisvocabulary": "CRYSTALLIZED_INTELLIGENCE",
    "nback": "EXECUTIVE_FUNCTIONS_N_BACK",
    "bfi_open_mindedness": "BIG5_OPENNESS",
    "eBMRQ_TOT": "EBMRQ",
    "ses_nuclear_family": "SOCIOECONOMIC_STATUS",
    "years_of_education": "YEARS_OF_EDUCATION",
    "age": "AGE",
    "PROMSTOTALSCORE": "PROMS",
    "PROMSTOTALSCORENOMELODY": "PROMSnomelody",
    "gold_msi_general_sophistication": "GOLD_MSI",
}


def _prepare(data: pd.DataFrame) -> pd.DataFrame:
    frame = data.copy()
    frame["UNIT"] = frame["unit_short"].astype("string")
    frame["GROUP"] = frame["group"].astype("string").str.casefold()
    frame["GROUP_MUSIC"] = frame["GROUP"].eq("musician").astype(float)
    for source, target in SOURCE_TO_MODEL.items():
        if source in frame and target not in frame:
            frame[target] = pd.to_numeric(frame[source], errors="coerce")
    for target in set(BASE_COVARIATES + SUBGROUP_COVARIATES + ["PROMSnomelody"]):
        if target in frame:
            values = pd.to_numeric(frame[target], errors="coerce")
            frame[f"{target}0"] = values - values.mean()
    for model_name, source in MEMORY.items():
        frame[model_name] = pd.to_numeric(frame[source], errors="coerce")
    return frame


def _formula(outcome: str, covariates: list[str], overall: bool) -> str:
    terms = []
    if overall:
        terms.append("GROUP_MUSIC")
    for covariate in covariates:
        centered = f"{covariate}0"
        terms.append(centered)
        if overall:
            terms.append(f"GROUP_MUSIC:{centered}")
    return f"{outcome} ~ " + (" + ".join(terms) if terms else "1")


def _fit(
    data: pd.DataFrame,
    formula: str,
    random_slope: bool,
    reml: bool,
):
    re_formula = "1 + GROUP_MUSIC" if random_slope else "1"
    last_error = None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            model = smf.mixedlm(
                formula, data=data, groups=data["UNIT"], re_formula=re_formula,
                missing="raise",
            )
            for method in ("lbfgs", "powell"):
                try:
                    result = model.fit(reml=reml, method=method, disp=False)
                    if bool(getattr(result, "converged", False)):
                        return result, random_slope
                    last_error = RuntimeError(f"{method} optimizer did not converge")
                except Exception as exc:  # A failed optimizer should not stop other outcomes.
                    last_error = exc
        except Exception as exc:
            last_error = exc

        # Fall back to a random intercept if the random group slope is singular.
        if random_slope:
            try:
                model = smf.mixedlm(
                    formula, data=data, groups=data["UNIT"], re_formula="1",
                    missing="raise",
                )
                for method in ("lbfgs", "powell"):
                    try:
                        result = model.fit(reml=reml, method=method, disp=False)
                        if bool(getattr(result, "converged", False)):
                            return result, False
                        last_error = RuntimeError(f"{method} intercept model did not converge")
                    except Exception as exc:
                        last_error = exc
            except Exception as exc:
                last_error = exc
    raise RuntimeError(str(last_error))


def _aic_selection(
    data: pd.DataFrame,
    outcome: str,
    covariates: list[str],
    overall: bool,
    random_slope: bool,
):
    active = list(covariates)
    formula = _formula(outcome, active, overall)
    current, used_slope = _fit(data, formula, random_slope, reml=False)
    while active:
        candidates = []
        for covariate in active:
            trial = [c for c in active if c != covariate]
            trial_formula = _formula(outcome, trial, overall)
            try:
                result, structure = _fit(data, trial_formula, used_slope, reml=False)
                if np.isfinite(result.aic):
                    candidates.append((float(result.aic), trial, trial_formula, result, structure))
            except Exception as exc:
                LOG.debug("Skipping model candidate for %s after %s", outcome, exc)
        if not candidates:
            break
        best = min(candidates, key=lambda item: item[0])
        if best[0] + 1e-8 >= float(current.aic):
            break
        _, active, formula, current, used_slope = best
    selected_aic = float(current.aic)
    final, used_slope = _fit(data, formula, used_slope, reml=True)
    return final, formula, active, used_slope, selected_aic


def _coefficients(result, name: str) -> list[dict]:
    rows = []
    estimates = result.fe_params
    errors = result.bse_fe
    pvalues = result.pvalues
    for term in estimates.index:
        estimate = float(estimates[term])
        se = float(errors[term])
        p = float(pvalues.get(term, np.nan))
        rows.append({
            "model": name, "term": term, "estimate": estimate, "se": se,
            "statistic_z": estimate / se if se else np.nan, "p": p,
            "ci_low": estimate - 1.96 * se, "ci_high": estimate + 1.96 * se,
        })
    return rows


def _fit_one(
    data: pd.DataFrame, model_name: str, outcome: str,
    covariates: list[str], overall: bool,
) -> tuple[dict, list[dict], object | None]:
    needed = [outcome, "UNIT", *covariates]
    if overall:
        needed.append("GROUP_MUSIC")
    absent = [name for name in needed if name not in data]
    if absent:
        return ({"model": model_name, "outcome": outcome, "status": "skipped",
                 "message": "Missing fields: " + ", ".join(absent)}, [], None)
    model_data = data.dropna(subset=needed).copy()
    if len(model_data) != len(data):
        return ({
            "model": model_name, "outcome": outcome, "status": "skipped",
            "n": len(model_data),
            "message": f"{len(data) - len(model_data)} row(s) contain missing model values; R used na.fail",
        }, [], None)
    if model_data["UNIT"].nunique() < 2:
        return ({"model": model_name, "outcome": outcome, "status": "skipped",
                 "message": "Fewer than two labs"}, [], None)
    try:
        result, formula, selected, random_slope, selected_aic = _aic_selection(
            model_data, outcome, covariates, overall, random_slope=overall
        )
        row = {
            "model": model_name, "outcome": outcome, "status": "fit",
            "n": len(model_data), "labs": model_data["UNIT"].nunique(),
            "formula": formula, "selected_covariates": ", ".join(selected),
            "random_effect": "intercept + group slope" if random_slope else "intercept",
            "aic_ml": selected_aic, "log_likelihood_reml": float(result.llf),
            "converged": bool(result.converged), "message": "",
        }
        return row, _coefficients(result, model_name), result
    except Exception as exc:
        return ({"model": model_name, "outcome": outcome, "status": "failed",
                 "message": str(exc)}, [], None)


def run_mixed_models(data: pd.DataFrame, table_dir: Path, plot_dir: Path) -> None:
    """Fit full-sample and group-specific mixed models and save coefficient tables."""
    frame = _prepare(data)
    fit_rows: list[dict] = []
    coefficient_rows: list[dict] = []
    primary_results = {}

    for model_outcome, source in MEMORY.items():
        row, coefficients, result = _fit_one(
            frame, f"all_{source}", model_outcome, BASE_COVARIATES, overall=True
        )
        fit_rows.append(row)
        coefficient_rows.extend(coefficients)
        if result is not None:
            primary_results[source] = (row, coefficients)

    for group in ("musician", "nonmusician"):
        subset = frame.loc[frame["GROUP"].eq(group)].copy()
        specifications = [
            ("visuo", "VISUOSPATIAL_MEMORY", SUBGROUP_COVARIATES),
            ("verbal", "VERBAL_MEMORY", SUBGROUP_COVARIATES),
            ("music1", "MUSIC_MEMORY", SUBGROUP_COVARIATES),
            ("music2", "MUSIC_MEMORY", BASE_COVARIATES + ["PROMSnomelody", "GOLD_MSI"]),
        ]
        for suffix, outcome, covariates in specifications:
            label = f"{group}_{suffix}"
            row, coefficients, _ = _fit_one(
                subset, label, outcome, covariates, overall=False
            )
            fit_rows.append(row)
            coefficient_rows.extend(coefficients)

    pd.DataFrame(fit_rows).to_csv(table_dir / "mixed_model_fit_summary.csv", index=False)
    pd.DataFrame(coefficient_rows).to_csv(table_dir / "mixed_model_coefficients.csv", index=False)

    if primary_results:
        fig, axes = plt.subplots(len(primary_results), 1,
                                 figsize=(9, max(5, 3.2 * len(primary_results))))
        axes = np.atleast_1d(axes)
        for ax, (outcome, (_, coefficients)) in zip(axes, primary_results.items()):
            terms = [row for row in coefficients if row["term"] != "Intercept"]
            y = np.arange(len(terms))
            ax.errorbar(
                [row["estimate"] for row in terms], y,
                xerr=[
                    [row["estimate"] - row["ci_low"] for row in terms],
                    [row["ci_high"] - row["estimate"] for row in terms],
                ],
                fmt="o", capsize=2, color="#375d88",
            )
            ax.axvline(0, color="black", linestyle="--", linewidth=.8)
            ax.set_yticks(y, [row["term"] for row in terms], fontsize=7)
            ax.set_title(f"All participants: {outcome}")
            ax.set_xlabel("Estimate and 95% Wald CI")
            ax.grid(axis="x", alpha=.2)
        fig.tight_layout()
        fig.savefig(plot_dir / "mixed_model_coefficients.png", dpi=200, bbox_inches="tight")
        fig.savefig(plot_dir / "mixed_model_coefficients.pdf", bbox_inches="tight")
        plt.close(fig)


