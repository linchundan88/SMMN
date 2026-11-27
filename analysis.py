"""Statistical analyses corresponding to reports/02-analysis.qmd."""
from __future__ import annotations

import logging
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.io import loadmat
from scipy.optimize import minimize_scalar
from scipy.stats import chi2, norm

LOG = logging.getLogger(__name__)
PRIMARY = {
    "melody_span": "Music short-term memory",
    "digit_span": "Verbal short-term memory",
    "spatial_span": "Visuospatial short-term memory",
}
EXPLORATORY = {
    "PROMSMELODYSCORE": ("PROMS", "Melody"),
    "PROMSTUNINGSCORE": ("PROMS", "Tuning"),
    "PROMSBEATSCORE": ("PROMS", "Beat"),
    "PROMSSPEEDSCORE": ("PROMS", "Speed"),
    "PROMSTOTALSCORE": ("PROMS", "PROMS total"),
    "PROMSTOTALSCORENOMELODY": ("PROMS", "PROMS without melody"),
    "eBMRQ_MusicSeeking": ("eBMRQ", "Music seeking"),
    "eBMRQ_EmotionEvocation": ("eBMRQ", "Emotion evocation"),
    "eBMRQ_MoodRegulation": ("eBMRQ", "Mood regulation"),
    "eBMRQ_SensoryMotor": ("eBMRQ", "Sensory motor"),
    "eBMRQ_SocialReward": ("eBMRQ", "Social reward"),
    "eBMRQ_MusicalAbsorption": ("eBMRQ", "Musical absorption"),
    "eBMRQ_TOT": ("eBMRQ", "eBMRQ total"),
    "gold_msi_active_engagement": ("Gold-MSI", "Active engagement"),
    "gold_msi_perceptual_abilities": ("Gold-MSI", "Perceptual abilities"),
    "gold_msi_musical_training": ("Gold-MSI", "Musical training"),
    "gold_msi_emotions": ("Gold-MSI", "Emotions"),
    "gold_msi_singing_abilities": ("Gold-MSI", "Singing abilities"),
    "gold_msi_general_sophistication": ("Gold-MSI", "General sophistication"),
    "waisvocabulary": ("Cognition and SES", "WAIS vocabulary"),
    "raven": ("Cognition and SES", "Raven"),
    "nback": ("Cognition and SES", "N-back"),
    "ses_nuclear_family": ("Cognition and SES", "Family SES"),
    "bfi_extraversion": ("Big Five", "Extraversion"),
    "bfi_agreeableness": ("Big Five", "Agreeableness"),
    "bfi_conscientiousness": ("Big Five", "Conscientiousness"),
    "bfi_negative_emotionality": ("Big Five", "Negative emotionality"),
    "bfi_open_mindedness": ("Big Five", "Open-mindedness"),
}


def hedges_g(m1, sd1, n1, m0, sd0, n0):
    """Hedges g and metafor's sampling-variance estimator."""
    n1, n0 = int(n1), int(n0)
    if min(n1, n0) < 2:
        return np.nan, np.nan
    df = n1 + n0 - 2
    pooled = math.sqrt(((n1 - 1) * sd1**2 + (n0 - 1) * sd0**2) / df)
    if not np.isfinite(pooled) or pooled <= 0:
        return np.nan, np.nan
    correction = 1 - 3 / (4 * (n1 + n0) - 9)
    g = correction * (m1 - m0) / pooled
    vi = (n1 + n0) / (n1 * n0) + g**2 / (2 * (n1 + n0))
    return float(g), float(vi)


def _reml_nll(tau2, yi, vi):
    v = vi + tau2
    w = 1 / v
    total = w.sum()
    mu = (w * yi).sum() / total
    q = (w * (yi - mu) ** 2).sum()
    return 0.5 * (np.log(v).sum() + np.log(total) + q)


def fit_reml(yi, vi):
    """Random-effects intercept-only meta-analysis using REML."""
    yi, vi = np.asarray(yi, float), np.asarray(vi, float)
    ok = np.isfinite(yi) & np.isfinite(vi) & (vi > 0)
    yi, vi = yi[ok], vi[ok]
    k = len(yi)
    if k < 2:
        raise ValueError("Need at least two labs with valid group estimates")
    upper = max(1., (np.var(yi) + np.max(vi)) * 100)
    opt = minimize_scalar(_reml_nll, args=(yi, vi), bounds=(0, upper),
                          method="bounded", options={"xatol": 1e-12})
    tau2 = min([(0., _reml_nll(0., yi, vi)), (float(opt.x), float(opt.fun))],
               key=lambda item: item[1])[0]
    w = 1 / (vi + tau2)
    estimate = float(np.sum(w * yi) / np.sum(w))
    se = float(np.sqrt(1 / np.sum(w)))
    z = estimate / se
    fixed_w = 1 / vi
    fixed_mean = np.sum(fixed_w * yi) / fixed_w.sum()
    q = float(np.sum(fixed_w * (yi - fixed_mean)**2))
    df = k - 1
    return {
        "k": k, "estimate": estimate, "se": se, "z": z,
        "p": float(2 * norm.sf(abs(z))),
        "ci_low": estimate - 1.96 * se, "ci_high": estimate + 1.96 * se,
        "tau2": tau2, "tau": math.sqrt(tau2), "Q": q,
        "Q_df": df, "Q_p": float(chi2.sf(q, df)),
        "I2": max(0., (q - df) / q) * 100 if q else 0.,
    }


def lab_effects(data, outcome):
    required = {"unit_short", "group", outcome}
    if not required.issubset(data):
        raise KeyError(f"Missing fields: {sorted(required - set(data.columns))}")
    rows = []
    for lab, frame in data.groupby("unit_short", dropna=True):
        frame = frame.copy()
        frame[outcome] = pd.to_numeric(frame[outcome], errors="coerce")
        groups = frame["group"].astype("string").str.casefold()
        musician = frame.loc[groups.eq("musician"), outcome].dropna()
        control = frame.loc[groups.eq("nonmusician"), outcome].dropna()
        if min(len(musician), len(control)) < 2:
            continue
        g, vi = hedges_g(musician.mean(), musician.std(ddof=1), len(musician),
                         control.mean(), control.std(ddof=1), len(control))
        rows.append({
            "unit": lab, "outcome": outcome, "n_musician": len(musician),
            "n_nonmusician": len(control), "g": g, "vi": vi,
            "se": np.sqrt(vi), "ci_low": g - 1.96 * np.sqrt(vi),
            "ci_high": g + 1.96 * np.sqrt(vi),
        })
    return pd.DataFrame(rows)


def pooled_outcome(data, outcome, label):
    studies = lab_effects(data, outcome)
    studies = studies.loc[np.isfinite(studies["g"]) & np.isfinite(studies["vi"])]
    fit = fit_reml(studies["g"], studies["vi"])
    return {"outcome": outcome, "label": label, **fit}, studies


def _forest(studies, pooled, title, path):
    studies = studies.sort_values("unit").reset_index(drop=True)
    y = np.arange(len(studies), 0, -1)
    fig, ax = plt.subplots(figsize=(9, max(4, .29 * (len(studies) + 2))))
    ax.errorbar(studies.g, y, xerr=[studies.g - studies.ci_low,
                                    studies.ci_high - studies.g],
                fmt="o", capsize=2, color="#28618a", markersize=4)
    ax.errorbar(pooled["estimate"], 0,
                xerr=[[pooled["estimate"] - pooled["ci_low"]],
                      [pooled["ci_high"] - pooled["estimate"]]],
                fmt="D", capsize=3, color="#aa4039")
    ax.axvline(0, color="black", linestyle="--", linewidth=.8)
    ax.set_yticks([*y, 0], [*studies.unit.astype(str), "REML pooled"])
    ax.set_xlabel("Hedges' g (musicians minus nonmusicians)")
    ax.set_title(title)
    ax.grid(axis="x", alpha=.2)
    fig.tight_layout()
    fig.savefig(path.with_suffix(".png"), dpi=200, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def _dataset_counts(data, label):
    rows = {
        "dataset": label,
        "rows": len(data),
        "unique_subjects": data["subject_ID"].nunique() if "subject_ID" in data else np.nan,
        "labs": data["unit_short"].nunique() if "unit_short" in data else np.nan,
    }
    if "group" in data:
        counts = data["group"].astype("string").str.casefold().value_counts()
        rows["musicians"] = int(counts.get("musician", 0))
        rows["nonmusicians"] = int(counts.get("nonmusician", 0))
    if "subject_ID of the matched participant" in data:
        matched = data["subject_ID of the matched participant"]
        valid = matched.notna() & matched.astype("string").str.casefold().ne("na")
        rows["matched_rows"] = int(valid.sum())
        rows["unmatched_rows"] = int((~valid).sum())
    return rows


def _descriptives(data, out_dir):
    rows = []
    for outcome, label in PRIMARY.items():
        for group, frame in data.groupby("group"):
            values = pd.to_numeric(frame[outcome], errors="coerce").dropna()
            rows.append({"outcome": outcome, "label": label, "group": group,
                         "n": len(values), "mean": values.mean(),
                         "sd": values.std(ddof=1)})
    pd.DataFrame(rows).to_csv(out_dir / "outcome_descriptives.csv", index=False)


def _correlations(data, plot_dir, table_dir):
    name_map = {
        "digit_span": "Verbal STM", "spatial_span": "Visuospatial STM",
        "melody_span": "Music STM", "PROMSTOTALSCORE": "PROMS",
        "PROMSTOTALSCORENOMELODY": "PROMS without melody",
        "gold_msi_general_sophistication": "Gold-MSI", "raven": "Raven",
        "waisvocabulary": "WAIS vocabulary", "nback": "N-back",
        "bfi_open_mindedness": "BFI-2 Open-mindedness", "eBMRQ_TOT": "eBMRQ",
        "ses_nuclear_family": "Family SES", "years_of_education": "Years of education",
        "age": "Age",
    }
    available = [col for col in name_map if col in data]
    for group in ("musician", "nonmusician"):
        subset = data.loc[data.group.eq(group), available].apply(pd.to_numeric, errors="coerce")
        corr = subset.corr(method="pearson").rename(index=name_map, columns=name_map)
        corr.to_csv(table_dir / f"correlations_{group}.csv")
        labels = list(corr.columns)
        fig, ax = plt.subplots(figsize=(max(8, .65 * len(labels)), max(7, .55 * len(labels))))
        image = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
        ax.set_xticks(range(len(labels)), labels, rotation=75, ha="right", fontsize=8)
        ax.set_yticks(range(len(labels)), labels, fontsize=8)
        for i in range(len(labels)):
            for j in range(len(labels)):
                value = corr.iat[i, j]
                if np.isfinite(value):
                    ax.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=6)
        fig.colorbar(image, ax=ax, label="Pearson r", shrink=.75)
        ax.set_title(f"Correlations: {group}")
        fig.tight_layout()
        fig.savefig(plot_dir / f"corrplot_{group}.png", dpi=200, bbox_inches="tight")
        fig.savefig(plot_dir / f"corrplot_{group}.pdf", bbox_inches="tight")
        plt.close(fig)


def _outcome_plot(data, plot_dir):
    fig, axes = plt.subplots(3, 1, figsize=(8, 10))
    for ax, (outcome, label) in zip(axes, PRIMARY.items()):
        values = pd.to_numeric(data[outcome], errors="coerce")
        z = (values - values.mean()) / values.std(ddof=1)
        keep = z.abs().lt(3)
        frame = data.loc[keep].copy()
        frame["_z"] = z.loc[keep]
        box = [frame.loc[frame.group.eq(group), "_z"].dropna()
               for group in ("nonmusician", "musician")]
        ax.boxplot(box, tick_labels=["Nonmusicians", "Musicians"], showfliers=False)
        ax.axhline(0, color="gray", linestyle=":", linewidth=.7)
        ax.set_ylabel("Standard score")
        ax.set_title(label)
    fig.tight_layout()
    fig.savefig(plot_dir / "outcomes_plot.png", dpi=200, bbox_inches="tight")
    fig.savefig(plot_dir / "outcomes_plot.pdf", bbox_inches="tight")
    plt.close(fig)


def _exploratory(data, table_dir, plot_dir):
    rows = []
    for outcome, (category, label) in EXPLORATORY.items():
        if outcome not in data:
            continue
        try:
            pooled, _ = pooled_outcome(data, outcome, label)
            rows.append({"category": category, **pooled})
        except (ValueError, KeyError) as exc:
            LOG.warning("Skipping exploratory measure %s: %s", outcome, exc)
    result = pd.DataFrame(rows)
    result.to_csv(table_dir / "meta_exploratory.csv", index=False)
    if result.empty:
        return
    result = result.sort_values(["category", "estimate"])
    y = np.arange(len(result))
    fig, ax = plt.subplots(figsize=(10, max(6, .32 * len(result))))
    ax.errorbar(result.estimate, y,
                xerr=[result.estimate - result.ci_low, result.ci_high - result.estimate],
                fmt="o", capsize=2, color="#39714f", markersize=4)
    ax.axvline(0, color="black", linestyle="--", linewidth=.8)
    ax.set_yticks(y, result.label, fontsize=8)
    ax.set_xlabel("Hedges' g (musicians minus nonmusicians)")
    ax.set_title("Exploratory random-effects meta-analyses")
    fig.tight_layout()
    fig.savefig(plot_dir / "exploratory_meta_forest.png", dpi=200, bbox_inches="tight")
    fig.savefig(plot_dir / "exploratory_meta_forest.pdf", bbox_inches="tight")
    plt.close(fig)


def _power(d, k, n, tau):
    vi = (2 * n) / (n * n) + d**2 / (2 * (2 * n))
    z = d / math.sqrt((vi + tau**2) / k)
    crit = norm.ppf(.975)
    return 1 - norm.cdf(crit - z) + norm.cdf(-crit - z)


def _prior_power(table_dir, plot_dir):
    rows = []
    for tau in (0., .11):
        for k in range(8, 21):
            candidates = np.arange(10, 501)
            powers = np.array([_power(.3, k, int(n), tau) for n in candidates])
            idx = np.argmin(np.abs(powers - .905))
            rows.append({"d": .3, "tau": tau, "k": k,
                         "n_per_group": candidates[idx], "power": powers[idx]})
    frame = pd.DataFrame(rows)
    frame.to_csv(table_dir / "power_a_priori.csv", index=False)
    fig, ax = plt.subplots(figsize=(8, 6))
    for tau, part in frame.groupby("tau"):
        ax.plot(part.k, part.n_per_group, marker="o", label=f"tau={tau:g}")
    ax.axvline(13, color="gray", linestyle="--")
    ax.axhline(21, color="gray", linestyle="--")
    ax.set(xlabel="Number of research groups (k)", ylabel="Participants per group",
           title="A-priori power for d=0.3, target power=0.90")
    ax.legend()
    ax.grid(alpha=.2)
    fig.tight_layout()
    fig.savefig(plot_dir / "plot_power.png", dpi=200, bbox_inches="tight")
    fig.savefig(plot_dir / "plot_power.pdf", bbox_inches="tight")
    plt.close(fig)


def _batch_reml_pvalues(yi, vi):
    """Solve REML tau-squared scores across simulations in vectorised batches."""
    count = yi.shape[0]

    def score(tau):
        weights = 1.0 / (vi + tau[:, None])
        sum_w = weights.sum(axis=1)
        mean = (weights * yi).sum(axis=1) / sum_w
        p_y = weights * (yi - mean[:, None])
        trace_p = sum_w - (weights**2).sum(axis=1) / sum_w
        return 0.5 * ((p_y**2).sum(axis=1) - trace_p)

    low = np.zeros(count)
    high = np.ones(count)
    active = score(low) > 0
    for _step in range(40):
        expand = active & (score(high) > 0)
        if not expand.any():
            break
        high[expand] *= 2
    for _step in range(48):
        middle = (low + high) / 2
        positive = active & (score(middle) > 0)
        low[positive] = middle[positive]
        high[active & ~positive] = middle[active & ~positive]
    tau2 = np.where(active, (low + high) / 2, 0.0)
    weights = 1.0 / (vi + tau2[:, None])
    mean = (weights * yi).sum(axis=1) / weights.sum(axis=1)
    se = np.sqrt(1.0 / weights.sum(axis=1))
    return 2 * norm.sf(np.abs(mean / se))


def _posthoc_power(data, table_dir, n_simulations, seed):
    """Monte Carlo power using the observed site-specific group sizes."""
    sizes = data.groupby(["unit_short", "group"]).size().unstack(fill_value=0)
    sizes.columns = sizes.columns.astype(str).str.casefold()
    if not {"musician", "nonmusician"}.issubset(sizes):
        return
    sizes = sizes.loc[(sizes.musician >= 2) & (sizes.nonmusician >= 2)]
    n1 = sizes["musician"].to_numpy(dtype=int)
    n0 = sizes["nonmusician"].to_numpy(dtype=int)
    k = len(sizes)
    rng = np.random.default_rng(seed)
    rows = []
    for tau2 in (.11**2, .1, .2):
        for effect in (0., .05, .1, .2, .3, .5):
            shifts = rng.normal(0, np.sqrt(tau2), size=(n_simulations, k))
            yi = np.empty((n_simulations, k))
            vi = np.empty((n_simulations, k))
            for i, (nm, nc) in enumerate(zip(n1, n0)):
                sample_m = rng.normal(effect + shifts[:, i, None], 1.,
                                      size=(n_simulations, int(nm)))
                sample_c = rng.normal(0., 1., size=(n_simulations, int(nc)))
                mean_m, mean_c = sample_m.mean(axis=1), sample_c.mean(axis=1)
                sd_m = sample_m.std(axis=1, ddof=1)
                sd_c = sample_c.std(axis=1, ddof=1)
                pooled_sd = np.sqrt(
                    ((nm - 1) * sd_m**2 + (nc - 1) * sd_c**2) / (nm + nc - 2)
                )
                correction = 1 - 3 / (4 * (nm + nc) - 9)
                yi[:, i] = correction * (mean_m - mean_c) / pooled_sd
                vi[:, i] = (nm + nc) / (nm * nc) + yi[:, i]**2 / (2 * (nm + nc))
            pvalues = _batch_reml_pvalues(yi, vi)
            rows.append({
                "d": effect, "tau2": tau2, "k": k,
                "power": float(np.mean(pvalues <= .05)),
                "simulations": int(n_simulations), "method": "REML Monte Carlo",
            })
    pd.DataFrame(rows).to_csv(table_dir / "power_posthoc.csv", index=False)

def _literature_figure(project_root, plot_dir):
    path = project_root / "scripts" / "create-figure-1" / "data-for-figure-1.mat"
    if not path.exists():
        return
    source = loadmat(path)
    if not {"power_data", "g_literature", "N_literature"}.issubset(source):
        LOG.warning("MATLAB figure input lacks expected variables")
        return
    power = np.asarray(source["power_data"], float)
    effects = np.arange(.2, 1.0001, .05)
    if power.shape[0] != len(effects) or power.shape[1] < 3:
        LOG.warning("Unexpected MATLAB power data shape; skipping figure 1")
        return
    fig, ax = plt.subplots(figsize=(7, 7))
    for i, (marker, level) in enumerate(zip(("^", "s", "*"), (.80, .90, .95))):
        ax.plot(effects, 2 * power[:, i], marker=marker, linestyle=":",
                color="black", markerfacecolor="white",
                label=f"Power = {level:.2f}")
    ax.scatter(np.ravel(source["g_literature"]), np.ravel(source["N_literature"]),
               color="black", label="Prior literature")
    ax.axvline(.3, color="black", linestyle=":")
    ax.set_yscale("log")
    ax.set(xlabel="Target effect size (Hedges' g)", ylabel="Total participants",
           title="Target sample size and published studies")
    ax.legend()
    fig.tight_layout()
    for ext in ("png", "pdf", "svg"):
        fig.savefig(plot_dir / f"figure1.{ext}", bbox_inches="tight")
    plt.close(fig)


def _write_report(primary, sensitivity, report_path):
    lines = [
        "# Analysis summary", "",
        "Primary random-effects meta-analysis (REML; positive Hedges g favours musicians):",
        "", "| Outcome | Labs | Hedges g | 95% CI | p | I2 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for r in primary.to_dict("records"):
        lines.append(
            f"| {r['label']} | {int(r['k'])} | {r['estimate']:.3f} "
            f"| [{r['ci_low']:.3f}, {r['ci_high']:.3f}] | {r['p']:.4g} "
            f"| {r['I2']:.1f}% |"
        )
    lines.extend(["", "Sensitivity analysis after applying participant criteria:", ""])
    for r in sensitivity.to_dict("records"):
        lines.append(
            f"- {r['label']}: g={r['estimate']:.3f}, "
            f"95% CI [{r['ci_low']:.3f}, {r['ci_high']:.3f}], p={r['p']:.4g}."
        )
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_analyses(clean_path: Path, full_path: Path | None, project_root: Path,
                 output_root: Path, n_simulations: int = 5000,
                 include_models: bool = True, seed: int = 20261126):
    plots = output_root / "plots"
    tables = output_root / "tables"
    reports = output_root / "reports"
    for folder in (plots, tables, reports):
        folder.mkdir(parents=True, exist_ok=True)

    data = pd.read_csv(clean_path, low_memory=False)
    data["group"] = data["group"].astype("string").str.casefold()
    for outcome in PRIMARY:
        data[outcome] = pd.to_numeric(data[outcome], errors="coerce")
    dataset_rows = [_dataset_counts(data, "main")]
    if full_path and full_path.exists() and full_path.resolve() != clean_path.resolve():
        full_data = pd.read_csv(full_path, low_memory=False)
        dataset_rows.append(_dataset_counts(full_data, "full"))
    pd.DataFrame(dataset_rows).to_csv(tables / "dataset_counts.csv", index=False)
    _descriptives(data, tables)

    primary_rows, sens_rows, unit_rows, sens_unit_rows = [], [], [], []
    for outcome, label in PRIMARY.items():
        pooled, studies = pooled_outcome(data, outcome, label)
        primary_rows.append(pooled)
        unit_rows.append(studies)
        if "to_keep" in data:
            keep = data["to_keep"].astype("string").str.casefold().isin({"true", "1", "yes"})
            subset = data.loc[keep]
        else:
            subset = data
        sens, sens_studies = pooled_outcome(subset, outcome, label)
        sens_rows.append(sens)
        sens_unit_rows.append(sens_studies)
        _forest(studies, pooled, label, plots / f"forest_{outcome}")

    primary, sensitivity = pd.DataFrame(primary_rows), pd.DataFrame(sens_rows)
    primary.to_csv(tables / "meta_primary.csv", index=False)
    sensitivity.to_csv(tables / "meta_sensitivity.csv", index=False)
    pd.concat(unit_rows, ignore_index=True).to_csv(tables / "meta_primary_by_unit.csv", index=False)
    pd.concat(sens_unit_rows, ignore_index=True).to_csv(tables / "meta_sensitivity_by_unit.csv", index=False)
    _write_report(primary, sensitivity, reports / "analysis_summary.md")

    _outcome_plot(data, plots)
    _exploratory(data, tables, plots)
    _correlations(data, plots, tables)
    _prior_power(tables, plots)
    if n_simulations > 0:
        _posthoc_power(data, tables, n_simulations, seed)
    _literature_figure(project_root, plots)

    if include_models:
        from models import run_mixed_models
        run_mixed_models(data, tables, plots)
    LOG.info("Wrote analysis results under %s", output_root)



