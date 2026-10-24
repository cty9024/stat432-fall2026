from __future__ import annotations

import html
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import RepeatedStratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


SEED = 432
TEST_SIZE = 0.25
CV_SPLITS = 5
CV_REPEATS = 3
C_VALUES = np.logspace(-2, 2, 5)
GAMMA_VALUES = np.logspace(-3, 0, 4)
PREDICTORS = [
    "sbp",
    "tobacco",
    "ldl",
    "adiposity",
    "famhist",
    "typea",
    "obesity",
    "alcohol",
    "age",
]
DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "saheart.csv"
OUTPUT_DIR = Path(__file__).resolve().parent


def load_data() -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    data = pd.read_csv(DATA_PATH)
    famhist = data["famhist"].astype("string").str.strip().str.lower()
    data["famhist"] = famhist.map({"absent": 1, "present": 2})
    if data[PREDICTORS + ["chd"]].isna().any().any():
        raise ValueError("Missing values or unrecognized famhist labels found.")
    X = data[PREDICTORS].apply(pd.to_numeric, errors="raise").to_numpy(float)
    y = pd.to_numeric(data["chd"], errors="raise").to_numpy(int)
    if set(np.unique(y)) != {0, 1}:
        raise ValueError("Expected chd to contain only 0 and 1.")
    return data, X, y


def build_candidates() -> list[dict[str, float | int | str]]:
    candidates: list[dict[str, float | int | str]] = []
    for cost in C_VALUES:
        candidates.append({"kernel": "linear", "C": float(cost)})
    for gamma in GAMMA_VALUES:
        for cost in C_VALUES:
            candidates.append(
                {"kernel": "rbf", "C": float(cost), "gamma": float(gamma)}
            )
    for degree in (2, 3):
        for gamma in GAMMA_VALUES:
            for cost in C_VALUES:
                candidates.append(
                    {
                        "kernel": "poly",
                        "degree": degree,
                        "C": float(cost),
                        "gamma": float(gamma),
                        "coef0": 1.0,
                    }
                )
    # The initial quadratic optimum was on both the lower-C and upper-gamma edges.
    candidates.extend(
        [
            {"kernel": "poly", "degree": 2, "C": 0.001, "gamma": 1.0, "coef0": 1.0},
            {"kernel": "poly", "degree": 2, "C": 0.01, "gamma": 10.0, "coef0": 1.0},
        ]
    )
    return candidates


def make_pipeline(params: dict[str, float | int | str]) -> Pipeline:
    svc_params = {key: value for key, value in params.items() if key != "kernel"}
    return Pipeline(
        [
            ("scale", StandardScaler()),
            ("svc", SVC(kernel=str(params["kernel"]), **svc_params)),
        ]
    )


def parameter_label(params: dict[str, object]) -> str:
    values = [f"C={params['C']:.3g}"]
    if "degree" in params:
        values += [f"degree={params['degree']}", f"gamma={params['gamma']:.3g}", "coef0=1"]
    elif "gamma" in params:
        values += [f"gamma={params['gamma']:.3g}"]
    return ", ".join(values)


def fmt_rate(value: float) -> str:
    return f"{100 * value:.1f}%"


def write_figure(results: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(9.4, 5.5), sharey=True)
    panels = [
        ("linear", None, "Linear"),
        ("poly", 2, "Polynomial, degree 2"),
        ("poly", 3, "Polynomial, degree 3"),
        ("rbf", None, "Radial basis (RBF)"),
    ]
    palette = plt.get_cmap("viridis")
    all_gammas = sorted(results["gamma"].dropna().unique())
    gamma_colors = {gamma: palette(i / max(len(all_gammas) - 1, 1)) for i, gamma in enumerate(all_gammas)}

    for axis, (kernel, degree, title) in zip(axes.flat, panels):
        subset = results[results["kernel"] == kernel]
        if degree is not None:
            subset = subset[subset["degree"] == degree]
        if kernel == "linear":
            axis.errorbar(
                subset["C"], subset["mean_error"] * 100,
                yerr=subset["se"] * 100, marker="o", capsize=2,
                color="#16697a", linewidth=1.3,
            )
        else:
            for gamma in sorted(subset["gamma"].dropna().unique()):
                curve = subset[np.isclose(subset["gamma"], gamma)].sort_values("C")
                axis.errorbar(
                    curve["C"], curve["mean_error"] * 100,
                    yerr=curve["se"] * 100, marker="o", capsize=2,
                    linewidth=1.0, color=gamma_colors[gamma],
                    label=f"γ={gamma:g}",
                )
            axis.legend(title="Gamma", fontsize=7, title_fontsize=7, ncol=2, frameon=False)
        axis.set_xscale("log")
        axis.set_title(title, fontsize=10, loc="left")
        axis.set_xlabel("Cost C (log scale)")
        axis.grid(axis="y", alpha=0.2)
    axes[0, 0].set_ylabel("Repeated CV misclassification (%)")
    axes[1, 0].set_ylabel("Repeated CV misclassification (%)")
    fig.suptitle("SVM tuning on training data: mean error ± 1 SE", fontsize=12, x=0.04, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(OUTPUT_DIR / "cv_tuning.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def render_report(summary: dict[str, object], kernel_table: pd.DataFrame) -> None:
    table_html = kernel_table.to_html(index=False, escape=True, border=0, classes="results")
    m = summary
    chosen = html.escape(str(m["chosen_description"]))
    cm = m["confusion_matrix"]
    report = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Heart disease SVM analysis</title>
<style>
@page {{ size: letter; margin: 0.42in; }}
:root {{ color-scheme: light; }}
* {{ box-sizing: border-box; }}
body {{ margin: 0 auto; max-width: 8.5in; color: #202b2f; font: 11pt/1.16 Georgia, 'Times New Roman', serif; }}
h1 {{ margin: 0 0 3pt; font: 700 20pt/1.05 'Avenir Next', 'Helvetica Neue', sans-serif; color: #113b46; }}
.dek {{ margin: 0 0 9pt; color: #47636a; font: 9pt 'Avenir Next', 'Helvetica Neue', sans-serif; }}
h2 {{ margin: 5pt 0 2pt; font: 700 10pt 'Avenir Next', 'Helvetica Neue', sans-serif; color: #16697a; }}
p {{ margin: 2pt 0; }}
.result {{ padding: 5pt 7pt; border-left: 3px solid #e09f3e; background: #f4f6f4; }}
table.results {{ width: 100%; border-collapse: collapse; margin: 2pt 0 1pt; font: 7.8pt 'Avenir Next', 'Helvetica Neue', sans-serif; }}
.results th {{ text-align: left; color: #fff; background: #16697a; font-weight: 600; }}
.results th, .results td {{ padding: 2pt 4pt; border-bottom: 1px solid #d8e0df; }}
.results tr:last-child td {{ border-bottom: 0; }}
figure {{ margin: 2pt 0; text-align: center; }}
figure img {{ width: 100%; max-height: 2.62in; object-fit: contain; }}
figcaption {{ font-size: 8pt; color: #52666a; text-align: left; }}
.metrics {{ display: grid; grid-template-columns: 1fr 1fr; gap: 0 14pt; }}
.small {{ font-size: 9pt; }}
footer {{ margin-top: 3pt; padding-top: 2pt; border-top: 1px solid #cad6d5; font-size: 8pt; color: #52666a; }}
@media screen {{ body {{ padding: 0.4in; }} }}
@media print {{ body {{ max-width: none; }} }}
</style>
</head>
<body>
<h1>Heart disease, one boundary at a time</h1>
<p class="dek">SVM model selection | SAheart data | {m['n']} observations, {m['p']} predictors</p>
<p class="result"><strong>Selected:</strong> {chosen}. Repeated-CV error was {fmt_rate(float(m['chosen_cv_error']))} ± {fmt_rate(float(m['chosen_cv_se']))} SE. This is the simplest candidate within one SE of the lowest observed CV error.</p>
<h2>Data and selection</h2>
<p class="small">CHD prevalence was {fmt_rate(float(m['chd_prevalence']))} ({m['class_counts']['1']}/{m['n']}); missing cells: {m['missing_cells']}; rows with a 1.5-IQR predictor outlier: {m['any_outlier_rows']}. SVM is reasonable for this moderate-sized binary task, with fold-local scaling handling predictor scales. Stratified holdout: {m['split_percent']}% (seed {SEED}); training selection: {CV_SPLITS}-fold CV x {CV_REPEATS} repeats. Kernels: linear baseline, RBF smooth boundary, and degree-2/3 polynomial interactions; base C={', '.join(f'{x:g}' for x in C_VALUES)}, gamma={', '.join(f'{x:g}' for x in GAMMA_VALUES)}, polynomial coef0=1. Two quadratic edge probes: (C=0.001, gamma=1) and (C=0.01, gamma=10). Table SE uses the {CV_SPLITS * CV_REPEATS} fold errors.</p>
{table_html}
<figure><img src="cv_tuning.png" alt="Repeated cross-validation error with one-standard-error bars by SVM tuning parameters"><figcaption>Figure 1. Mean training-set repeated-CV error ± 1 fold-level SE. Gamma is color-coded; cost is on a logarithmic axis. The held-out test data did not inform tuning.</figcaption></figure>
<h2>Held-out test performance</h2>
<div class="metrics"><p class="small">Test error: <strong>{fmt_rate(float(m['test_error']))}</strong> ({m['test_errors']} / {m['test_n']})<br>Confusion matrix (actual rows 0, 1; predicted columns 0, 1):<br>[[{cm[0][0]}, {cm[0][1]}], [{cm[1][0]}, {cm[1][1]}]]<br>False positives: {m['fp']}; false negatives: {m['fn']}</p><p class="small">Sensitivity: {fmt_rate(float(m['sensitivity']))}; specificity: {fmt_rate(float(m['specificity']))}. In a CHD screening context, false negatives can miss people who need follow-up and may be costlier, while false positives can prompt avoidable assessment. The threshold was not altered to encode these costs.</p></div>
<p class="small">The fit uses {m['support_vectors']} support vectors ({m['support_vector_percent']:.1f}% of training data). {html.escape(str(m['boundary_note']))} RBF had the lowest CV error, but the linear model was within one SE; the winner was not clearly separated. This is predictive, not causal, and the selected kernel does not support simple predictor-effect interpretation.</p>
<footer>Reproducible code: analyze_svm.py | Full per-setting numerical results: cv_results.csv | Random seed: {SEED}. Test set evaluated once after model selection.</footer>
</body></html>"""
    (OUTPUT_DIR / "report.html").write_text(report, encoding="utf-8")

    markdown = f"""# Heart disease SVM analysis

**Selected:** {m['chosen_description']}. Repeated-CV error: {fmt_rate(float(m['chosen_cv_error']))} +/- {fmt_rate(float(m['chosen_cv_se']))} SE. The chosen model is the simplest candidate within one SE of the minimum mean CV error.

## Data and selection

The dataset has {m['n']} observations and {m['p']} predictors. CHD prevalence is {fmt_rate(float(m['chd_prevalence']))} ({m['class_counts']['1']} cases; {m['class_counts']['0']} non-cases), with {m['missing_cells']} missing cells and {m['any_outlier_rows']} rows having at least one 1.5-IQR predictor outlier. This moderate-sized, nine-predictor binary classification task is reasonable for SVM comparison. Predictors were standardized within each training fold. A stratified {m['split_percent']}% test set was held out with seed {SEED}; training-set selection used {CV_SPLITS}-fold CV repeated {CV_REPEATS} times.

Candidates were linear (simple baseline), RBF (smooth flexible boundary), and degree-2/3 polynomial (low-order interactions). Base C grid: {', '.join(f'{x:g}' for x in C_VALUES)}; base RBF/polynomial gamma grid: {', '.join(f'{x:g}' for x in GAMMA_VALUES)}; polynomial coef0=1. Two extra quadratic settings checked the prior boundary result (C=0.001 at gamma=1; gamma=10 at C=0.01). The table shows the best setting per kernel/degree; SE is the sample standard deviation of its {CV_SPLITS * CV_REPEATS} fold errors divided by the square root of the fold count.

{kernel_table.to_markdown(index=False)}

![Repeated CV error versus tuning parameters, with +/-1 SE bars](cv_tuning.png)

## Test evaluation

Test misclassification rate: {fmt_rate(float(m['test_error']))} ({m['test_errors']}/{m['test_n']}). Confusion matrix (actual rows 0,1; predicted columns 0,1): `[[{cm[0][0]}, {cm[0][1]}], [{cm[1][0]}, {cm[1][1]}]]`. FP={m['fp']}; FN={m['fn']}; sensitivity={fmt_rate(float(m['sensitivity']))}; specificity={fmt_rate(float(m['specificity']))}.

For CHD screening, false negatives may be costlier because missed cases could delay follow-up; false positives can lead to unnecessary assessment. This comparison uses the default classification threshold. The selected fit has {m['support_vectors']} support vectors ({m['support_vector_percent']:.1f}% of training cases). {m['boundary_note']} {m['winner_note']} Results are predictive rather than causal; the selected kernel does not support simple individual-effect interpretation.

**Artifacts:** `analyze_svm.py` contains the complete analysis; `cv_results.csv` has all {m['candidate_count']} candidate settings and fold-level results; `summary.json` contains the computed summary; `cv_tuning.png` is the tuning figure. The test set was evaluated once after selection.
"""
    (OUTPUT_DIR / "report.md").write_text(markdown, encoding="utf-8")


def main() -> None:
    data, X, y = load_data()
    indices = np.arange(len(y))
    train_idx, test_idx = train_test_split(
        indices, test_size=TEST_SIZE, random_state=SEED, stratify=y
    )
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    cv = RepeatedStratifiedKFold(
        n_splits=CV_SPLITS, n_repeats=CV_REPEATS, random_state=SEED
    )
    splits = list(cv.split(X_train, y_train))
    scaled_splits = []
    for fit_idx, valid_idx in splits:
        scaler = StandardScaler().fit(X_train[fit_idx])
        scaled_splits.append(
            (
                scaler.transform(X_train[fit_idx]),
                scaler.transform(X_train[valid_idx]),
                y_train[fit_idx],
                y_train[valid_idx],
            )
        )

    records: list[dict[str, object]] = []
    for params in build_candidates():
        fold_errors = []
        svc_params = {key: value for key, value in params.items() if key != "kernel"}
        for X_fit, X_valid, y_fit, y_valid in scaled_splits:
            model = SVC(kernel=str(params["kernel"]), **svc_params)
            model.fit(X_fit, y_fit)
            fold_errors.append(float(np.mean(model.predict(X_valid) != y_valid)))
        record: dict[str, object] = dict(params)
        record["mean_error"] = float(np.mean(fold_errors))
        record["se"] = float(np.std(fold_errors, ddof=1) / np.sqrt(len(fold_errors)))
        record["fold_errors"] = json.dumps(fold_errors)
        records.append(record)
    results = pd.DataFrame(records)
    results["degree"] = results["degree"].astype("Int64")
    results.to_csv(OUTPUT_DIR / "cv_results.csv", index=False)

    family_best: dict[str, pd.Series] = {}
    family_best["Linear"] = results[results.kernel == "linear"].sort_values("mean_error").iloc[0]
    for degree in (2, 3):
        family_best[f"Polynomial (degree {degree})"] = (
            results[(results.kernel == "poly") & (results.degree == degree)]
            .sort_values("mean_error")
            .iloc[0]
        )
    family_best["RBF"] = results[results.kernel == "rbf"].sort_values("mean_error").iloc[0]
    best_overall = results.loc[results.mean_error.idxmin()]
    threshold = float(best_overall.mean_error + best_overall.se)
    eligible = results[results.mean_error <= threshold].copy()
    complexity = {"linear": 0, "poly": 1, "rbf": 2}
    eligible["complexity"] = eligible.kernel.map(complexity)
    eligible["degree_order"] = eligible.degree.fillna(0)
    chosen = eligible.sort_values(["complexity", "degree_order", "mean_error"]).iloc[0]
    chosen_params = {key: chosen[key] for key in ["kernel", "C", "gamma", "degree", "coef0"] if key in chosen.index and pd.notna(chosen[key])}
    fitted = make_pipeline(chosen_params)
    fitted.fit(X_train, y_train)
    predictions = fitted.predict(X_test)
    matrix = confusion_matrix(y_test, predictions, labels=[0, 1])
    tn, fp, fn, tp = (int(value) for value in matrix.ravel())
    test_error = float(np.mean(predictions != y_test))
    sensitivity = tp / (tp + fn) if tp + fn else float("nan")
    specificity = tn / (tn + fp) if tn + fp else float("nan")
    support_vectors = int(fitted.named_steps["svc"].n_support_.sum())

    numeric = data[PREDICTORS].copy()
    outlier_by_feature = {}
    for col in numeric.columns:
        low = numeric[col].quantile(0.25)
        high = numeric[col].quantile(0.75)
        spread = high - low
        outlier_by_feature[col] = int(((numeric[col] < low - 1.5 * spread) | (numeric[col] > high + 1.5 * spread)).sum())
    any_outlier_rows = int((numeric.apply(lambda col: (col < col.quantile(0.25) - 1.5 * (col.quantile(0.75) - col.quantile(0.25))) | (col > col.quantile(0.75) + 1.5 * (col.quantile(0.75) - col.quantile(0.25)))).any(axis=1)).sum())
    class_counts = {str(label): int(count) for label, count in pd.Series(y).value_counts().sort_index().items()}

    chosen_kernel = str(chosen.kernel)
    chosen_description = f"{chosen_kernel.upper()} SVM ({parameter_label(chosen_params)})"
    all_family_errors = [float(row.mean_error) for row in family_best.values()]
    if chosen_kernel == "linear":
        boundary_note = "A linear kernel gives a linear boundary in the standardized predictor space."
    else:
        boundary_note = "The selected " + ("RBF" if chosen_kernel == "rbf" else f"degree-{int(chosen.degree)} polynomial") + " kernel allows a nonlinear boundary."
    gap = float(max(all_family_errors) - min(all_family_errors))
    if gap < 0.01:
        winner_note = "The best kernel-family CV errors were within one percentage point, so the apparent winner was not clearly separated."
    else:
        winner_note = f"The range between the best and worst family-level CV errors was {100 * gap:.1f} percentage points; treat CV differences in light of their fold-level SEs."

    summary: dict[str, object] = {
        "data_file": str(DATA_PATH.relative_to(DATA_PATH.parent.parent)),
        "n": int(len(y)),
        "p": int(X.shape[1]),
        "missing_cells": int(data[PREDICTORS + ["chd"]].isna().sum().sum()),
        "class_counts": class_counts,
        "chd_prevalence": float(np.mean(y)),
        "predictor_outliers_by_feature": outlier_by_feature,
        "any_outlier_rows": any_outlier_rows,
        "split_percent": int(round(TEST_SIZE * 100)),
        "seed": SEED,
        "train_n": int(len(train_idx)),
        "test_n": int(len(test_idx)),
        "cv_splits": CV_SPLITS,
        "cv_repeats": CV_REPEATS,
        "candidate_count": len(results),
        "chosen_description": chosen_description,
        "chosen_params": {key: (int(value) if key == "degree" else float(value) if isinstance(value, (float, np.floating)) else str(value)) for key, value in chosen_params.items()},
        "chosen_cv_error": float(chosen.mean_error),
        "chosen_cv_se": float(chosen.se),
        "best_cv_error": float(best_overall.mean_error),
        "one_se_threshold": threshold,
        "test_error": test_error,
        "test_errors": int(np.sum(predictions != y_test)),
        "confusion_matrix": matrix.tolist(),
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "support_vectors": support_vectors,
        "support_vector_percent": 100 * support_vectors / len(train_idx),
        "boundary_note": boundary_note,
        "winner_note": winner_note,
        "family_best": {
            name: {
                "params": {key: (int(row[key]) if key == "degree" else float(row[key]) if pd.notna(row[key]) and isinstance(row[key], (float, np.floating)) else str(row[key])) for key in ["kernel", "C", "gamma", "degree", "coef0"] if key in row.index and pd.notna(row[key])},
                "mean_error": float(row.mean_error),
                "se": float(row.se),
            }
            for name, row in family_best.items()
        },
    }
    (OUTPUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")

    table_rows = []
    for name, row in family_best.items():
        matches_chosen = row.kernel == chosen.kernel and (
            (pd.isna(row.degree) and pd.isna(chosen.degree))
            or (pd.notna(row.degree) and pd.notna(chosen.degree) and row.degree == chosen.degree)
        )
        display_row = chosen if matches_chosen else row
        params = {key: display_row[key] for key in ["kernel", "C", "gamma", "degree", "coef0"] if key in display_row.index and pd.notna(display_row[key])}
        support_count = int(make_pipeline(params).fit(X_train, y_train).named_steps["svc"].n_support_.sum())
        table_rows.append({
            "Kernel family": name,
            "Best parameters": parameter_label(params),
            "Mean CV error ± SE": f"{fmt_rate(float(display_row.mean_error))} ± {fmt_rate(float(display_row.se))}",
            "Support vectors": support_count,
            "Selection": "CHOSEN" if matches_chosen else "",
        })
    kernel_table = pd.DataFrame(table_rows)
    write_figure(results)
    render_report(summary, kernel_table)

    print(json.dumps({
        "n": summary["n"],
        "class_counts": class_counts,
        "missing_cells": summary["missing_cells"],
        "any_outlier_rows": any_outlier_rows,
        "candidate_count": summary["candidate_count"],
        "family_best": summary["family_best"],
        "chosen_description": chosen_description,
        "chosen_cv_error": summary["chosen_cv_error"],
        "chosen_cv_se": summary["chosen_cv_se"],
        "test_error": test_error,
        "confusion_matrix": matrix.tolist(),
        "fp": fp,
        "fn": fn,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "support_vectors": support_vectors,
        "outputs": ["cv_results.csv", "summary.json", "cv_tuning.png", "report.md", "report.html"],
    }, indent=2))


if __name__ == "__main__":
    main()
