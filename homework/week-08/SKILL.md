---
name: lda-qda-analysis
description: Compare LDA/QDA classifiers with CV and a one-page report.
---

# LDA QDA analysis

## Rules

- At most 10 candidate configurations, including ordinary LDA and QDA on all usable original covariates. Count each classifier/representation/dimension, including failures; exclude fold fits and final refit.
- Report every data change, reason, affected variables/counts and recoding mappings/levels. Keep a reproducible record; summarize changes in report. State when none.
- Report <=1 A4 page total. Render/check pagination; Markdown alone cannot verify it. Reference code/diagnostics/overflow tables in analysis artifact. Disclose unavailable results, failures or incomplete CV.

## 1. Explore and prepare

Identify outcome/predictors; ask if unclear. Inspect dimensions, types, rows, summaries, classes, distributions, outliers, duplicates and missingness counts/patterns/codes.

If missingness exists, ask: “Is at least missing at random (MAR) reasonable: missingness may depend on observed information, but not on missing values conditional on observed information?” Explain that MCAR is stronger and observed data alone cannot establish MAR. Wait before treatment; continue exploration. No automatic deletion/imputation; MAR alone does not justify treatment.

Recode outcome as factor/categorical; preserve labels/missing labels, record levels/positive class, check unintended NAs. Summarize exploration/questions.

## 2. Compare models

Reserve test data before fitting; honor existing splits. Choose 5, 10 or 20 folds; stratify independent observations or use group/time splits. Share training-only folds across candidates. Check fold feasibility; ask if no allowed count works. Record seed, priors, warnings/failures.

Fit ordinary LDA/QDA on all usable covariates. Check constants/dependence/covariance rank; QDA needs more training cases per class than predictors and full-rank class covariance. Report infeasibility, no silent substitution.

For many covariates, try LDA/QDA with high-variance subsets or leading PCs at several dimensions. Record variables/variances, PCA centering/scaling and explained variance. Variance depends on units; standardized variances cannot rank features. High variance/PCs need not separate classes.

Fit imputation/selection/scaling/PCA/embedding only within fold training; apply to held-out cases. Exclude outcome from selection/PCA. Record each fold’s held-out accuracy/error/full matrix, predictions and preprocessing.

Select lowest mean fold error (=1-mean fold accuracy), not pooled accuracy; declare ties. Lock choice, refit selected pipeline on all training data, evaluate once on test. No test-based tuning; selection CV can be optimistic.

## 3. Report

### Data and procedure: list with these headings

- **Variables:** Original predictor names/types; outcome name followed by **(outcome)** and original type.
- **Missing value:** **no**, or briefly describe affected variables and treatment.
- **Recoding:** Only recoded variables and mappings, including outcome levels.
- **CV:** Seed, training/testing sizes, fold count, split handling, shared folds, fold-fitted preprocessing and each iteration’s accuracy/error/matrix calculation. No fold-count rationale.
- **Compared models:** Briefly describe every candidate (all 10 if tried), including ordinary LDA/QDA, dimensions and variance/PCA procedure.

### Results

Table: each candidate’s mean fold accuracy (%), no training metrics/matrices. Binary confusion cells: TN, FP, FN, TP as decimal mean fold counts with SE = sample SD of fold counts / sqrt(k). Multiclass: same mean/SE for every cell, including zeros; rows actual, columns predicted, stable labeled class order. Label mean fold counts, not pooled totals. SEs are descriptive: folds vary in size/composition and overlap in training, so no independent confidence intervals. Retain full outputs in artifact.

### Selected model and estimated prediction error: numbered list

1. **Chosen model:** Classifier and representation/dimension; no repeated CV error.
2. **Refitting and application:** Refit preprocessing/model on whole training set, then apply once to reserved testing set; no training performance.
3. **Testing performance:** Error and accuracy percentages with incorrect/total and correct/total counts.
4. **Testing confusion matrix and error types:** State negative/positive classes; explicitly label integer counts in TN, FP, FN, TP order. Type I (false positive) = FP/(TN+FP); Type II (false negative) = FN/(FN+TP). Give percentages and counts; zero denominator means undefined. Multiclass: full labeled matrix and per-class one-versus-rest rates. If testing labels unavailable, mark metrics unavailable, not CV-as-test.

### Main findings and limitations

Explain preference, reduction effects, class errors and limitations: size, imbalance, assumptions, stability/generalization. Note selection optimism; small differences do not prove superiority.
