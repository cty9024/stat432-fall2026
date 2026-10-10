---
name: lda-qda-analysis
summary: Reusable, explicitly invoked LDA/QDA classification analysis and one-page report.
---

# LDA/QDA Classification Analysis

## Activation
Use this skill ONLY when the user explicitly asks to use this SKILL.md. Never auto-apply it. The user supplies the dataset/path, response column, predictor columns, and any required coding. Do not assume any dataset or variable names.

## Objective
Compare ordinary LDA and ordinary QDA on held-out classification performance, optionally including PCA+LDA or PCA+QDA. Select by training-only cross-validation (CV), then evaluate the selected model once on a reserved test set. Report actual computed results; never fabricate scores.

## Data and split
1. Validate response, predictor types, missing values, class counts, and sample size. Respect user-specified predictor coding. Report any exclusions or imputation.
2. Set and report a reproducible integer random seed (default 432). Before fitting, reserve a stratified 20% test set (unless the user provides a designated test set). Do not use it for any selection or revisions.
3. Use stratified 5-fold CV on the 80% training set; use the SAME fold assignments for every candidate and hyperparameter setting. If class counts are too small for 5 folds, lower the number of folds, document why, and keep it consistent.

## Models and preprocessing
- Always include ordinary LDA and ordinary QDA with empirical class priors estimated from each training fold, without regularization.
- Optionally test PCA+LDA and PCA+QDA using a small, feasible grid of PCA component counts selected entirely by training CV. Standardize continuous predictors for PCA. Fit the scaler, imputer (if needed), PCA, and classifier INSIDE a pipeline so each CV fold learns preprocessing only from that fold's training observations.
- For ordinary LDA/QDA, use numeric predictors and no unnecessary scaling. Handle missingness in a train-only fitted pipeline if necessary. Do not treat test data as training data.
- Use only LDA, QDA, or a combination of their class probabilities; no unrelated classifier. If any model fails owing to singular covariance or other numerical issues, report the failure rather than silently replacing it with regularized QDA.

## Selection and evaluation
1. Optimize mean CV misclassification rate. Display mean and, if space allows, SD across folds for each candidate and chosen parameter setting. Break ties by preferring the simpler model (LDA, then QDA, then PCA extensions), and state the rule.
2. Refit the selected preprocessing and classifier on ALL training observations.
3. Predict ONCE on the reserved test set and report test misclassification rate = incorrect predictions / test observations, along with counts, and optionally a small confusion matrix if space permits.
4. The test score is an estimate, not a guarantee of future performance; note uncertainty due to a single split and the limited sample size.

## Report: at most one printed page
Produce a concise, legible report (PDF or print-ready HTML) with: (a) title, dataset/task summary, sample sizes, seed, and split; (b) short candidate-model and identical-fold CV description; (c) a small TABLE of numeric mean CV errors (and optional SD) for candidates; (d) a FIGURE comparing the candidates' CV errors, with clear labels; (e) selected model, chosen settings, final held-out test error and numerator/denominator; (f) 2-3 sentences on key findings and limitations. Use readable font sizes and ensure the entire analysis report prints on one page. Keep code and full skill text outside this one-page report.

## Quality checks
Check for overlapping train/test rows, target leakage, inconsistent label encoding, class imbalance, fold reproducibility, failed models, and preprocessing fitted before CV. Never adjust models or this skill in response to test-set results. Clearly distinguish CV performance from held-out test performance.
