---
name: svm-fit
description: Fit and justify an SVM model (kernel choice, tuning, CV-based selection, one-page report). Use only when explicitly invoked.
disable-model-invocation: true
---

# SVM Model Fitting

The user supplies the dataset, response, predictors, and coding when invoking. Assume nothing about them.

## 1. Data suitability
- Check n, p, response type, class balance, missing values, outliers, predictor types and scales.
- Code categorical predictors numerically as the user specifies. Standardize predictors inside each training fold only.
- State in 1-2 sentences whether SVM is reasonable; flag concerns (severe imbalance, very large n, p >> n).

## 2. Kernels
- Always consider at least: linear, radial (RBF), and polynomial (degree 2 and 3).
- Add other kernels only with a stated reason.
- Justify each in one phrase (linear = simple baseline; RBF = flexible smooth boundary; polynomial = low-order interactions).

## 3. Tuning
- Tune cost C for every kernel (log-spaced grid, at least 5 values); gamma for RBF/polynomial (log-spaced, at least 4 values).
- Use grid search with repeated k-fold CV (e.g., 10-fold x 3+ repeats) and a fixed seed.
- If the best value sits on a grid edge, extend the grid.

## 4. Evaluation and selection
- Before any tuning, set aside a stratified held-out test set; report its proportion and the seed.
- Fit all preprocessing on training folds only; use the test set exactly once, after the final model is chosen.
- Compare candidates by mean CV error AND fold-level SE. Choose the simplest model (linear < polynomial < RBF) within one SE of the best.

## 5. Required figure and table
- One compact figure: CV error vs tuning parameters, by kernel, with +/-1 SE bars.
- One compact table: per kernel, best parameters, mean CV error, SE, number of support vectors; mark the chosen model.

## 6. Interpretation and reporting
- Report the chosen kernel and parameters, CV error +/- SE, test error, and the test confusion matrix (classification).
- Report FP and FN counts, sensitivity, and specificity; discuss which error is costlier in context.
- Interpret the fit: support-vector count, linear vs nonlinear boundary, how clearly the winner beat the others.
- Claim predictor effects only if the kernel supports it (e.g., linear).

## 7. One-page limit
- At most one printed page at 11-point font, figure and table included.
- Priority: (1) final model and rationale, (2) table, (3) figure, (4) test performance with confusion matrix, (5) data suitability, (6) brief caveats. Cut the rest.
- Before finishing, verify every reported number against the code output and confirm the page count by rendering.