
---
name: clustering-analysis
description: Explore, compare, and select clustering methods and generate a concise one-page analysis report. Use only when explicitly invoked.
disable-model-invocation: true
---

# Clustering Analysis Skill

## 1. Inputs

Obtain the dataset, feature columns, optional observation identifiers, optional known class labels, and output location from the invocation. Do not hard-code dataset-specific information.

Treat each row as one observation. Exclude identifiers and known class labels from clustering features.

## 2. Data Preparation

- Inspect data dimensions, missing values, constant features, outliers, and feature scales.
- Handle missing values and remove noninformative features when appropriate. Document preprocessing decisions.
- Standardize features when differences in scale may dominate distance calculations.
- For high-dimensional data, consider PCA to reduce dimensionality and noise.
- Compare the original preprocessed features with PCA representations retaining approximately 80%, 90%, or 95% of total variance when appropriate.
- Do not assume dimension reduction necessarily improves clustering.

## 3. Clustering Methods and Tuning

Evaluate at least three distinct clustering algorithms:

1. K-means: tune the number of clusters and use multiple random initializations.
2. Hierarchical clustering: tune the number of clusters and compare Ward, complete, and average linkage where applicable.
3. Spectral clustering: tune the number of clusters, affinity type, and relevant affinity parameters.

Start with candidate cluster counts from 2 to 6 and adjust the range according to sample size and data characteristics.

Use fixed random seeds for reproducibility. Check for convergence, empty clusters, singleton clusters, and numerical problems. Do not count PCA or other embeddings as clustering algorithms.

## 4. Evaluation and Model Selection

- Use the mean silhouette score as the primary internal evaluation metric.
- Use the Calinski-Harabasz and Davies-Bouldin indices as supporting metrics.
- Compare candidate partitions in a common evaluation space using a consistent distance metric.
- Assess clustering stability across random seeds or repeated subsamples when feasible.
- Prefer well-separated, stable, and interpretable solutions. Avoid selecting a model based solely on one metric or smaller within-cluster sum of squares.
- Select exactly one final algorithm, preprocessing configuration, hyperparameter setting, and number of clusters.
- Explain the selection using the evaluation criteria and relevant trade-offs.

## 5. External Evaluation

If known class labels are available, use them only after the final clustering model has been selected.

Evaluate agreement using the Adjusted Rand Index (ARI) and a contingency table. Optionally report normalized mutual information (NMI).

Do not use known labels to fit models, tune hyperparameters, select the number of clusters, or choose the final method. Distinguish discovered clusters from known classes.

## 6. Visualization and Interpretation

- Produce a two-dimensional PCA scatterplot colored by the selected cluster assignments.
- Report the explained variance of the displayed principal components.
- Use other visualizations, such as a dendrogram, when more appropriate for the selected method.
- If known labels are available, optionally distinguish them using point shapes.
- Describe cluster sizes, separation, overlap, and agreement with known labels when available.
- Discuss relevant limitations, including sensitivity to scaling, dimensionality, sample size, and algorithm assumptions.
- Do not interpret a two-dimensional projection as definitive evidence of separation in the full clustering space.


## 7. Final Report

Generate a concise, clearly readable analysis report limited
to one printed page, including all text, figures, and tables.

Include:
- Analysis objective and preprocessing summary.
- Algorithms considered and model selection criteria.
- Selected algorithm, hyperparameters, and number of clusters.
- Brief justification for the selected method.
- One meaningful visualization of the selected clustering.
- Selected model evaluation metrics and interpretation.
- Relevant limitations.

Present clustering results only for the selected algorithm.
Other methods may be briefly mentioned to justify selection,
but do not display their scores, assignments, figures,
or result tables.

Use readable fonts and compact formatting. Render the report
and verify that the analysis fits within one printed page.

Export the report as a PDF or self-contained HTML file,
using the output filename specified in the invocation.

Append the complete, unchanged skill definition after the
one-page analysis report. The appendix is outside the
one-page limit.

Keep analysis code and intermediate comparisons separate
from the final report. Ensure reproducibility and do not
fabricate results.