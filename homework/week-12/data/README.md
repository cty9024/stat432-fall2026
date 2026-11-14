# Golub gene-expression data

`golub.csv` contains the preprocessed Golub leukemia data distributed with the Bioconductor **multtest 2.68.0** package (Bioconductor 3.23). It has 38 rows, one per biological sample, and 3,053 columns:

- `sample_id`: `sample_01` through `sample_38`, identifying columns 1 through 38 of the original matrix in their original order.
- `leukemia_type`: `ALL` for acute lymphoblastic leukemia (27 samples, source code 0) or `AML` for acute myeloid leukemia (11 samples, source code 1).
- `gene_1` through `gene_3051`: numeric expression measurements, preserving the original gene order.

For clustering and method selection, use only the gene columns. `sample_id` is an identifier. Use `leukemia_type` only to interpret the final selected clustering.

The package's `golub` matrix has 3,051 genes as rows and 38 samples as columns. This CSV transposes that matrix and maps its accompanying `golub.cl` labels to text. No expression values were transformed, rescaled, filtered, or imputed during conversion. Decimal values are written with 17 significant digits, and reading the CSV back was verified to reproduce every source numeric value exactly.

These are already processed expression measurements, not raw microarray intensities. The upstream preprocessing script applies intensity thresholds of 100 and 16,000, retains genes with a maximum/minimum ratio greater than 5 and a maximum-minus-minimum difference greater than 500, takes base-10 logarithms, and centers and scales each sample across the retained genes.

## Source and provenance

- [Bioconductor multtest package](https://bioconductor.org/packages/3.23/bioc/html/multtest.html), version 2.68.0, published April 28, 2026; retrieved October 25, 2026.
- [Exact source archive](https://bioconductor.org/packages/3.23/bioc/src/contrib/multtest_2.68.0.tar.gz). Objects were loaded directly from `multtest/data/golub.RData`, without installing the package. The package description declares license **LGPL**, without a version number; no separate data license is stated in the dataset documentation.
- Source archive SHA-256: `eeba5414411e26b010f8cead730c44dc3ef4ed203629d3acb42a4baa19ce3fd2`.
- [Package reference manual](https://bioconductor.org/packages/3.23/bioc/manuals/multtest/man/multtest.pdf), entry `golub`; the source archive also includes `man/golub.Rd` and preprocessing script `vignettes/golub.R`.
- Original study: Golub et al. (1999), "Molecular classification of cancer: class discovery and class prediction by gene expression monitoring," *Science* 286, 531-537. Preprocessing reference: Dudoit, Fridlyand, and Speed (2002), "Comparison of discrimination methods for the classification of tumors using gene expression data," *Journal of the American Statistical Association* 97, 77-87.
