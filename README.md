# variability-quantifier
## Project Description
This project's aim is to calculate different aspects of variability for gene expression.
The variability score is calculated on experiment results of single cell mRNA Fluorescence In Situ Hybridization (FISH).
Python is the programming language of this project and it uses diverse packages, including pandas, numpy, statsmodels and more.
So far, the variability metrics avalible are (The log base is 2):
* log Coefficient of Variation (CV)
* log the ratio of the mean expression of top expressing subpopulation, to the bottom expressing subpopulation.
* log kurtosis (Fisher's)
* log entropy
* percent of the population that is expressing the gene
## Installation
The python dependencies are managed via conda and can be installed using the provided quantify_variability.yaml file:
```
conda env create -f quantify_variability.yaml
conda activate quantify_variability
```

## Usage
Notebook with example usage is included.
## Constant parameters
- NORMALIZED_DATA_PATH - cell by gene file or files location
- CHANNELS_FILE_PATH - genes to readouts and channels file or files location
- GENE_FIRST_COL - first gene column in the cell by gene file
- CONTROL_PER_LIBRARY - par-seq: False and empty "CONTROL_GENES_ETC" list. par-squared: True or False, filled "CONTROL_GENES_ETC" list
- CONTROL_GENES_ETC - list of special genes
- LOAD_FILE - assign True if there is already "statistics_summary.csv" file in your directory
- GENES_TO_EXCLUDE - list of genes to exclude from analysis
- Mean expression cutoff where genes below min and above max are filtered, for each channel:
- MIN_MEAN_EXPRESSION_THRESH_A488
- MIN_MEAN_EXPRESSION_THRESH_A550
- MIN_MEAN_EXPRESSION_THRESH_A647
- MAX_MEAN_EXPRESSION_THRESH_A488
- MAX_MEAN_EXPRESSION_THRESH_A550
- MAX_MEAN_EXPRESSION_THRESH_A647
- SD_ABOVE_MEAN_THRESH - if cell is expressing gene higher than this constant, this measurement is dropped
- MINIMUM_CELLS_EXPRESSING - genes that expressed by lower number of cells than this constant are filtered
- RESIDUALS_X_AXIS - x-axis in the residual analysis
- RESIDUALS_Y_AXIS - list of metrics to calculate the residuals when they are the y-axis
- TOP_TO_BOTTOM_RATIO_TAIL - size of the subpopulations in tails
- FIT_PER_CHANNEL - calculate the residuals from different fit for each channel
- ROBUST_RESIDUALS - calculate residuals from the closest confidence interval
- CONFIDENCE_LEVEL - (1-alpha) for the confidence intervals
- POLYNOMIAL_DEGREES - dictionary which maps from metric to the degree of polynomial that is best describes it's relationship with log mean of expression
## Acknowledgements
