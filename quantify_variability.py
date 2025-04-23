#############################################################
# Author: Vadim Litvinov
# Date: 22 July 2024
#############################################################
from file_handler import readCsvFile, writeToFile, writeResToFile
from statistical_scores import percentCellsExpressing, meanBottomOrTopPercentileExpression, calculateEntropy,\
    calculateKurtosis, fitPolynomialWithCI, residualsFromConfidenceInterval, calculaterCriticalValuesForFilteringCells
from pathlib import Path
import numpy as np
import pandas as pd
import time
from pandas.core.groupby import DataFrameGroupBy


def excludeGenes(data: pd, genes_to_exclude: list[str]) -> None:
    """
    Excludes control genes etc.
    :param data: the data that needs to be modified
    :param genes_to_exclude: list of genes to drop
    :return: Nothing, the dropping happens inplace
    """
    data.drop(columns=genes_to_exclude, inplace=True, errors='ignore')


def createOpticalDensityColumn(gene_channel_ro: pd.DataFrame, split_per_dilution: bool) -> None:
    """Checks if there are different OD's or not, if there are, create new column of OD. Additionally, create dilution
     column if requested"""
    if split_per_dilution:
        gene_channel_ro['OD'] = gene_channel_ro['sample_name']
    elif pd.notna(gene_channel_ro['sample_name'][0]) and '_' in gene_channel_ro['sample_name'][0]:
        gene_channel_ro['OD'] = gene_channel_ro['sample_name'].str.split('_').str[-1]
    else:
        gene_channel_ro['OD'] = 'constant'


def deriveGeneNameAndLibraryNumber(original_gene_name: str, control_genes_etc: list[str], control_per_library: bool)\
        -> tuple[str, str]:
    """
    Producing gene name and library number
    :param original_gene_name: just the gene name
    :param control_genes_etc: list of the special genes
    :param control_per_library: control genes shared for all libraries or not
    :return: gene name and library number
    """
    lib_num = np.nan
    if original_gene_name.split('__')[0] in control_genes_etc and control_per_library:
        # Only gene name, without library num
        lib_num = original_gene_name.split('__')[1]
        gene_name = original_gene_name.split('__')[0]
    else:
        gene_name = original_gene_name
    return gene_name, lib_num


def deriveGeneLibraryChannelAndReadout(original_gene_name: str, control_genes_etc: list[str], control_per_library: bool,
                                       channels_df: pd.DataFrame) -> tuple[str, str, str, str, int]:
    """Return the properties of the gene"""
    gene_name, lib_num = deriveGeneNameAndLibraryNumber(original_gene_name, control_genes_etc, control_per_library)
    matching_row = channels_df[channels_df['gene_name'] == gene_name]
    channel = matching_row['channel'].values[0]
    readout = matching_row['readout'].values[0]
    probes_num = matching_row['probes_per_gene'].values[0]
    return gene_name, channel, readout, lib_num, probes_num


def addChannelAndReadoutToColumnName(original_gene_name: str, control_genes_etc: list[str], control_per_library: bool,
                                     channels_df: pd.DataFrame, gene_channel_ro: pd.DataFrame,
                                     new_columns: dict[str: str]) -> None:
    """
    Changes the column names
    :param original_gene_name: just the gene name
    :param control_genes_etc: list of the special genes
    :param control_per_library: control genes shared for all libraries or not
    :param channels_df: dataframe that contains genes to channels
    :param gene_channel_ro: the new dataframe that created
    :param new_columns: dictionary that maps old column names to new
    :return: the new dataframe created
    """
    gene_name, channel, readout, lib_num, probes_num = deriveGeneLibraryChannelAndReadout(original_gene_name,
                                                                                          control_genes_etc,
                                                                                          control_per_library,
                                                                                          channels_df)
    if len(control_genes_etc) > 0 and gene_name not in control_genes_etc:
        # First row where there is a valid value for this gene, return its library number
        lib_num = gene_channel_ro['gene_library'][gene_channel_ro[gene_name].first_valid_index()]
        lib_num = lib_num.split('_')[1]
        new_columns[original_gene_name] = f"{probes_num}__{gene_name}__{channel}__{readout}__{lib_num}"
    elif not control_per_library:
        new_columns[original_gene_name] = f"{probes_num}__{gene_name}__{channel}__{readout}"
    else:
        new_columns[original_gene_name] = f"{probes_num}__{gene_name}__{channel}__{readout}__{lib_num}"


def mergeGenesWithChannels(channels_df: pd, norm_data_df: pd, control_genes_etc: list[str], gene_first_col: int,
                           control_per_library: bool, split_per_dilution: bool) -> pd:
    """
    Merging data frames to concatenate genes with their channels and readout
    :param split_per_dilution: split OD per culture dilution
    :param channels_df: genes to channels dataframe
    :param norm_data_df: cell by gene dataframe, normalized by probe number
    :param control_genes_etc: list of genes to drop
    :param gene_first_col: first column number where gene name appears
    :param control_per_library: for par^2-seq FISH, if the control genes should appear per library or for all libraries
    :return: combined dataframe with channel and readout for each gene
    """
    start_time = time.time()
    # Create a copy of the norm_data_df
    gene_channel_ro = norm_data_df.copy()
    createOpticalDensityColumn(gene_channel_ro, split_per_dilution)
    new_columns = {}
    last_gene_col = -1
    for original_gene_name in gene_channel_ro.columns[gene_first_col:last_gene_col]:
        addChannelAndReadoutToColumnName(original_gene_name, control_genes_etc, control_per_library, channels_df,
                                         gene_channel_ro, new_columns)
    gene_channel_ro.rename(columns=new_columns, inplace=True)
    end_time = time.time()
    print('Merging genes with channels and readouts runtime:', end_time - start_time, 'seconds')
    return gene_channel_ro


def initializeHelperDataStructures(gene_channel_od_df: pd.DataFrame, gene_first_col: int, sd_above_mean_thresh: int) ->\
        tuple[DataFrameGroupBy, float, list[str], pd.DataFrame]:
    """
    Initialize and group genes by OD
    :param gene_channel_od_df: data of the distributions
    :param gene_first_col: the genes appearing from this column until the end
    :param sd_above_mean_thresh: gene expressions in cells that higher than this threshold are dropped
    :return: grouped genes by OD
    """
    print('Creating main statistics summary...')
    cols = ['gene', 'library', 'OD', 'readout', 'number of probes', 'channel', 'n', 'mean', 'log₂(mean expression)', 'variance',
            'log₂(variance)',
            'standard deviation', 'CV', 'log₂(CV)', 'percent expressing', 'expressing cells number',
            'log₂(percent expressing)', 'mean expression 5% bottom', 'mean expression 5% top', 'top/bottom ratio',
            'log₂(top/bottom ratio)', 'entropy', 'log₂(entropy)', 'kurtosis', 'log₂(kurtosis)']
    #if split_by_dilution:
    #    cols.insert(2, 'culture dilution')
    # Exclude observations that are outlayer
    only_genes = gene_channel_od_df.iloc[:, gene_first_col:]
    only_genes = filterOutlayers(only_genes, sd_above_mean_thresh)
    start_time = time.time()
    summary_df = pd.DataFrame(columns=cols)
    # Group by OD, and calculate statistics
    grouped = only_genes.groupby(['OD'])
    return grouped, start_time, cols, summary_df


def allStatisticsCalculationForGene(group: pd.DataFrame, gene_col: str, od: str,
                                    control_genes_etc: list[str], control_per_library: bool, tail_percent: float,
                                    summary_df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """
    Calculation of all the statistics desired to be shown in summary file
    :param group: group sliced by condition
    :param gene_col: gene name with channel and readout
    :param od: gene name
    :param control_genes_etc: list of special genes
    :param control_per_library: for par^2-seq FISH, if the control genes should appear per library or for all libraries
    :param tail_percent: size of subpopulations to look at in top/bottom ratio metric
    :param summary_df: statistics summary file
    :param cols: columns to appear
    :return: main statistics summary file
    """
    gene_values = group[gene_col]
    # Drop NA
    gene_values = gene_values[~np.isnan(gene_values)]
    probes_num = int(gene_col.split('__')[0])
    gene = gene_col.split('__')[1]
    if len(gene_col.split('__')) < 3:
        print('Please make sure that the control is the first row')
    if len(gene_col.split('__')) == 2:
        print('gene', gene_col, 'have no channel and readout')
    channel = gene_col.split('__')[2]
    readout = gene_col.split('__')[3]
    lib = 'reference'
    if len(control_genes_etc) > 0 and gene not in control_genes_etc:
        lib = gene_col.split('__')[4]
        gene = gene + '__' + lib
    elif gene in control_genes_etc and not control_per_library:
        lib = 'reference'
    elif control_per_library:
        lib = gene_col.split('__')[4]
    sample_size = len(gene_values)
    mean = gene_values.mean()
    log_mean = np.log2(mean) if mean > 0 else 0
    var = gene_values.var()
    log_var = np.log2(var) if var > 0 else 0
    sd = np.sqrt(var)
    cv = np.divide(sd, mean) if mean > 0 else 0
    log_cv = np.log2(cv) if cv > 0 else 0
    percent_expressing = percentCellsExpressing(gene_values)
    num_expressing = np.count_nonzero(gene_values)
    log_percent_expressing = np.log2(percent_expressing) if percent_expressing > 0 else 0
    expression_bottom_5 = meanBottomOrTopPercentileExpression(gene_values, tail_percent, 'bottom') if mean > 0 \
        else 0
    expression_top_5 = meanBottomOrTopPercentileExpression(gene_values, tail_percent, 'top') if mean > 0 else 0
    top_bottom_ratio = expression_top_5 / expression_bottom_5 if expression_bottom_5 > 0 else 0
    log_top_bottom_ratio = np.log2(top_bottom_ratio) if top_bottom_ratio > 0 else 0
    entropy = calculateEntropy(gene_values)
    log_entropy = np.log2(entropy) if entropy > 0 else 0
    kurt = calculateKurtosis(gene_values)
    # Kurtosis can be negative, problematic with log
    log_kurt = np.log2(kurt) if kurt > 0 else 0
    row = [gene, lib, od, readout, probes_num, channel, sample_size, mean, log_mean, var, log_var, sd, cv, log_cv,
           percent_expressing, num_expressing, log_percent_expressing, expression_bottom_5, expression_top_5,
           top_bottom_ratio, log_top_bottom_ratio, entropy, log_entropy, kurt, log_kurt]
    summary_df = pd.concat([summary_df if not summary_df.empty else None, pd.DataFrame([row],
                                                                                       columns=cols)],
                           ignore_index=True)
    return summary_df


def filterRoundAndWriteToFile(summary_df, minimum_cells_expressing, start_time, control_genes_etc):
    """Drop problematic genes, round to 2 digits after the decimal point, and save to file"""
    summary_df = filterZeroExpressionGenes(summary_df, minimum_cells_expressing)
    print('Statistics calculated.')
    summary_df = summary_df.round(2)
    summary_df['gene_od'] = summary_df['gene'] + '_' + summary_df['OD'].astype(str)
    summary_df['gene_od'] = summary_df.apply(
        lambda row: f"{row['gene']}_{row['library']}_{row['OD']}" if row['gene'] in control_genes_etc else row[
            'gene_od'], axis=1)
    summary_df.set_index('gene_od', inplace=True)
    writeToFile(summary_df, 'statistics_summary.csv')
    end_time = time.time()
    print('Calculating statistics runtime:', end_time - start_time, 'seconds')
    return summary_df


def mainStatisticsSummaryFile(load_file: bool, gene_channel_od_df: pd = None, gene_first_col: int = None,
                              sd_above_mean_thresh: int = None, minimum_cells_expressing: int = None,
                              control_genes_etc: list[str] = None, control_per_library: bool = None,
                              tail_percent: float = 0.05,
                              constraints_per_channel: dict[str: tuple[float, float]] = None,
                              min_probe_number: int = None) -> pd.DataFrame:
    """
    Creating or loading for each gene and condition the statistics for the distribution
    :param min_probe_number:
    :param constraints_per_channel:
    :param load_file: to load the file if it already exists
    :param gene_channel_od_df: data of the distributions
    :param gene_first_col: first column where gene name appears
    :param sd_above_mean_thresh: gene expressions in cells that higher than this threshold are dropped
    :param minimum_cells_expressing: genes that expressed by less cells than this threshold are dropped
    :param control_genes_etc: list of control genes
    :param control_per_library: for par^2-seq FISH, if the control genes should appear per library or for all libraries
    :param tail_percent: size of the tail subpopulation to calculate the top to bottom ratio
    :return: main statistics dataframe
    """
    if Path('statistics_summary.csv').exists and load_file:
        print('Loading main statistics summary...')
        return readCsvFile('statistics_summary.csv')
    grouped, start_time, cols, summary_df = initializeHelperDataStructures(gene_channel_od_df, gene_first_col,
                                                                           sd_above_mean_thresh)
    # Loop over all OD's, and loop over all genes in each OD
    for name, group in grouped:
        # "name" is tuple with the OD inside. unpack it:
        (name,) = name
        for gene_col in group.columns[:-1]:  # Exclude the 'OD' column
            summary_df = allStatisticsCalculationForGene(group, gene_col, name, control_genes_etc, control_per_library,
                                                         tail_percent, summary_df, cols)
    summary_df = filterLowAndHighMean(summary_df, constraints_per_channel)
    summary_df = filterByProbesNumber(summary_df, min_probe_number)
    summary_df = filterRoundAndWriteToFile(summary_df, minimum_cells_expressing, start_time, control_genes_etc)
    return summary_df


def filterByProbesNumber(summary_df, min_probe_thresh):
    summary_df = summary_df[summary_df['number of probes'] >= min_probe_thresh]
    return summary_df


def calculateResidualsOfOneFeature(group_data: pd, predictor_column: str, response_column: str, robust_residuals: bool,
                                   alpha: float, polynomial_degree: dict[str: int]) -> pd:
    """
    For given statistic metric, calculate the residuals from the fit polynomial for all gene
    :param alpha:
    :param group_data: dataframe of genes
    :param predictor_column: x-axis variable
    :param response_column: y-axis variable
    :param robust_residuals: parameter that decides if to calculate the residual from the polynomial line or from the
     confidence interval
    :param polynomial_degree: dictionary which maps metric to polynomial degree
    :return: residuals from the polynomial
    """
    x = group_data[predictor_column]
    y = group_data[response_column]
    deg = polynomial_degree[response_column]
    print('\nPolynomial of degree', deg, 'coefficients with', response_column, 'as y axis:')
    pred, ci_lower, ci_upper = fitPolynomialWithCI(x, y, deg, alpha, visualization=False)
    if robust_residuals:
        residuals = residualsFromConfidenceInterval(y, ci_upper, ci_lower)
    else:
        residuals = y - pred
    return residuals


def copyColumns(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Copying general descriptive columns from summary dataframe to the residuals dataframe
    :param df: statistics summary dataframe
    :return: summary statistics and dataframe that is ready for residuals assignment
    """
    residuals_df = pd.DataFrame()
    #df['gene_od'] = df['gene'] + '_' + df['OD'].astype(str)
    #df['gene_od'] = df.apply(
    #    lambda row: f"{row['gene']}_{row['library']}_{row['OD']}" if row['gene'] in control_genes_etc else row[
    #        'gene_od'], axis=1)
    residuals_df['library'] = df['library']
    residuals_df['readout'] = df['readout']
    residuals_df['channel'] = df['channel']
    residuals_df['mean'] = df['mean']
    residuals_df['percent expressing'] = df['percent expressing']
    residuals_df['gene_od'] = df.index
    residuals_df.set_index('gene_od', inplace=True)
    #df.set_index('gene_od', inplace=True)
    return df, residuals_df


def fitPerChannel(grouped: DataFrameGroupBy, residuals_x_axis: str, statistic_column: str, robust_residuals: bool,
                  alpha: float, polynomial_degree: dict[str: int], temp_residuals: list[str],
                  residuals_df: pd, residuals_column: str) -> pd:
    """
    This function creates polynomial fit for genes that measured per channel
    :param alpha:
    :param grouped: genes that associated to a specific channel
    :param residuals_x_axis: the x-axis
    :param statistic_column: y_axis, the statistic metric
    :param robust_residuals: parameter that decides if to calculate the residuals from the confidence interval or not
    :param polynomial_degree: dictionary that maps metrics to degree of polynomial
    :param temp_residuals: temporary dataframe which hold the residuals
    :param residuals_df: residuals dataframe
    :param residuals_column: name of the column, statistic_column + "_res"
    :return: dataframe of residuals from fit polynomial per channel
    """
    for group_name, group_data in grouped:
        group_residuals = calculateResidualsOfOneFeature(group_data, residuals_x_axis, statistic_column,
                                                         robust_residuals,
                                                         alpha,
                                                         polynomial_degree)
        temp_residuals.append(group_residuals)
    residuals_df[residuals_column] = pd.concat(temp_residuals).sort_index()
    return residuals_df


def initializeAndGroupByChannel(statistic_column: str, df: pd.DataFrame) -> tuple[DataFrameGroupBy, list, str]:
    """Initialize helper variables and groups genes by channel"""
    residuals_column = f'{statistic_column} res'
    grouped = df.groupby('channel')
    return grouped, [], residuals_column


def calculateVariabilityScores(df: pd.DataFrame, residuals_x_axis: str, residuals_y_axis: list[str],
                               fit_per_channel: bool, robust_residuals: bool,
                               alpha: float, polynomial_degree: dict[str: int]) -> pd.DataFrame:
    """
    Main function for calculating the residuals
    :param alpha:
    :param df: summary statistics of genes
    :param residuals_x_axis: x-axis, probably the log_2(mean_expression)
    :param residuals_y_axis: the statistic metrics
    :param fit_per_channel: weather to create polynomial per channel or not
    :param robust_residuals: weather to calculate the residual from the polynomial itself or from the confidence
     interval
    :param polynomial_degree: dictionary of polynomial degrees for each metric
    :return: dataframe of residuals
    """
    df, residuals_df = copyColumns(df)
    for statistic_column in residuals_y_axis:
        grouped, temp_residuals, residuals_column = initializeAndGroupByChannel(statistic_column, df)
        if fit_per_channel:
            residuals_df = fitPerChannel(grouped, residuals_x_axis, statistic_column, robust_residuals,
                                         alpha, polynomial_degree, temp_residuals, residuals_df,
                                         residuals_column)
        # Else one fit for all and not per channel
        else:
            temp_residuals = calculateResidualsOfOneFeature(df, residuals_x_axis, statistic_column, robust_residuals,
                                                            alpha, polynomial_degree)
            residuals_df[residuals_column] = temp_residuals
    # Make the residuals file numbers rounded like the summary statistics file
    writeResToFile(residuals_df, 'residuals.csv')
    return residuals_df


def filterZeroExpressionGenes(data: pd.DataFrame, minimum_cells_expressing: int) -> pd.DataFrame:
    """Receives data and "minimum cells" threshold, and filters genes that expressed by cells lower than
    that threshold"""
    data = data[data['expressing cells number'] > minimum_cells_expressing]
    return data


def filterLowAndHighMean(df: pd.DataFrame, constraints: dict[str: tuple[int, int]]) -> pd.DataFrame:
    """Receives data and mean expression constraints, and filters genes out of those minimum and maximum constraints"""
    df = df[
        df.apply(lambda row: constraints[row['channel']][0] < row['mean'] < constraints[row['channel']][1],
                 axis=1)]
    return df


def filterOutlayers(df: pd.DataFrame, sd_above_mean_thresh: int) -> pd.DataFrame:
    """
    Filtering cells that express more than the threshold
    :param df: cell by gene dataframe
    :param sd_above_mean_thresh: threshold for filtering expression of cells
    :return: filtered dataframe
    """
    critical_values, start_time = calculaterCriticalValuesForFilteringCells(df, sd_above_mean_thresh)
    merged_df = pd.merge(df, critical_values, on='OD', suffixes=('', '_crit'))
    for gene in df.columns[:-1]:
        crit_col = f'{gene}_crit'
        orig_col = f'{gene}'
        merged_df.loc[merged_df[orig_col] > merged_df[crit_col], orig_col] = np.nan
    result_df = merged_df[df.columns]
    end_time = time.time()
    print('Filtering outliers runtime:', end_time - start_time, 'seconds')
    return result_df


def consensusScore(res_df_normalized: pd.DataFrame, residuals_y_axis: list[str]) -> pd.DataFrame:
    """
    Calculates the standard deviation of the residuals, penalizes the variability score, and produces consensus score
    :param res_df_normalized: dataframe of residuals from different metrics
    :param residuals_y_axis: all the metrics
    :return: modified dataframe, with consensus scores
    """
    num_of_metrics = len(residuals_y_axis)
    sd = np.float64(res_df_normalized.iloc[:, :num_of_metrics].std(axis=1))
    res_df_normalized['residuals sd'] = sd
    modified_sd = sd + 2
    log_modified_sd = np.log2(modified_sd)
    res_df_normalized['consensus penalty'] = log_modified_sd
    res_df_normalized['~CONSENSUS SCORE~'] = res_df_normalized['<VARIABILITY SCORE>'] / log_modified_sd
    return res_df_normalized


def positiveNegativeMetrics(res_df: pd.DataFrame) -> None:
    """Assigns negativity and positivity to residuals according to variability direction"""
    if 'log₂(entropy) res' in res_df:
        res_df['log₂(entropy) res'] = - res_df['log₂(entropy) res']
    if 'percent expressing res' in res_df:
        res_df['percent expressing res'] = - res_df['percent expressing res']


def insertRoundWriteToFile(res_df: pd.DataFrame, res_df_normalized: pd.DataFrame) -> None:
    """Copies columns from residuals dataframe to normalized dataframe, rounding the numbers and saves to file"""
    res_df_normalized.insert(0, 'percent expressing', res_df['percent expressing'])
    res_df_normalized.insert(0, 'mean', res_df['mean'])
    res_df_normalized.insert(0, 'channel', res_df['channel'])
    res_df_normalized.insert(0, 'readout', res_df['readout'])
    res_df_normalized.insert(0, 'library', res_df['library'])
    res_df_normalized = res_df_normalized.round(2)
    writeResToFile(res_df_normalized, 'standardized_residuals.csv')


def unifyMetricsNormalizeAndProduceSingleScore(res_df: pd.DataFrame, residuals_y_axis: list[str]) -> pd.DataFrame:
    """
    Given residuals for all metrics, assign correct negativity for them, produce mean residual and save to file
    :param res_df: non standardized residuals df
    :param residuals_y_axis: list of all metrics
    :return: standardized residuals dataframe
    """
    metric_first_col = 5
    positiveNegativeMetrics(res_df)
    # Z score standardization
    res_df_normalized = res_df.iloc[:, metric_first_col:]
    res_df_normalized = (res_df_normalized - res_df_normalized.mean()) / res_df_normalized.std()
    res_df_normalized['<VARIABILITY SCORE>'] = res_df_normalized.sum(axis=1) / len(residuals_y_axis)
    res_df_normalized = consensusScore(res_df_normalized, residuals_y_axis)
    insertRoundWriteToFile(res_df, res_df_normalized)
    return res_df_normalized
