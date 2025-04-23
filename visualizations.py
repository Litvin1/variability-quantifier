#############################################################
# Author: Vadim Litvinov
# Date: 22 July 2024
#############################################################
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
#import umap
from sklearn.manifold import TSNE
from bokeh.layouts import column, gridplot
from bokeh.plotting import figure, show
from bokeh.models import HoverTool, UIElement
from statistical_scores import logisticFunction, fitPolynomialWithCI, initializeInputAndFitModel, createPolynomialCurveForVisualizations
from bokeh.models import ColumnDataSource, LinearColorMapper
import pandas as pd
from typing import Union, Hashable
from bokeh.io import export_png, export_svgs
from statsmodels.regression.linear_model import RegressionResultsWrapper
from pandas.core.groupby import DataFrameGroupBy
from bokeh.transform import linear_cmap, factor_cmap
from bokeh.palettes import Viridis256, Category10
import scipy.optimize as opt


def defineFeaturesForSpecialGenes(special_gene_colors, special_gene_shapes, special_gene_sizes, df, c, data_point_shape,
                                  data_point_size, avg_per_od):
    if special_gene_colors and special_gene_shapes:
        df['colors'] = df['gene'].apply(lambda gene: special_gene_colors.get(gene, c))
        df['shapes'] = df['gene'].apply(lambda gene: special_gene_shapes.get(gene, data_point_shape))
        df['sizes'] = df['gene'].apply(lambda gene: special_gene_sizes.get(gene, data_point_size))
    else:
        df['colors'] = c
        df['shapes'] = data_point_shape
        df['sizes'] = data_point_size
    if avg_per_od:
        return None
    else:
        source = ColumnDataSource(data=df)
        return source


def plotFitLineWithCI(channel_colors: dict[str:str], p: figure, df: pd.DataFrame, x_column: str, y: str,
                      alpha: float, channel: Union[str, Hashable, None],
                      polynomial_degree: dict[str:int], data_point_size, data_point_transparency,
                      data_point_color, fit_curve_color,
                      confidence_band_transparency, confidence_band_color,
                      point_perimeter_color, point_perimeter_transparency,
                      point_perimeter_width, special_gene_colors,
                      data_point_shape, special_gene_shapes,
                      special_gene_sizes
                      ) -> figure.scatter:
    """
    Visualize the fitted polynomial with confidence intervals
    :param alpha:
    :param data_point_transparency:
    :param data_point_size:
    :param channel_colors: colors dictionary for each channel
    :param p: bokeh figure
    :param df: the data
    :param x_column: the x-axis
    :param y: the y-axis
    :param channel: determines if the fit is per channel or not
    :param polynomial_degree: dictionary of polynomial degrees
    :return: scatter plot with fitted polynomials
    """
    print('\nPolynomial of degree', polynomial_degree[y], 'coefficients with', y, 'as y axis:')
    mean_ci_lower, mean_ci_upper, x_fit, y_fit = fitPolynomialWithCI(df[x_column], df[y],
                                                                     polynomial_degree[y],
                                                                     alpha, visualization=True)
    c = channel_colors[channel] if channel else data_point_color
    source = defineFeaturesForSpecialGenes(special_gene_colors, special_gene_shapes, special_gene_sizes, df, c, data_point_shape,
                                           data_point_size, avg_per_od=False)
    scatter = p.scatter(x_column, y, source=source, size='sizes', fill_alpha=data_point_transparency,
                        line_color=point_perimeter_color, line_alpha=point_perimeter_transparency,
                        line_width=point_perimeter_width, color='colors', marker='shapes')
    # Make axis font size bigger
    p.axis.axis_label_text_font_size = '20px'
    c = channel_colors[channel] if channel else fit_curve_color
    p.line(x_fit, y_fit, line_width=1.7, color=c)
    p.varea(x=x_fit, y1=mean_ci_lower, y2=mean_ci_upper, fill_alpha=confidence_band_transparency,
            color=confidence_band_color)
    return scatter


def addHoverToolAndAppend(p: figure, plots: list, renderers: list[figure.scatter]) -> None:
    """Receives figure and added hovering tool to it, to make it interactive"""
    hover = HoverTool(renderers=renderers)
    hover.tooltips = [('gene', f"@gene"),
                      ('OD', f"@OD"),
                      ('library', f"@library"),
                      ('channel', f"@channel"),
                      ('readout', f"@readout")]
    p.add_tools(hover)
    plots.append(p)


def fitPolynomialA488(channel_data: pd.DataFrame, x_column: str, y_column: str, polynomial_degree: dict[str: int],
                      all_channels_x: pd.Series, partially_transformed_channels_y: pd.Series)\
        -> tuple[pd.Series, pd.Series, RegressionResultsWrapper]:
    """
    This function receives channel A488 data and fits polynomial to it
    :param channel_data: data of the A488 channel genes
    :param x_column: usually log_2 of mean expression
    :param y_column: metric
    :param polynomial_degree: dictionary that maps metric to its polynomial degree
    :param all_channels_x: data structure to hold the log mean expression values for all genes
    :param partially_transformed_channels_y: data structure to hold the y values. A488 channel will not be transformed
    :return: x values, y values, and polynomial for channel A488 data
    """
    x_a488 = channel_data[x_column]
    y_a488 = channel_data[y_column]
    deg = polynomial_degree[y_column]
    # Regular fit line
    if deg == -1:
        polynomial_a488 = opt.curve_fit(logisticFunction, x_a488, y_a488)
    else:
        _, polynomial_a488 = initializeInputAndFitModel(x_a488, y_a488, deg)
        print(polynomial_a488.params, '\n')
    all_channels_x = x_a488 if all_channels_x.empty else pd.concat([all_channels_x, x_a488])
    partially_transformed_channels_y = y_a488 if partially_transformed_channels_y.empty \
        else pd.concat([partially_transformed_channels_y, y_a488])
    return all_channels_x, partially_transformed_channels_y, polynomial_a488


def fitPolynomialNonA488Channels(channel_data: pd.DataFrame, x_column: str, y_column: str,
                                 polynomial_degree: dict[str: int], alpha: float,
                                 polynomial_a488: RegressionResultsWrapper, all_channels_x: pd.Series,
                                 partially_transformed_channels_y: pd.Series) -> tuple[pd.Series, pd.Series]:
    """
    This function normalizes all channels to A488 channel
    :param alpha:
    :param channel_data: data of non A488 channels
    :param x_column: usually log_2 of mean expression
    :param y_column: metric
    :param polynomial_degree: dictionary that maps metric to its polynomial degree
    :param polynomial_a488: base polynomial we want to normalize other polynomials to
    :param all_channels_x: data structure to hold the log mean expression values for all genes
    :param partially_transformed_channels_y: data structure to hold the y values. A488 channel will not be transformed
    :return: x values and y values for plotting
    """
    # Create dataframe of all genes belongs to channels we want to transform
    x = channel_data[x_column]
    y = channel_data[y_column]
    deg = polynomial_degree[y_column]
    #x_polynomial_input = x
    x_polynomial_input = pd.DataFrame({'x': x})
    if deg == -1:
        cur_polynomial = opt.curve_fit(logisticFunction, x, y)
        [popt, covb] = cur_polynomial
        y_hat = logisticFunction(x_polynomial_input, *popt)['x']
        # from 1 column DF, to series
        #y_hat = y_hat['x']
        [popt_488, covb_488] = polynomial_a488
        y_hat_polynomial_488 = logisticFunction(x_polynomial_input, *popt_488)['x']
    else:
        _, cur_polynomial = initializeInputAndFitModel(x, y, deg)
        y_hat = cur_polynomial.get_prediction(x_polynomial_input).summary_frame(alpha=alpha)['mean']
        y_hat.index = y.index
        y_hat_polynomial_488 = polynomial_a488.get_prediction(x_polynomial_input
                                                              ).summary_frame(alpha=alpha)['mean']
    residuals = y - y_hat
    y_hat_polynomial_488.index = y.index
    y_transformed = y_hat_polynomial_488 + residuals
    all_channels_x = x if all_channels_x.empty else pd.concat([all_channels_x, x])
    partially_transformed_channels_y = y_transformed if partially_transformed_channels_y.empty \
        else pd.concat([partially_transformed_channels_y, y_transformed])
    return all_channels_x, partially_transformed_channels_y


def highlightAveragePointPerOD(result_df, p, x_column, y_column, data_point_transparency, point_perimeter_color,
                               point_perimeter_transparency, point_perimeter_width):
    od_categories = result_df["OD"].unique().tolist()
    palette = Category10[len(od_categories)]
    mean_df = result_df.groupby('OD').agg({x_column: 'mean', y_column: 'mean'})
    mean_df['colors'] = palette
    mean_df['sizes'] = 20
    mean_df['shapes'] = 'diamond'
    mean_df['is_mean'] = True
    mean_df['legend_labels'] = mean_df.index
    mean_df['transparency'] = 2.0
    result_df['transparency'] = data_point_transparency
    result_df['legend_labels'] = None
    result_df['is_mean'] = False
    combined_df = pd.concat([result_df, mean_df], ignore_index=False)
    source = ColumnDataSource(combined_df)
    scatter = p.scatter(x_column, y_column, source=source, size='sizes', fill_alpha='transparency',
                        color='colors', line_color=point_perimeter_color, line_alpha=point_perimeter_transparency,
                        line_width=point_perimeter_width, marker='shapes', legend_field='legend_labels')
    return scatter


def createScatterForChannelNormalizedData(grouped_channels: DataFrameGroupBy, all_channels_x: pd.Series, x_column: str,
                                          partially_transformed_channels_y: pd.Series, y_column: str,
                                          special_gene_colors: dict[str: str], special_gene_shapes: dict[str: str],
                                          special_gene_sizes: dict[str: float],
                                          data_point_color: str, data_point_shape: str, data_point_size: float,
                                          data_point_transparency: float, point_perimeter_color: str,
                                          point_perimeter_transparency: float, point_perimeter_width: float,
                                          polynomial_a488: RegressionResultsWrapper, alpha: float,
                                          p: figure, fit_curve_color: str, confidence_band_transparency: float,
                                          confidence_band_color: str, avg_genes_by_od: bool,
                                          fit_per_od: bool, polynomial_degrees: dict[str, int]) -> figure.scatter:
    """
    This function creates bokeh scatter based on input visualization parameters
    :param alpha:
    :param polynomial_degrees: dictionary of degrees for each metric
    :param fit_per_od: polynomial for each OD
    :param avg_genes_by_od: calculate mean point for each OD
    :param special_gene_sizes: dictionary which maps from gene name to size
    :param grouped_channels: data of genes grouped by channel
    :param all_channels_x: data structure to hold the log mean expression values for all genes
    :param x_column: usually log_2 of mean expression
    :param partially_transformed_channels_y: data structure to hold the y values. A488 channel will not be transformed
    :param y_column: metric
    :param special_gene_colors: dictionary which maps from gene name to color
    :param special_gene_shapes: dictionary which maps from gene name to shape
    :param data_point_color: default data point color
    :param data_point_shape: default data point shape
    :param data_point_size: point size
    :param data_point_transparency: point transparency
    :param point_perimeter_color: point edges color
    :param point_perimeter_transparency: point edges alpha
    :param point_perimeter_width: point edges thickness
    :param polynomial_a488: polynomial that is obtained by fitting to channel A488 genes
    :param p: bokeh figure
    :param fit_curve_color: color of the polynomial
    :param confidence_band_transparency: transparency of polynomial confidence interval
    :param confidence_band_color: color of polynomial confidence interval
    :return: scatter figure of the data, with polynomial line and confidence band
    """
    combined_df = pd.concat([group for _, group in grouped_channels])
    selected_columns = combined_df[['gene', 'OD', 'library', 'channel', 'readout']]
    all_channels_x.name = x_column
    partially_transformed_channels_y.name = y_column
    result_df = pd.concat([selected_columns, all_channels_x, partially_transformed_channels_y], axis=1)
    # drop the OD suffix from gene name
    result_df['gene'] = result_df['gene'].str.split('__').str[0]
    #source = defineFeaturesForSpecialGenes(special_gene_colors, special_gene_shapes, special_gene_sizes, result_df,
    #                                       data_point_color, data_point_shape, data_point_size)
    source = defineFeaturesForSpecialGenes(special_gene_colors, special_gene_shapes, special_gene_sizes, result_df,
                                           data_point_color, data_point_shape, data_point_size, avg_genes_by_od)
    if avg_genes_by_od:
        scatter = highlightAveragePointPerOD(result_df, p, x_column, y_column, data_point_transparency,
                                             point_perimeter_color, point_perimeter_transparency, point_perimeter_width)
    else:
        scatter = p.scatter(x_column, y_column, source=source, size='sizes', fill_alpha=data_point_transparency,
                            color='colors', line_color=point_perimeter_color, line_alpha=point_perimeter_transparency,
                            line_width=point_perimeter_width, marker='shapes')
    p.axis.axis_label_text_font_size = '20px'
    if fit_per_od:
        ods = result_df['OD'].unique()
        palette = list(Category10[len(ods)])
        od_to_color = dict(zip(ods, palette))
        grouped_ods = result_df.groupby('OD')
        for od, group_od in grouped_ods:
            specific_od_x = group_od[x_column]
            specific_od_y = group_od[y_column]
            _, polynomial_general_per_od = initializeInputAndFitModel(specific_od_x, specific_od_y, deg=polynomial_degrees[y_column])

            mean_ci_lower, mean_ci_upper, x_fit_specific_od, y_fit_specific_od = createPolynomialCurveForVisualizations(
                                                                                            specific_od_x,
                                                                                            polynomial_general_per_od,
                                                                                            alpha)
            p.line(x_fit_specific_od, y_fit_specific_od, line_width=1.7, color=od_to_color[od], legend_label=od)
            p.varea(x=x_fit_specific_od, y1=mean_ci_lower, y2=mean_ci_upper, fill_alpha=confidence_band_transparency,
                    color=od_to_color[od])
            print('debug')
    else:
        # Now, we have the 1 polynomial and all the partially transformed data. Visualize it
        # The polynomial for this new data is the same as polynomial_a488
        mean_ci_lower, mean_ci_upper, x_fit, y_fit = createPolynomialCurveForVisualizations(all_channels_x,
                                                                                            partially_transformed_channels_y,
                                                                                            polynomial_a488,
                                                                                            alpha)
        p.line(x_fit, y_fit, line_width=1.7, color=fit_curve_color)
        p.varea(x=x_fit, y1=mean_ci_lower, y2=mean_ci_upper, fill_alpha=confidence_band_transparency,
                color=confidence_band_color)
    return scatter


def plotOneChannelFitTransformOthers(grouped_channels: DataFrameGroupBy, p: figure, x_column: str, y_column: str,
                                     alpha: float, polynomial_degree: dict[str: int], data_point_size: float,
                                     data_point_transparency: float, data_point_color: str, fit_curve_color: str,
                                     confidence_band_transparency: float, confidence_band_color: str,
                                     point_perimeter_color: str, point_perimeter_transparency: float,
                                     point_perimeter_width: float, special_gene_colors: dict[str: str],
                                     data_point_shape: str, special_gene_shapes: dict[str: str],
                                     special_gene_sizes: dict[str: float], avg_genes_by_od: bool,
                                     fit_per_od: bool) -> figure.scatter:
    """
    This function loops over channels, transforms y values for channel normalization, and returns scatter with fit
    :param alpha:
    :param fit_per_od: polynomial for each OD
    :param avg_genes_by_od: calculate and show mean point for each OD
    :param special_gene_sizes: dictionary which maps from gene name to size
    :param grouped_channels: data of genes grouped by channel
    :param p: bokeh figure
    :param x_column: usually log_2 of mean expression
    :param y_column: metric
    :param polynomial_degree: dictionary that maps metric to its polynomial degree
    :param data_point_size: point size
    :param data_point_transparency: point transparency
    :param data_point_color: default data point color
    :param fit_curve_color: color of the polynomial
    :param confidence_band_transparency: transparency of polynomial confidence interval
    :param confidence_band_color: color of polynomial confidence interval
    :param point_perimeter_color: point edges color
    :param point_perimeter_transparency: point edges alpha
    :param point_perimeter_width: point edges thickness
    :param special_gene_colors: dictionary which maps from gene name to color
    :param data_point_shape: default data point shape
    :param special_gene_shapes: dictionary which maps from gene name to shape
    :return: scatter figure of the data, with polynomial line and confidence band
    """
    print('\npolynomial coefficients with', y_column, 'as y axis:')
    all_channels_x = pd.Series()
    partially_transformed_channels_y = pd.Series()
    polynomial_a488 = None
    for channel, channel_data in grouped_channels:
        if channel == 'A488':
            all_channels_x, partially_transformed_channels_y, polynomial_a488\
                = fitPolynomialA488(channel_data, x_column, y_column, polynomial_degree, all_channels_x,
                                    partially_transformed_channels_y)
        else:
            all_channels_x, partially_transformed_channels_y\
                = fitPolynomialNonA488Channels(channel_data, x_column, y_column, polynomial_degree, alpha,
                                               polynomial_a488, all_channels_x, partially_transformed_channels_y)
    return createScatterForChannelNormalizedData(grouped_channels, all_channels_x, x_column,
                                                 partially_transformed_channels_y,
                                                 y_column, special_gene_colors, special_gene_shapes, special_gene_sizes,
                                                 data_point_color,
                                                 data_point_shape, data_point_size, data_point_transparency,
                                                 point_perimeter_color, point_perimeter_transparency,
                                                 point_perimeter_width,
                                                 polynomial_a488, alpha, p, fit_curve_color,
                                                 confidence_band_transparency, confidence_band_color, avg_genes_by_od,
                                                 fit_per_od, polynomial_degree)


def addPointsAndFitToFigure(fit_per_channel: bool, df: pd.DataFrame, renderers, channel_colors: dict[str: str],
                            p: figure, x_column: str, y: str, alpha: float,
                            polynomial_degree: dict[str: int], normalize_channel_fits: bool, data_point_size,
                            data_point_transparency, data_point_color, fit_curve_color, confidence_band_transparency,
                            confidence_band_color, point_perimeter_color, point_perimeter_transparency,
                            point_perimeter_width, special_gene_colors, data_point_shape, special_gene_shapes,
                            special_gene_sizes, fit_per_dilution, avg_genes_by_od, fit_per_od) -> None:
    """Given x and y datapoints, this function adds them, and their polynomial fit, to the figure"""
    if fit_per_channel:
        grouped_channels = df.groupby('channel')
        if normalize_channel_fits:
            renderers.append(plotOneChannelFitTransformOthers(grouped_channels, p, x_column,
                                                              y, alpha, polynomial_degree,
                                                              data_point_size, data_point_transparency,
                                                              data_point_color, fit_curve_color,
                                                              confidence_band_transparency, confidence_band_color,
                                                              point_perimeter_color, point_perimeter_transparency,
                                                              point_perimeter_width, special_gene_colors,
                                                              data_point_shape, special_gene_shapes,
                                                              special_gene_sizes, avg_genes_by_od, fit_per_od))
            return None
        # Else don't normalize channels, and produce 3 curves
        for channel, channel_data in grouped_channels:
            renderers.append(plotFitLineWithCI(channel_colors, p, channel_data, x_column, y, alpha,
                                               channel, polynomial_degree, data_point_size, data_point_transparency,
                                               data_point_color, fit_curve_color,
                                               confidence_band_transparency, channel_colors[channel],
                                               point_perimeter_color, point_perimeter_transparency,
                                               point_perimeter_width, special_gene_colors,
                                               data_point_shape, special_gene_shapes,
                                               special_gene_sizes))
    # Else one polynomial for all channels, without normalize by channel
    else:
        # Fit a curve per dilution
        if fit_per_dilution:
            df['dilution'] = df['OD'].apply(lambda x: x.split('_')[0])
            df = df[df['dilution'] != 'no']
            grouped_dilutions = df.groupby('dilution')
            for dilution, dilution_data in grouped_dilutions:
                renderers.append(plotFitLineWithCI(channel_colors, p, dilution_data, x_column, y, alpha,
                                                   dilution, polynomial_degree, data_point_size,
                                                   data_point_transparency, data_point_color, fit_curve_color,
                                                   confidence_band_transparency, channel_colors[dilution],
                                                   point_perimeter_color, point_perimeter_transparency,
                                                   point_perimeter_width, special_gene_colors,
                                                   data_point_shape, special_gene_shapes,
                                                   special_gene_sizes))
            print('\n')
        # else its just 1 curve for all
        else:
            renderers.append(plotFitLineWithCI(channel_colors, p, df, x_column, y, alpha, None,
                                               polynomial_degree, data_point_size, data_point_transparency,
                                               data_point_color, fit_curve_color,
                                               confidence_band_transparency, confidence_band_color,
                                               point_perimeter_color, point_perimeter_transparency,
                                               point_perimeter_width, special_gene_colors,
                                               data_point_shape, special_gene_shapes,
                                               special_gene_sizes))


def interactivePlotsBokeh(df: pd.DataFrame, x_column: str, y_columns: list[str], alpha: float,
                          fit_per_channel: bool, polynomial_degree: dict[str:int], normalize_channel_fits: bool,
                          data_point_color: str = 'blue', data_point_size: str = '4.5',
                          data_point_transparency: str = 1, fit_curve_color: str = 'red',
                          confidence_band_transparency: str = 1, confidence_band_color: str = 'red',
                          point_perimeter_color: str = 'red', point_perimeter_transparency: float = 1,
                          point_perimeter_width: float = 1, x_limits: dict[str: tuple[float: float]] = None,
                          y_limits: dict[str: tuple[float: float]] = None, special_gene_colors: dict[str: str] = None,
                          data_point_shape: str = 'circle', special_gene_shapes: dict[str: str] = None,
                          special_gene_sizes: dict[str: float] = None, fit_per_dilution: bool = None,
                          avg_genes_by_od: bool = False, fit_per_od: bool = False) -> list[figure]:
    """
    Main visualization function
    :param fit_per_od: create polynomial for each OD
    :param avg_genes_by_od: make a highlighted points that will represent the optical densities
    :param fit_per_dilution: for Danielle.L
    :param special_gene_sizes: dictionary which maps from gene name to size
    :param special_gene_shapes: dictionary which maps from gene name to shape
    :param data_point_shape: default data point shape
    :param special_gene_colors: dictionary which maps from gene name to color
    :param y_limits: list of limits on x-axis for to each metric
    :param x_limits: list of limits on y-axis for to each metric
    :param point_perimeter_width: point edges thickness
    :param point_perimeter_transparency: point edges alpha
    :param point_perimeter_color: point edges color
    :param confidence_band_color: color of polynomial confidence interval
    :param confidence_band_transparency: transparency of polynomial confidence interval
    :param fit_curve_color: color of the polynomial
    :param data_point_transparency: point alpha
    :param data_point_size: point size
    :param data_point_color: point color
    :param df: data to visualize
    :param x_column: the x-axis
    :param y_columns: the metrics for y-axis
    :param fit_per_channel: polynomial per channel or one for all
    :param polynomial_degree: dictionary that maps from metric to polynomial degree
    :param normalize_channel_fits: transform the y values while keeping the residuals
    :return: list of bokeh plots.
    """
    channel_colors = {'A647': 'red',
                      'A488': 'blue',
                      'A550': 'green',
                      'D100': 'blue',
                      'D10000': 'yellow',
                      'no': 'black'}
    plots = []
    # Loop on metrics
    for y in y_columns:
        renderers = []
        if x_limits and y_limits:
            p = figure(output_backend='svg', width=700, height=350, tools="pan,wheel_zoom,box_zoom,reset,save",
                       x_axis_label=x_column, y_axis_label=y, x_range=x_limits[y], y_range=y_limits[y])
        else:
            p = figure(width=700, height=350, tools="pan,wheel_zoom,box_zoom,reset,save",
                       x_axis_label=x_column, y_axis_label=y)
        addPointsAndFitToFigure(fit_per_channel, df, renderers, channel_colors, p, x_column, y, alpha,
                                polynomial_degree, normalize_channel_fits,  data_point_size, data_point_transparency,
                                data_point_color, fit_curve_color, confidence_band_transparency, confidence_band_color,
                                point_perimeter_color,
                                point_perimeter_transparency, point_perimeter_width, special_gene_colors,
                                data_point_shape, special_gene_shapes, special_gene_sizes, fit_per_dilution,
                                avg_genes_by_od, fit_per_od)
        addHoverToolAndAppend(p, plots, renderers)
    #output_notebook()  # Use this if you are in a Jupyter notebook
    show(column(*plots))
    return plots


def exportBokehToFile(plot: UIElement, file_name: str, file_type: str) -> None:
    """Save bokeh as svg file"""
    file_and_extension = file_name + '.' + file_type
    if file_type == 'png':
        export_png(plot, filename=file_and_extension)
    elif file_type == 'svg':
        export_svgs(plot, filename=file_and_extension)


def residualsToResiduals(variability_score: pd.DataFrame, metrics: list, point_size: str, point_color: str,
                         point_transparency: float) -> None:
    """Simple plot of pairplots for the residuals"""
    metrics_res = [metric + ' res' for metric in metrics]
    plots = []
    for i, metric1 in enumerate(metrics_res):
        for j, metric2 in enumerate(metrics_res):
            if i < j:
                p = figure(title=f"{metric1} vs {metric2}",
                           x_axis_label=metric1,
                           y_axis_label=metric2,
                           width=400,
                           height=400)
                source = ColumnDataSource(variability_score)
                p.scatter(x=metric1, y=metric2, source=source, size=point_size, color=point_color,
                          alpha=point_transparency)
                p.axis.axis_label_text_font_size = '20px'
                plots.append(p)
    n = len(metrics)
    grid = gridplot([plots[i:i + n - 1] for i in range(0, len(plots), n - 1)])
    # Show the plot
    show(grid)


def plotPca(df: pd.DataFrame, n_components: int = 2) -> None:
    """calculate and plot principal component analysis"""
    pca = PCA(n_components=n_components)
    components = pca.fit_transform(df)
    plt.figure(figsize=(8, 6))
    plt.scatter(components[:, 0], components[:, 1], cmap='viridis')
    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    plt.title('PCA Plot')
    plt.show()


'''
def plotUmap(df, n_components=2):
    reducer = umap.UMAP(n_components=n_components)
    embedding = reducer.fit_transform(df)
    plt.figure(figsize=(8, 6))
    plt.scatter(embedding[:, 0], embedding[:, 1], cmap='viridis')
    plt.xlabel('UMAP Component 1')
    plt.ylabel('UMAP Component 2')
    plt.title('UMAP Plot')
    plt.show()
'''


def plotTsne(df: pd.DataFrame, n_components: int = 2) -> None:
    """calculate and plot t-distributed Stochastic Neighbor Embedding"""
    tsne = TSNE(n_components=n_components)
    tsne_results = tsne.fit_transform(df)
    plt.figure(figsize=(8, 6))
    plt.scatter(tsne_results[:, 0], tsne_results[:, 1], cmap='viridis')
    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')
    plt.title('t-SNE Plot')
    plt.show()
