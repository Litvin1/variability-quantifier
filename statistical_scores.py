#############################################################
# Author: Vadim Litvinov
# Date: 22 July 2024
#############################################################
import math
import numpy as np
from scipy import stats
import pandas as pd
import statsmodels.formula.api as smf
from statsmodels.regression.linear_model import RegressionResultsWrapper
from typing import Union, Tuple
import time
import scipy.optimize as opt
from delta_method import delta_method


def percentCellsExpressing(data: pd.Series) -> float:
    """Receives distribution, and calculates the percent of non-zero observations"""
    percentage_non_zero = np.count_nonzero(data) / len(data)
    return percentage_non_zero*100


def meanBottomOrTopPercentileExpression(data: pd.Series, percentile: float, tail: str) -> np.float64:
    """
    Calculates the top to bottom expression ratio
    :param data: cell by gene data
    :param percentile: the size of the subpopulations on the tail, in percentile
    :param tail: bottom tail or top tail
    :return: mean expression on the tail
    """
    non_zero_values = data[data != 0]
    # Calculate the number of elements corresponding to the bottom 5%
    tail_percent = math.ceil(percentile * len(non_zero_values))
    sorted_non_zero_values = np.sort(non_zero_values) if tail == 'bottom' else -np.sort(-non_zero_values)
    # Calculate the mean of the bottom 5% of non-zero values
    mean_tail_5_percent = np.mean(sorted_non_zero_values[:tail_percent])
    return mean_tail_5_percent


def logisticFunction(x, l, x0, k, c):
    return l / (1 + np.exp(-k*(x-x0))) + c


def initializeInputAndFitModel(x: pd.Series, y: pd.Series, deg: int) -> tuple[pd.DataFrame, RegressionResultsWrapper]:
    """
    Fits a polynomial with degree "deg" using ordinary least squares
    :param x: x-vector
    :param y: y-vector
    :param deg: degree of polynomial
    :return: the vectors in one dataframe, and the model
    """
    data = pd.DataFrame(columns=['y', 'x'])
    data['x'] = x
    data['y'] = y
    formula = 'y ~ ' + ' + '.join([f'np.power(x, {i})' for i in range(1, deg + 1)])
    model = smf.ols(formula=formula, data=data).fit()
    return data, model


def createPolynomialCurveForVisualizations(x: pd.Series, y, model: RegressionResultsWrapper, alpha: float)\
        -> tuple[pd.Series, pd.Series, np.ndarray, np.ndarray]:
    """
    Creates artificial x data and predicting its y values, for visualization of the curve
    :param alpha:
    :param x: x-vector
    :param model: polynomial function
    :return: the confidence interval of the polynomials, and the data for the visualization
    """
    x_fit = np.linspace(min(x), max(x), 100)
    data_new = pd.DataFrame({'x': x_fit})
    if type(model) != RegressionResultsWrapper:
        [popt, covb] = model
        y_fit = logisticFunction(data_new, *popt)
        d = delta_method(covb, popt, data_new['x'], logisticFunction, x, y, alpha)
        #mean_ci_lower, mean_ci_upper, x_fit, y_fit = percentExpressingProcedure(x, y, alpha, visualization=True)
        mean_ci_lower = d['lwr_conf']
        mean_ci_upper = d['upr_conf']
    else:
        predictions = model.get_prediction(data_new).summary_frame(alpha=alpha)
        mean_ci_lower = predictions['mean_ci_lower']
        mean_ci_upper = predictions['mean_ci_upper']
        y_fit = predictions['mean']
    return mean_ci_lower, mean_ci_upper, x_fit, y_fit


def createPolynomialCurveForResiduals(model: RegressionResultsWrapper, data: pd.DataFrame, alpha: float)\
        -> tuple[pd.Series, pd.Series, pd.Series]:
    """
    Given the polynomial function, predict y values for the given x
    :param alpha:
    :param model: polynomial function
    :param data:
    :return: y predictions given the x values of the genes, and the confidence intervals
    """
    predictions = model.get_prediction(data).summary_frame(alpha=alpha)
    predictions.index = data.index
    mean_ci_lower = predictions['mean_ci_lower'].round(2)
    mean_ci_upper = predictions['mean_ci_upper'].round(2)
    y_hat = predictions['mean'].round(2)
    return y_hat, mean_ci_lower, mean_ci_upper


def percentExpressingProcedure(x, y, alpha, visualization):
    data = pd.DataFrame(columns=['y', 'x'])
    data['x'] = x
    data['y'] = y
    model = opt.curve_fit(logisticFunction, data['x'], data['y'])
    data['x_for_vis'] = np.linspace(min(x), max(x), len(x))
    if visualization:
        [popt, covb] = model
        l, x0, k, c = popt
        y_hat = logisticFunction(data['x_for_vis'], l, x0, k, c)
        d = delta_method(covb, popt, data['x_for_vis'], logisticFunction, data['x'], data['y'], alpha)
        mean_ci_lower = d['lwr_conf']
        mean_ci_upper = d['upr_conf']
        return mean_ci_lower, mean_ci_upper, data['x_for_vis'], y_hat
    # Else we are calculating residuals and not visualizing
    else:
        [popt, covb] = model
        l, x0, k, c = popt
        y_hat = logisticFunction(data['x'], l, x0, k, c).round(2)
        d = delta_method(covb, popt, data['x'], logisticFunction, data['x'], data['y'], alpha)
        mean_ci_lower = d['lwr_conf'].round(2)
        mean_ci_upper = d['upr_conf'].round(2)
        return y_hat, mean_ci_lower, mean_ci_upper


def fitPolynomialWithCI(x: pd.Series, y: pd.Series, deg: int, alpha: float, visualization: bool = False) ->\
        Union[Tuple[pd.Series, pd.Series, np.ndarray, np.ndarray], Tuple[pd.Series, pd.Series, pd.Series]]:
    """
    Fits polynomial to data with confidence intervals
    :param alpha:
    :param x: x vector
    :param y: y vector
    :param deg: degree of polynomial
    :param visualization: to visualize the polynomial or not
    :return: confidence intervals and y-axis predictions
    """
    if deg == -1:
        return percentExpressingProcedure(x, y, alpha, visualization)
    data, model = initializeInputAndFitModel(x, y, deg)
    print(model.params)
    if visualization:
        return createPolynomialCurveForVisualizations(x, y, model, alpha)
    # Else we are calculating residuals and not visualizing
    else:
        return createPolynomialCurveForResiduals(model, data, alpha)


def calculateKurtosis(gene_values: pd.Series) -> Union[int, np.float64]:
    """calculates kurtosis for a distribution"""
    return 0 if pd.isna(stats.kurtosis(gene_values, fisher=True, bias=True)) else \
        stats.kurtosis(gene_values, fisher=True, bias=True)


def calculateEntropy(data: pd.Series) -> np.float64:
    """calculates Shannon entropy for a distribution"""
    num_bins = 100
    counts, bins = np.histogram(data, bins=num_bins)
    # Calculate probability function
    prob_function = counts / len(data)
    # Remove zero probabilities to avoid log(0) errors
    prob_function = prob_function[np.nonzero(prob_function)]
    # Calculate entropy
    entropy = -np.sum(prob_function * np.log2(prob_function))
    return entropy


def residualsFromConfidenceInterval(y: pd.Series, ci_upper: pd.Series, ci_lower: pd.Series) -> pd.Series:
    """Calculates the distance of y value for each metric from the closest confidence interval"""
    residuals = pd.Series(0.0, index=y.index, dtype='float64')
    # Calculate residuals where y is greater than ci_upper
    residuals[y > ci_upper] = y[y > ci_upper] - ci_upper[y > ci_upper]
    # Calculate residuals where y is less than ci_lower
    residuals[y < ci_lower] = y[y < ci_lower] - ci_lower[y < ci_lower]
    return residuals


def calculaterCriticalValuesForFilteringCells(df: pd.DataFrame, sd_above_mean_thresh: int)\
        -> tuple[pd.DataFrame, float]:
    """Given standard deviations value, calculate the actual value equal to it in terms of expression, and filter cells
     that express higher than this value"""
    start_time = time.time()
    grouped_means = df.groupby('OD').apply(lambda group: group.iloc[:, :].apply(lambda x: x[x > 0].mean()),
                                           include_groups=False)
    grouped_stds = df.groupby('OD').apply(lambda group: group.iloc[:, :].apply(lambda x: x[x > 0].std()),
                                          include_groups=False)
    critical_values = grouped_means + (sd_above_mean_thresh * grouped_stds)
    return critical_values, start_time
