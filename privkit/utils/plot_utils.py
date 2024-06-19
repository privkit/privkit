"""
Plot utils

This module contains methods that are useful to plot data
"""
import numpy as np
import scipy.stats as spstats

from typing import Any
from matplotlib import pyplot as plt

# ========================= General plot methods =========================


def plot_information(title: str = '', x_label: str = '', y_label: str = '', show: bool = False):
    """
    This method adds title and axis labels to plots

    :param str title: title of the plot
    :param str x_label: label of the x axis
    :param str y_label: label of the y axis
    :param bool show: if True, show the plot.
    """
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)

    if show:
        plt.show()


def plot_xy_data(x: Any, y: Any, title: str = '', x_label: str = '', y_label: str = '', show: bool = False, **kwargs):
    """
    This method plots x values and y values.

    :param Any x: values to plot in the x axis
    :param Any y: values to plot in the y axis
    :param str title: title of the plot
    :param str x_label: label of the x axis
    :param str y_label: label of the y axis
    :param bool show: if True, show the plot.
    :param kwargs: the keyword arguments are used for specifying arguments according to the plot read methods.
    """
    plt.plot(x, y, 'r.', **(kwargs or {}))
    plot_information(title, x_label, y_label, show)


def pie_plot(values: Any, labels: Any, title: str = '', show: bool = False, **kwargs):
    """
    This method produces a pieplot from the given values and labels

    :param Any values: values to plot
    :param Any labels: labels of the values to plot
    :param str title: title of the plot
    :param bool show: if True, show the plot.
    :param kwargs: the keyword arguments are used for specifying arguments according to the plot read methods.
    """
    plt.pie(values, labels=labels, **(kwargs or {}))
    plot_information(title=title, show=show)


def boxplot(values: Any, labels: Any = None, title: str = '', show: bool = False, **kwargs):
    """
    This method produces a boxplot from the given values and labels

    :param Any values: values to plot
    :param Any labels: labels of the values to plot
    :param str title: title of the plot
    :param bool show: if True, show the plot.
    :param kwargs: the keyword arguments are used for specifying arguments according to the plot read methods.
    """
    plt.boxplot(values, labels=labels, **(kwargs or {}))
    plot_information(title=title, show=show)


def plot_errorbar(x: Any, y: Any, confidence_level: float = 0.95, title: str = '', x_label: str = '', y_label: str = '', show: bool = False, **kwargs):
    """
    This method plots an error bar according to x values and y values.

    :param Any x: values to plot in the x axis
    :param float confidence_level: level of confidence for the error bar
    :param Any y: values to plot in the y axis
    :param str title: title of the plot
    :param str x_label: label of the x axis
    :param str y_label: label of the y axis
    :param bool show: if True, show the plot.
    :param kwargs: the keyword arguments are used for specifying arguments according to the plot read methods.
    """
    ci_array = np.empty([2, 1])
    results_avg = np.average(y)
    ci_array[:, 0] = np.abs(
        results_avg - spstats.t.interval(confidence_level, len(y) - 1, loc=results_avg, scale=spstats.sem(y)))

    plt.errorbar(x, results_avg, yerr=ci_array, fmt='o', **(kwargs or {}))

    plot_information(title, x_label, y_label, show)
