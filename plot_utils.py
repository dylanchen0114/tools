# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib_venn import venn2
from plotly.graph_objs import *
from plotly.offline import init_notebook_mode, iplot

sns.set_style("whitegrid")
sns.set_style({'font.size': 5, 'axes.labelsize': 9, 'legend.fontsize': 8.5,
               'axes.titlesize': 12, 'xtick.labelsize': 7, 'ytick.labelsize': 7,
               'lines.linewidth': 1.5, 'grid.linewidth': 0.2})
seq_col_brew = sns.color_palette("Blues_r", 2)
sns.set_palette(seq_col_brew)


# Continuous Variable
# uni-variable
def plot_histograms(df, var, **kwargs):
    """
    uni-variable frequency histogram
    :param df: input DataFrame
    :param var: x-axis
    :param kwargs: row/col: to plot by row/col group variable; hue: target variable
    :return: plot
    example: plot_histograms(titanic, 'Age', row='Sex')
    """
    row = kwargs.get('row', None)
    col = kwargs.get('col', None)
    hue = kwargs.get('hue', None)
    title = kwargs.get('title', '')
    xlabel = kwargs.get('xlabel', '')
    ylabel = kwargs.get('ylabel', '')
    facet = sns.FacetGrid(df, hue=hue, aspect=3.5, size=2.5, row=row, col=col)
    facet.map(sns.distplot, var, kde=False)
    sns.plt.title(title)
    facet.set_xlabels(xlabel)
    facet.set_ylabels(ylabel)
    facet.add_legend(title=hue)


def plot_distribution(df, var, **kwargs):
    """
    kernel density estimate for plotting the shape of uni-variable
    :param df: input DataFrame
    :param var: x-axis
    :param kwargs: row/col: to plot by row/col group variable; hue: target variable
    :return: plot
    example: plot_distribution( titanic[titanic['Survived'] == 0] , var = 'Age', hue='Sex') 
    """
    row = kwargs.get('row', None)
    col = kwargs.get('col', None)
    hue = kwargs.get('hue', None)
    title = kwargs.get('title', '')
    xlabel = kwargs.get('xlabel', '')
    ylabel = kwargs.get('ylabel', '')
    facet = sns.FacetGrid(df, hue=hue, aspect=3.5, size=2.5, row=row, col=col)
    facet.map(sns.kdeplot, var, shade=False)
    sns.plt.title(title)
    facet.set_xlabels(xlabel)
    facet.set_ylabels(ylabel)
    facet.set(xlim=(0, df[var].max()))
    # facet.add_legend(title=hue)


# bi-variable
def plot_scatter(df, x, y, plot_type=plt.scatter, **kwargs):
    """
    bi-variable scatter plot
    :param df: input DataFrame
    :param x: x-axis
    :param y: y-axis
    :param plot_type: plt.scatter or sns.regplot
    :param kwargs:  row/col: to plot by row/col group variable; hue: target variable
    :return: plot
    example: plot_scatter(titanic, 'Age', 'Fare')
    """
    row = kwargs.get('row', None)
    col = kwargs.get('col', None)
    hue = kwargs.get('hue', None)
    title = kwargs.get('title', '')
    xlabel = kwargs.get('xlabel', '')
    ylabel = kwargs.get('ylabel', '')
    facet = sns.FacetGrid(df, hue=hue, aspect=3.5, size=2.5, row=row, col=col)
    facet.map(plot_type, x, y)
    sns.plt.title(title)
    facet.set_xlabels(xlabel)
    facet.set_ylabels(ylabel)
    facet.add_legend(title=hue)


def plot_line(df, x, y, **kwargs):
    """
    bi-variable line plot, usually for time series
    :param df: input DataFrame
    :param x: x-axis
    :param y: y-axis
    :param kwargs:  row/col: to plot by row/col group variable; hue: target variable
    :return: plot
    example: plot_line(titanic, 'Age', 'Fare')
    """
    row = kwargs.get('row', None)
    col = kwargs.get('col', None)
    hue = kwargs.get('hue', None)
    title = kwargs.get('title', '')
    xlabel = kwargs.get('xlabel', '')
    ylabel = kwargs.get('ylabel', '')
    facet = sns.FacetGrid(df, hue=hue, aspect=3.5, size=2.5, row=row, col=col)
    facet.map(plt.plot, x, y)
    sns.plt.title(title)
    facet.set_xlabels(xlabel)
    facet.set_ylabels(ylabel)
    facet.add_legend(title=hue)


# Categorical Variable

# uni-variable
def plot_count(df, **kwargs):
    """
    count plot for uni-variable
    :param df: input DataFrame
    :param kwargs: x: for 'v'; y for 'h'; hue for group
    :return: plot
    example: plot_count(titanic, x='Sex', hue='Pclass')
    """
    hue = kwargs.get('hue', None)
    x = kwargs.get('x', None)
    y = kwargs.get('y', None)
    title = kwargs.get('title', '')
    xlabel = kwargs.get('xlabel', '')
    ylabel = kwargs.get('ylabel', '')
    facet = sns.FacetGrid(df, aspect=3.5, size=2.5)
    facet.map(sns.countplot, data=df, x=x, y=y, hue=hue)
    sns.plt.title(title)
    facet.set_xlabels(xlabel)
    facet.set_ylabels(ylabel)
    facet.add_legend(title=hue)


def plot_percent(df, **kwargs):
    """
    Percentage Plot
    :param df: input dataFrame
    :param kwargs: x: x-axis; y: random choose one continuous variable; hue:
    :return: plt
    example: plot_percent(df=titanic, x='Pclass', y='Age', hue='Sex')
    """
    hue = kwargs.get('hue', None)
    x = kwargs.get('x', None)
    y = kwargs.get('y', None)
    title = kwargs.get('title', '')
    xlabel = kwargs.get('xlabel', '')
    ylabel = kwargs.get('ylabel', '')
    facet = sns.FacetGrid(df, aspect=3.5, size=2.5)
    facet.map(sns.barplot, data=df, x=x, y=y, hue=hue, estimator=lambda x: len(x) / len(df) * 100)
    sns.plt.title(title)
    facet.set_xlabels(xlabel)
    facet.set_ylabels(ylabel)
    facet.add_legend(title=hue)


def plot_stacked_bar(df, var, **kwargs):
    """
    stacked count bar chart
    :param df: input DataFrame
    :param var: variable Name string or list
    :param kwargs: 
    :return: plot
    example: plot_stacked_bar(titanic, ['Sex', 'Pclass'])
    """
    title = kwargs.get('title', '')
    xlabel = kwargs.get('xlabel', '')
    ylabel = kwargs.get('ylabel', '')
    
    for_plot = df.groupby(var).size().unstack()
    for_plot.plot(kind='bar', stacked=True)


# bi-variable
def plot_box(df, **kwargs):
    """
    box plot for both categorical and continuous variable
    :param df: input DataFrame
    :param kwargs: optional x or y
    :return: plot
    example: plot_box(titanic, x='Sex', y='Age', hue='Pclass')
    """
    hue = kwargs.get('hue', None)
    x = kwargs.get('x', None)
    y = kwargs.get('y', None)
    title = kwargs.get('title', '')
    xlabel = kwargs.get('xlabel', '')
    ylabel = kwargs.get('ylabel', '')
    
    facet = sns.FacetGrid(df, aspect=3.5, size=2.5)
    facet.map(sns.boxplot, data=df, x=x, y=y, hue=hue)
    sns.plt.title(title)
    facet.set_xlabels(xlabel)
    facet.set_ylabels(ylabel)
    facet.add_legend(title=hue)


def plot_cate_bar(df, estimator=np.mean, **kwargs):
    """
    bi-variable plot for mean, min, max, sum, count y-axis
    :param df: input DataFrame
    :param estimator: len for count number of row, sum or np.mean etc...
    :param kwargs: x for x-axis; y for y-axis; 
    :return: plot
    example: plot_cate_bar(df=titanic, x="Sex", y='Survived', hue="Pclass", estimator=len);
    """
    hue = kwargs.get('hue', None)
    x = kwargs.get('x', None)
    y = kwargs.get('y', None)
    ci = kwargs.get('ci', None)
    title = kwargs.get('title', '')
    xlabel = kwargs.get('xlabel', '')
    ylabel = kwargs.get('ylabel', '')
    facet = sns.FacetGrid(df, aspect=3.5, size=2.5)
    facet.map(sns.barplot, data=df, x=x, y=y, hue=hue, ci=ci, estimator=estimator)
    sns.plt.title(title)
    facet.set_xlabels(xlabel)
    facet.set_ylabels(ylabel)
    facet.add_legend(title=hue)


# all variable
def plot_correlation_map(df):
    corr = df.corr()
    _, ax = plt.subplots(figsize=(12, 10))
    cmap = sns.diverging_palette(220, 10, as_cmap=True)
    _ = sns.heatmap(
        corr,
        cmap=cmap,
        square=True,
        cbar_kws={'shrink': .9},
        ax=ax,
        annot=True,
        annot_kws={'fontsize': 12}
    )


def plot_pair(df, **kwargs):
    hue = kwargs.get('hue', None)
    sns.pairplot(df.dropna(), hue=hue)


def iplot_scatter(df, x, y, text, color='b', size=10):
    init_notebook_mode(connected=True)
    iplot({'data': [Scatter(x=df[x],y=df[y],text=df[text],marker=Marker(color=color, size=size, opacity=0.3), mode='markers')],
           'layout': Layout(xaxis=XAxis(title=x), yaxis=YAxis(title=y))})


def venn_two_column(list1, list2, fig_dir=None, col=None, name=('left', 'right'), save=False):

    plt.rcParams["figure.figsize"] = [5, 5]
    plt.rcParams['font.size'] = 20

    if isinstance(list1, list):
        list1 = set(list1)
    if isinstance(list2, list):
        list2 = set(list2)

    if isinstance(list1, pd.Series):
        list1 = set(list1.values.tolist())
    if isinstance(list2, pd.Series):
        list2 = set(list2.values.tolist())

    plt.figure()
    venn2([list1, list2], name)

    if save:
        fig_file = "%s/%s.pdf" % (fig_dir, col)
        plt.savefig(fig_file)


