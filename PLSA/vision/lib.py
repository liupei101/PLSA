#coding=utf-8
"""Module for visualizing common curve

The function of this Module is served for visualizing common curve.

"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

def plot_cphCoef(dfx, coef_col='coef', se_col='se(coef)', c_col='p', name_col=None, ci=0.95,
                 error_bar='hr', xlabel="Name of variable", ylabel="", 
                 title="Variable's coefficient of CPH model", 
                 figsize=(8, 6), save_fig_as=""):
    """Visualize variables' coefficient in lifelines.CPH model

    Parameters
    ----------
    dfx : pandas.DataFrame
        Object equals to cph.summary.
    coef_col : str
        Name of column indicating coefficient.
    se_col : str
        Name of column indicating standard error.
    c_col: str
        Name of column indicating color.
    name_col: str
        Name of x-axis's column. 
    ci : float
        Confidence interval, default 0.95.
    error_bar : str
        Type of error bars, 'hr' for asymmetrical error bars,
        'log-hr' for symmetrical error bars.

    Returns
    -------
    None
        Plot figure of coefficient.

    Examples
    --------
    >>> plot_cphCoef(cph.summary, 'coef', 'se(coef)', 'p')
    """
    df = dfx.copy(deep=True)
    N = len(df)
    if name_col is None:
        name_col = 'name__'
        df[name_col] = df.index
    df['idx'] = range(N)
    df['1 - P-value'] = 1 - df[c_col]
    # Calculate CI
    df['CI'] = abs(norm.ppf((1-ci)/2)) * df[se_col]
    # Plot figure
    fig, ax = plt.subplots(figsize=figsize)
    if error_bar == 'log-hr':
        df.plot.scatter(x='idx', y=coef_col, c='1 - P-value', 
                    marker='s', s=120, cmap=plt.cm.get_cmap('YlOrRd'), ax=ax)
        ylabel = ('Coefficient' if ylabel == '' else ylabel)
        ax.axhline(y=0, linestyle='--', color='black', linewidth=1)
        ax.errorbar(df['idx'], df[coef_col], yerr=df['CI'],
                    ecolor='black', elinewidth=0.8, linestyle='')
    elif error_bar == 'hr':
        # calculate
        df['er_lower'] = np.exp(df[coef_col]) - np.exp(df[coef_col] - df['CI'])
        df['er_upper'] = np.exp(df[coef_col] + df['CI']) - np.exp(df[coef_col])
        df[coef_col] = np.exp(df[coef_col])
        df.plot.scatter(x='idx', y=coef_col, c='1 - P-value',
                    marker='s', s=120, cmap=plt.cm.get_cmap('YlOrRd'), ax=ax)
        ylabel = ('Hazard Ratio' if ylabel == '' else ylabel)
        ax.axhline(y=1, linestyle='--', color='black', linewidth=1)
        ax.errorbar(df['idx'], df[coef_col], yerr=[df['er_lower'].values, df['er_upper'].values],
                    ecolor='black', elinewidth=0.8, linestyle='')
    else:
        # TODO
        pass
    ax.set_xticks(list(df['idx']))
    ax.set_xticklabels(list(df[name_col]), rotation=0)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.show()
    if save_fig_as != "":
        fig.savefig(save_fig_as, format='png', dpi=600, bbox_inches='tight')
    # Drop DataFrame
    del df