import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm

def plot_cphCoef(df, coef_col='coef', se_col='se(coef)', c_col='p', name_col=None, ci=0.95,
                 xlabel="Name of variable", ylabel="Coefficient", 
                 title="Variable's coefficient of CPH model", save_fig_as=""):
    """
    Short description about your function.

    Parameters:
        df: pandas.DataFrame, is cph.summary.
        coef_col: Coefficient.
        se_col: Standard error.
        c_col: Color.
        name_col: Name of x-axis's column. 
        ci: Confidence interval, default 0.95.

    Returns:

    Examples:
        plot_cphCoef(cph.summary, 'coef', 'se(coef)', 'p')
    """
    N = len(df)
    if name_col is None:
        name_col = 'name__'
        df[name_col] = df.index
    df['idx__'] = range(N)
    df['1 - P-value'] = 1 - df[c_col]
    # Calculate p for CI
    df['CI__'] = abs(norm.ppf((1-ci)/2)) * df[se_col]
    # Plot figure
    fig, ax = plt.subplots(figsize=(10, 6))
    df.plot.scatter(x='idx__', y=coef_col, c='1 - P-value', 
                    marker='s', s=120, cmap=plt.cm.get_cmap('YlOrRd'), ax=ax)
    ax.axhline(y=0, linestyle='--', color='black', linewidth=1)
    df.plot(x='idx__', y=coef_col, kind='bar', yerr='CI__', color="None", legend=False, ax=ax)
    ax.set_xticklabels(list(df[name_col]), rotation=0)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.show()
    if save_fig_as != "":
        fig.savefig(save_fig_as, format='png', dpi=600)
    # Drop columns
    df.drop(['name__', 'idx__', '1 - P-value', 'CI__'], axis=1, inplace=True, errors='ignore')