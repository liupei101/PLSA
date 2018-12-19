#coding=utf-8
"""Module for visualizing a kind of curves in survival analyze

The function of this Module is served for visualizing a kind of curves in survival analyze.

"""
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter
from lifelines.plotting import add_at_risk_counts

def plot_riskGroups(data_groups, event_col, duration_col, labels=[], plot_join=False, 
                    xlabel="Survival time (Month)", ylabel="Survival Rate", legend="Risk Groups",
                    title="Survival function of Risk groups", save_fig_as=""):
    """Plot survival curve for different risk groups.

    Parameters
    ----------
    data_groups : list(`pandas.DataFame`) 
        list of DataFame[['E', 'T']], risk groups from lowest to highest.
    event_col : str
        column in DataFame indicating events.
    duration_col : atr
        column in DataFame indicating durations.
    labels : list(str), default []
        One text label for one group.
    plot_join : bool, default False
        Is plotting for two adjacent risk group, default False.
    save_fig_as : str
        Name of file for saving in local.

    Returns
    -------
    None
        Plot figure of each risk-groups.

    Examples
    --------
    >>> plot_riskGroups(df_list, "E", "T", labels=["Low", "Mid", "High"])
    """
    # init labels
    N_groups = len(data_groups)
    if len(labels) == 0:
        for i in range(N_groups):
            labels.append(str(i+1))
    # Plot
    fig, ax = plt.subplots(figsize=(8, 6))
    kmfit_groups = []
    for i in range(N_groups):
        kmfh = KaplanMeierFitter()
        sub_group = data_groups[i]
        kmfh.fit(sub_group[duration_col], event_observed=sub_group[event_col], label=labels[i])
        kmfh.survival_function_.plot(ax=ax)
        kmfit_groups.append(kmfh)
    # Plot two group (i, i + 1)
    if plot_join:
        for i in range(N_groups - 1):
            kmfh = KaplanMeierFitter()
            sub_group = pd.concat([data_groups[i], data_groups[i+1]], axis=0)
            kmfh.fit(sub_group[duration_col], event_observed=sub_group[event_col], label=labels[i]+'&'+labels[i+1])
            kmfh.survival_function_.plot(ax=ax)
            kmfit_groups.append(kmfh)
    plt.ylim(0, 1.01)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend(loc="best", title=legend)
    add_at_risk_counts(*kmfit_groups, ax=ax)
    plt.show()
    if save_fig_as != "":
        fig.savefig(save_fig_as, format='png', dpi=600)

def plot_timeAUC(x, y_train, y_test, labels=['Train', 'Validation'],
                 xlabel="Time", ylabel="AUC", 
                 title="Model Performance", save_fig_as=""):
    """Plot line chart about time and AUC.

    Parameters
    ----------
    x : list
        Time.
    y_train : list
        AUC of train.
    y_test : list
        AUC of test.
    **kws 
        Setting of plot.

    Returns
    -------
    None
        Plot figure of auc with time.

    Examples
    --------
    >>> plot_timeAUC([1, 3, 5, 10], train_list, test_list)
    """
    # Plot
    fig, ax = plt.subplots(figsize=(8, 6))
    lw = 2
    Cx = ['darkorange', 'cornflowerblue']
    plt.plot(x, y_train, color=Cx[0], lw=lw, label=labels[0], ls='-', marker='o', mec=Cx[0], mfc='w')
    plt.plot(x, y_test, color=Cx[1], lw=lw, label=labels[1], ls='--', marker='o', mec=Cx[1], mfc='w')
    plt.ylim(0, 1.01)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend(loc="best", title="Time-AUC")
    plt.show()
    if save_fig_as != "":
        fig.savefig(save_fig_as, format='png', dpi=600)

def plot_rsRisk(data, x_col, y1_col, y2_col, labels=["Line-1", "Line2"],
                xlabel="Risk Score", ylabel="Rate of Risk", 
                title="Curve of risk score and rate of risk", save_fig_as=""):
    """Plot continues function between risk score and rate of risk.

    Parameters
    ----------
    data : pandas.DataFame
        Full survival data.
    x_col : str
        Name of column indicating risk score.
    y1_col : str
        Name of column indicating rate of risk at t1.
    y2_col : str
        Name of column indicating rate of risk at t2.
    **kws
        Setting of plot.

    Returns
    -------
    None
        Plot figure of RS-rate.

    Examples
    --------
    >>> plot_rsRisk(data, 'RS', 'pred_idfs_y5', 'pred_idfs_y10', labels=['5 Year.', '10 Year.'])
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    Cx = ['darkorange', 'cornflowerblue']
    sns.regplot(x=x_col, y=y1_col, data=data, order=2, ax=ax, ci=95, scatter_kws={"alpha": 0}, line_kws={"label": labels[0], "lw": 2, "color": Cx[0]})
    sns.regplot(x=x_col, y=y2_col, data=data, order=2, ax=ax, ci=95, scatter_kws={"alpha": 0}, line_kws={"label": labels[1], "lw": 2, "color": Cx[1]})
    plt.xlim(0, 50)
    plt.ylim(0, 1.0)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend(loc="best", title="Time")
    plt.show()
    if save_fig_as != "":
        fig.savefig(save_fig_as, format='png', dpi=600)