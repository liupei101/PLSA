import pandas as pd
import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter
from lifelines.plotting import add_at_risk_counts

def plot_riskGroups(data_groups, event_col, duration_col, labels=[], plot_join=False, 
                    xlabel="Survival time (Month)", ylabel="Survival Rate", title="Survival function of Risk groups", save_fig_as=""):
    """
    Plot survival curve for different risk groups.

    Parameters:
        data_groups: Python list of DataFame[['E', 'T']], risk groups from lowest to highest.
        event_col: column in DataFame indicating events.
        duration_col: column in DataFame indicating durations.
        labels: One text label for one group.
        plot_join: Is plotting for two adjacent risk group, default False.
        save_fig_as: Name of file for saving in local.

    Examples:
        plot_riskGroups(df_list, "E", "T", labels=["Low", "Mid", "High"])
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
        kmfh.fit(sub_group[duration_col], event_observed=sub_group[event_col], label=labels[i] + ' Risk Group')
        kmfh.survival_function_.plot(ax=ax)
        kmfit_groups.append(kmfh)
    # Plot two group (i, i + 1)
    if plot_join:
        for i in range(N_groups - 1):
            kmfh = KaplanMeierFitter()
            sub_group = pd.concat([data_groups[i], data_groups[i+1]], axis=0)
            kmfh.fit(sub_group[duration_col], event_observed=sub_group[event_col], label=labels[i]+'&'+labels[i+1] + ' Risk Group')
            kmfh.survival_function_.plot(ax=ax)
            kmfit_groups.append(kmfh)
    plt.ylim(0, 1.01)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend(loc="best", title="Risk Groups")
    add_at_risk_counts(*kmfit_groups, ax=ax)
    plt.show()
    if save_fig_as != "":
        fig.savefig(save_fig_as, format='png', dpi=600)

def plot_timeAUC(x, y_train, y_test,
                 xlabel="Time", ylabel="AUC", 
                 title="Model Performance", save_fig_as=""):
    """
    Plot line chart about time and AUC.

    Parameters:
        x: Time.
        y_train: AUC of train.
        y_test: AUC of test.
        **kws: Setting of figures.

    Returns:

    Examples:
        plot_timeAUC([1, 3, 5, 10], train_list, test_list)
    """
    # Plot
    fig, ax = plt.subplots(figsize=(8, 6))
    lw = 2
    Cx = ['darkorange', 'cornflowerblue']
    Lb = ['Training', 'Validation']
    plt.plot(x, y_train, color=Cx[0], lw=lw, label=Lb[0])
    plt.plot(x, y_test, color=Cx[1], lw=lw, label=Lb[1])
    plt.ylim(0, 1.01)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend(loc="best", title="Time-AUC")
    plt.show()
    if save_fig_as != "":
        fig.savefig(save_fig_as, format='png', dpi=600)