#coding=utf-8
"""Module for visualizing ROC curve

The function of this Module is served for visualizing ROC curve.

"""
import numpy as np
import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter
from sklearn import metrics
from PLSA.surv import utils

def plot_twoROC(train_roc, test_roc, labels=['Train', 'Validation'],
                xlabel="1 - Specificity", ylabel="Sensitivity", 
                title="Model Performance", save_fig_as=""):
    """Plot two ROC curve in one figure.

    Parameters
    ----------
    train_roc : `dict`
        Python dict contains values about 'FP', 'TP', 'AUC'.
    test_roc : `dict`
        Python dict contains values about 'FP', 'TP', 'AUC'.
    save_fig_as: str
        Name of file for saving in local.

    Examples
    --------
    >>> plot_twoROC(train_roc, test_roc)
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    lw = 2
    Cx = ['darkorange', 'cornflowerblue']
    # ROC of training
    plt.plot(train_roc['FP'], train_roc['TP'], color = Cx[0],
             lw = lw, label = labels[0] + ' (AUC=%0.2f)' % (train_roc['AUC']))
    # ROC of Validation
    plt.plot(test_roc['FP'], test_roc['TP'], color = Cx[1],
             lw = lw, label = labels[1] + ' (AUC=%0.2f)' % (test_roc['AUC']))

    plt.plot([0, 1], [0, 1], color='navy', lw=lw-1, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.01])
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend(loc="best", title=title)
    plt.show()
    if save_fig_as != "":
        fig.savefig(save_fig_as, format='png', dpi=600)


def plot_ROC(data_roc, xlabel="1 - Specificity", ylabel="Sensitivity", 
             title="Model Performance", save_fig_as=""):
    """Plot one ROC curve in one figure.

    Parameters
    ----------
    data_roc : dict
        Python dict contains values about 'FP', 'TP', 'AUC'.
    save_fig_as: str
        Name of file for saving in local.

    Examples
    --------
    >>> plot_ROC(data_roc)
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    # plot ROC of data
    lw = 2
    plt.plot(data_roc['FP'], data_roc['TP'],
             lw=lw, label='AUC = %0.2f' % (data_roc['AUC']))
    plt.plot([0, 1], [0, 1], color='navy', lw=lw-1, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.01])
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend(loc="best", title=title)
    plt.show()
    if save_fig_as != "":
        fig.savefig(save_fig_as, format='png', dpi=600)

def plot_DROC(y_true, y_pred, x_true=None, x_pred=None, **kws):
    """Plot ROC curve for giving data.

    Parameters
    ----------
        y_true
            True label in train data.
        y_pred
            Predict label in train data.
        x_true
            True label in test data.
        x_pred
            Predict label in test data.
        **kws
            Arguments for plotting.

    Returns
    -------
    None
        Plot figure of AUC

    Examples
    --------
    >>> plot_DROC(train_y, train_pred, test_y, test_pred)
    """
    data_roc = dict()
    data_roc['FP'], data_roc['TP'], _ = metrics.roc_curve(y_true, y_pred, pos_label=1)
    data_roc['AUC'] = metrics.auc(data_roc['FP'], data_roc['TP'])
    print "__________________AUC____________________"
    print "AUC on train set :", data_roc['AUC']
    if not (x_true is None or x_pred is None):
        data_roc_ext = dict()
        data_roc_ext['FP'], data_roc_ext['TP'], _ = metrics.roc_curve(x_true, x_pred, pos_label=1)
        data_roc_ext['AUC'] = metrics.auc(data_roc_ext['FP'], data_roc_ext['TP'])
        print "AUC on test  set :", data_roc_ext['AUC']
        plot_twoROC(data_roc, data_roc_ext, **kws)
        return
    plot_ROC(data_roc, **kws)

def plot_SROC(data_train, data_test, pred_col, duration_col, event_col, 
              pt=None, labels=['Train', 'Validation'],
              **kws):
    """Plot Time-Dependent survival ROC curve for giving data.

    Parameters
    ----------
    data_train : pandas.DataFrame
        Train DataFrame included columns of Event, Duration, Pred.
    data_train : pandas.DataFrame
        Test DataFrame included columns of Event, Duration, Pred.
    pred_col : str
        Name of column indicating predicted value.
    duration_col : str
        Name of column indicating time.
    event_col : str
        Name of column indicating event.
    pt : int
        Predicte time.
    **kws 
        Arguments for plotting.

    Returns
    -------
    None
        Plot figure of AUC

    Examples
    --------
    >>> plot_SROC(data_train, data_test, "X", "T", "E", pt=5)
    """
    train_roc = utils.surv_roc(data_train, pred_col, duration_col, event_col, pt=pt)
    test_roc = utils.surv_roc(data_test, pred_col, duration_col, event_col, pt=pt)
    if "title" not in kws.keys():
        kws['title'] = "Survival ROC at Time %d" % int(pt)
    print "__________________AUC____________________"
    print "AUC on", labels[0], "set :", train_roc['AUC']
    print "AUC on", labels[1], "set :", test_roc['AUC']
    plot_twoROC(train_roc, test_roc, labels=labels, **kws)