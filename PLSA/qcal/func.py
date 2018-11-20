#coding=utf-8
"""Module for quick calling

The function of this Module is served for quick calling functions, and functions
of other modules will be called by it.

"""
import pandas as pd
from lifelines import CoxPHFitter
from lifelines.statistics import logrank_test
from lifelines.utils import concordance_index
from PLSA.surv.cutoff import youden_twocut
from PLSA.surv.utils import survival_status, surv_roc
from PLSA.data.processing import cut_groups
from PLSA.vision.survrisk import plot_riskGroups, plot_timeAUC
from PLSA.vision.calibration import plot_DCalibration

def div_three_groups(data, pred_col, duration_col, event_col, 
                     cutoffs=None, methods='youden', pt=None, **kws):
    """Divide data into three groups using methods and summarize result.

    Parameters
    ----------
    data : pandas.DataFame
        Full survival data.
    pred_col : str 
        Name of column to reference for dividing groups.
    duration_col : str
        Name of column indicating time.
    event_col : str
        Name of column indicating event.
    cutoffs : default `None` or `tuple`
        Given cutoffs for risk groups.
        If `cutoffs` is not None, then methods will not be called.
    methods : str
        Methods for selecting cutoffs, default "youden".
    pt : int
        Predicted time.

    Returns
    -------
    None
        Print summary of result and plot KM-curve of each groups.

    Examples
    --------
    >>> # Youden index to give cutoffs
    >>> div_three_groups(data, "X", "T", "E")
    >>> # Give cutoffs explicitly
    >>> div_three_groups(data, "X", "T", "E", cutoffs=(20, 50))
    """
    if not (cutoffs is None):
        ct1, ct2 = cutoffs
    elif methods == "youden":
        ct1, ct2 = youden_twocut(data, pred_col, duration_col, event_col, pt=pt)
    else:
        #TODO
        raise NotImplementedError('Methods not implemented')
    data_groups = cut_groups(data, pred_col, [data[pred_col].min(), ct1, ct2, data[pred_col].max()])
    
    Lgp = data_groups[0]
    Mgp = data_groups[1]
    Hgp = data_groups[2]

    Th = Hgp[duration_col].values
    Eh = Hgp[event_col].values
    Tm = Mgp[duration_col].values
    Em = Mgp[event_col].values
    Tl = Lgp[duration_col].values
    El = Lgp[event_col].values

    print "_________________Result of division__________________"
    print "Cut-off: Low vs Middle  =", ct1
    print "         Middle vs High =", ct2
    print 'Number of low risk group :', len(Lgp)
    print '          middle risk group :', len(Mgp)
    print '          high risk group :', len(Hgp)
    plot_riskGroups(data_groups, event_col, duration_col, **kws)
    # logrank test
    summary12_ = logrank_test(Th, Tm, Eh, Em, alpha=0.95)
    summary11_ = logrank_test(Tl, Tm, El, Em, alpha=0.95)
    print "# High-Risk vs Middle-Risk :"
    print summary12_
    print "# Middle-Risk vs Low_Risk :"
    print summary11_

def surv_calibration(data, duration_col, event_col, pred_proba, 
                     pt=None, n_bins=10,
                     xlabel="Predicted Risk Probability", 
                     ylabel="Observed Risk Probability", 
                     title="Model Performance", save_fig_as=""):
    """Evaluate calibration of predicted survival probability at time pt.

    Parameters
    ----------
    data: pandas.DataFame
        Full survival data.
    duration_col : str
        Name of column indicating time.
    event_col : str
        Name of column indicating event.
    pred_proba: np.array
        Predicted survival probability at time pt.
    pt : int
        Predicted time.

    Returns
    -------
    None
        Print summary of result and plot curve of calibration.

    Examples
    --------
    >>> surv_calibration(data, "T", "E", surv_function[10], pt=10)
    """
    if pt is None:
        pt = data[duration_col].max()
    T_col, E_col = survival_status(data, duration_col, event_col, pt)
    plot_DCalibration(E_col.values, 1-pred_proba, n_bins=n_bins, summary=True,
                      xlabel=xlabel, ylabel=ylabel, title=title, save_fig_as=save_fig_as)

def surv_time_auc(data_train, data_test, pred_col, duration_col, event_col, 
                  pt=[], labels=['Train', 'Validation'], **kws):
    """Plot curve of auc at some predicted time.

    Parameters
    ----------
    data_train : pandas.DataFame
        Full survival data for train.
    data_test : pandas.DataFame
        Full survival data for test.
    pred_col : str
        Name of column indicating target value.
    duration_col : str
        Name of column indicating time.
    event_col : str
        Name of column indicating event.
    pt : list(int)
        Predicted time indicating list of watching. 

    Returns
    -------
    None
        Print summary of result and plot curve of auc with time.

    Examples
    --------
    >>> surv_time_auc(train_data, test_data, 'X', 'T', 'E', pt=[1, 3, 5, 10])
    """
    train_list, test_list = [], []
    for t in pt:
        train_list.append(surv_roc(data_train, pred_col, duration_col, event_col, pt=t)['AUC'])
        test_list.append(surv_roc(data_test, pred_col, duration_col, event_col, pt=t)['AUC'])
    print "__________Summary of Surv-AUC__________"
    print "Time\tAUC", labels[0], "\tAUC", labels[1]
    for i in range(len(pt)):
        print "%.2f\t%.2f \t%.2f" % (float(pt[i]), train_list[i], test_list[i])
    plot_timeAUC(pt, train_list, test_list, labels=labels, **kws)

def surv_coxph(data_train, x_cols, duration_col, event_col, 
               data_test=None, pt=None, show_extra=True):
    """Integrate functions that include modeling using Cox Regression and evaluating 

    Parameters
    ----------
    data_train : pandas.DataFame
        Full survival data for train.
    x_cols : list of str
        Name of column indicating variables.
    duration_col : str
        Name of column indicating time.
    event_col : str
        Name of column indicating event.
    data_test : pandas.DataFame
        Full survival data for test, default None.
    pt : float
        Predicted time for AUC.

    Returns
    -------
    object
        Object of cox model in `lifelines.CoxPHFitter`.

    Examples
    --------
    >>> surv_coxph(train_data, ['x1', 'x2'], 'T', 'E', test_data, pt=5*12)
    """
    y_cols = [event_col, duration_col]
    cph = CoxPHFitter()
    cph.fit(data_train[x_cols + y_cols], 
            duration_col=duration_col, event_col=event_col, 
            show_progress=True)
    # CI of train
    pred_X_train = cph.predict_partial_hazard(data_train[x_cols])
    pred_X_train.rename(columns={0: 'X'}, inplace=True)
    ci_train = concordance_index(data_train[duration_col], -pred_X_train, data_train[event_col])
    # AUC of train at pt
    df = pd.concat([data_train[y_cols], pred_X_train], axis=1)
    roc_train = surv_roc(df, 'X', duration_col, event_col, pt=pt)
    if data_test is not None:
        # CI of test
        pred_X_test = cph.predict_partial_hazard(data_test[x_cols])
        pred_X_test.rename(columns={0: 'X'}, inplace=True)
        ci_test = concordance_index(data_test[duration_col], -pred_X_test, data_test[event_col])
        # AUC of test at pt
        df = pd.concat([data_test[y_cols], pred_X_test], axis=1)
        roc_test = surv_roc(df, 'X', duration_col, event_col, pt=pt)
    # Print Summary of CPH
    cph.print_summary()
    print "__________Metrics CI__________"
    print "CI of train: %.4f" % ci_train
    if data_test is not None:
        print "CI of test : %.4f" % ci_test
    print "__________Metrics AUC__________"
    print "AUC of train: %.4f" % roc_train['AUC']
    if data_test is not None:
        print "AUC of test : %.4f" % roc_test['AUC']
    
    if not show_extra:
        return cph
    # Print Coefficients
    print "__________Summary of Coefficients in CPH__________"
    cols = ['coef', 'p', 'lower 0.95', 'upper 0.95']
    print cols[0], ":"
    for i in cph.summary.index:
        print "%.4f" % (cph.summary.loc[i, cols[0]])
    print "__________"
    print cols[1], ":"
    for i in cph.summary.index:
        print "%.4f" % (cph.summary.loc[i, cols[1]])
    print "__________"
    print "95% CI :"
    for i in cph.summary.index:
        print "[%.4f, %.4f]" % (cph.summary.loc[i, cols[2]], cph.summary.loc[i, cols[3]])
    return cph