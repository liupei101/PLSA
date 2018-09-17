from PLSA.surv.cutoff import youden_twocut
from PLSA.surv.utils import survival_status
from PLSA.data.processing import cut_groups
from PLSA.vision.survrisk import plot_riskGroups
from PLSA.vision.calibration import plot_DCalibration
from lifelines.statistics import logrank_test

def div_three_groups(data, pred_col, duration_col, event_col, 
                     pt=None, methods='youden', **kws):
    """
    Divide data into three groups using methods and summarize result.

    Parameters:
        data: DataFame, full survival data.
        pred_col: Name of column to reference for dividing groups.
        duration_col: Name of column indicating time.
        event_col: Name of column indicating event.
        pt: Predicted time.
        methods: Methods for selecting cutoffs, default "youden".

    Returns:

    Examples:
        div_three_groups(data, "X", "T", "E")
    """
    if methods == "youden":
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
    """
    Evaluate calibration of predicted survival probability at time pt.

    Parameters:
        data: pd.DataFame, full survival data.
        duration_col: Name of column indicating time.
        event_col: Name of column indicating event.
        pred_proba: np.array, Predicted survival probability at time pt.
        pt: Predicted time.

    Returns:

    Examples:
        surv_calibration(data, "T", "E", surv_function[10], pt=10)
    """
    if pt is None:
        pt = data[duration_col].max()
    T_col, E_col = survival_status(data, duration_col, event_col, pt)
    plot_DCalibration(E_col.values, 1-pred_proba, n_bins=n_bins, summary=True,
                      xlabel=xlabel, ylabel=ylabel, title=title, save_fig_as=save_fig_as)