from PLSA.surv.cutoff import youden_twocut
from PLSA.data.processing import cut_groups
from PLSA.vision.survrisk import plt_riskGroups
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
    plt_riskGroups(data_groups, event_col, duration_col, **kws)
    # logrank test
    summary12_ = logrank_test(Th, Tm, Eh, Em, alpha=0.95)
    summary11_ = logrank_test(Tl, Tm, El, Em, alpha=0.95)
    print "# High-Risk vs Middle-Risk :"
    print summary12_
    print "# Middle-Risk vs Low_Risk :"
    print summary11_