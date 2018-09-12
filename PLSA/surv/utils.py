import numpy as np
import pandas as pd
import pyper as pr

def surv_roc(X, T, E, pt=None):
    """
    Get survival ROC at predicted time.

    Parameters:
        X, T, E: np.array with same shape.
        pt: Predicted time.

    Returns:
        Object include "FP", "TP" and "AUC" in ROC.

    Examples:
        surv_roc(X, T, E, pt=5)
    """
    if pt is None:
        pt = T.max()
    r = pr.R(use_pandas=True)
    r.assign("t", T)
    r.assign("e", E)
    r.assign("mkr", np.reshape(X, E.shape))
    r.assign("pt", pt)
    r.assign("mtd", "KM")
    r.assign("nobs", X.shape[0])
    # different predict.time may plot 1, 5, or 10 year ROC
    r("src<-survivalROC::survivalROC(Stime = t, status = e, marker = mkr, predict.time = pt, span = 0.25*nobs^(-0.20))")
    # r.src['AUC'] r.src['FP'], r.src['TP']
    return r.src

def surv_data_at_risk(data, duration_col, points=None):
    """
    Get number of people at risk at some timing.

    Parameters:
        data: DataFrame, survival data.
        duration_col: Name of column indicating time.
        points: Points of Time selected to watch.

    Returns:
        DataFrame, number of people at risk.

    Examples:
        surv_data_at_risk(data, "T", points=[0, 10, 20, 30, 40, 50])
    """
    Td = data[duration_col].value_counts()
    TList = list(Td.index)
    TList.sort()
    N, deaths, S = len(data), 0, 0
    # Initial
    res = [(0, N, 0)]
    for idx in TList:
        data_at_time = data[data['T'] == idx]
        deaths += sum(data_at_time['E'])
        res.append((int(idx), N, deaths))
        S += len(data_at_time)
        N -= len(data_at_time)
    res.append((TList[-1] + 1e5, 0, deaths))
    assert S == len(data)
    # Summary result
    if points is None:
        Tm = [0] + TList
    else:
        Tm = points
    T, Obs, Deaths = [], [], []
    j = 0
    for t in Tm:
        while res[j][0] < t:
            j += 1
        if res[j][0] == t:
            T.append(t)
            Obs.append(res[j][1])
            Deaths.append(res[j][2])
        else:
            T.append(t)
            Obs.append(res[j][1])
            Deaths.append(res[j-1][2])
    return pd.DataFrame({"Time": T, "Obs": Obs, "Deaths": Deaths})