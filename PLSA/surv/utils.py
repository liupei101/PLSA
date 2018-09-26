import numpy as np
import pandas as pd
import pyper as pr

def surv_roc(data, pred_col, duration_col, event_col, pt=None):
    """Get survival ROC at predicted time.

    Parameters
    ----------
    data : pandas.DataFrame
        Full survival data.
    pred_col : str
        Name of column to reference for dividing groups.
    duration_col : str
        Name of column indicating time.
    event_col : str
        Name of column indicating event.
    pt : int
        Predicted time.

    Returns
    -------
    `dict`
        Object of dict include "FP", "TP" and "AUC" in ROC.

    Examples
    --------
    >>> surv_roc(data, 'X', 'T', 'E', pt=5)
    """
    X = data[pred_col].values
    T = data[duration_col].values
    E = data[event_col].values
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
    """Get number of people at risk at some timing.

    Parameters
    ----------
    data : pandas.DataFrame
        Full survival data.
    duration_col : str
        Name of column indicating time.
    points : list(int)
        Points of Time selected to watch.

    Returns
    -------
    `pandas.DataFrame` 
        Number of people at risk.

    Examples
    --------
    >>> surv_data_at_risk(data, "T", points=[0, 10, 20, 30, 40, 50])
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

def survival_by_hr(T0, S0, pred):
    """Get survival function of patients according to giving hazard ratio.

    Parameters
    ----------
    T0 : np.array
        time.
    S0 : np.array
        based estimated survival function of patients.
    pred : pandas.Series
        hazard ratio of patients. 

    Returns
    -------
    `tuple`
        T0, ST indicating survival function of patients.

    Examples
    --------
    >>> survival_by_hr(T0, S0, data['hazard_ratio'])
    """
    hazard_ratio = pred.values.reshape((pred.shape[0], 1))
    # Estimate S0(t) using data(base_X, base_label)
    ST = S0**(hazard_ratio)

    return T0, ST

def survival_status(data, duration_col, event_col, end_time, inplace=False):
    """Get status of event at a specified time. 

    0: status = 0, Time = end_time (T >= end_time) 
       status = 0, Time = T  (T < end_time)
    1: status = 1, Time = T  (T <= end_time)
       status = 0, Time = end_time (T > end_time)

    Parameters
    ----------
    data : pandas.DataFrame
        Full survival data.
    duration_col : str
        Name of column indicating time.
    event_col : str
        Name of column indicating event.
    end_time : int
        End time of study.
    inplace : bool, default False
        Do replace original data.

    Returns
    -------
    None or tuple
        data indicates status of survival.
        
        None or tuple(time(pandas.Series), status(pandas.Series))

    Examples
    --------
    >>> survival_status(data, 'T', 'E', 10, inplace=False)
    """
    if inplace:
        data.loc[(data[event_col] == 1) & (data[duration_col] > end_time), event_col] = 0
        data.loc[data[duration_col] > end_time, duration_col] = end_time
    else:
        T = data[duration_col].copy()
        E = data[event_col].copy()
        T[data[duration_col] > end_time] = end_time
        E[(data[event_col] == 1) & (data[duration_col] > end_time)] = 0
        return T, E