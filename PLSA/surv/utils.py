import numpy as np
import pandas as pd
import pyper as pr

def surv_roc(data, pred_col, duration_col, event_col, pt=None):
    """
    Get survival ROC at predicted time.

    Parameters:
        data: DataFrame, full survival data.
        pred_col: Name of column to reference for dividing groups.
        duration_col: Name of column indicating time.
        event_col: Name of column indicating event.
        pt: Predicted time.

    Returns:
        Object include "FP", "TP" and "AUC" in ROC.

    Examples:
        surv_roc(data, 'X', 'T', 'E', pt=5)
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

def prepare_data(x, label):
    """
    Prepare data for survival analyze(Deep Surival).

    Parameters:
        x: np.array, two-dimension array indicating variables.
        label: Python dict contain 'e', 't'.

    Returns:
        Sorted (x, label) tuple.

    Examples:
        prepare_data(data[x_cols].values, {'e': data['e'].values, 't': data['t'].values})
    """
    if isinstance(label, dict):
       e, t = label['e'], label['t']

    # Sort Training Data for Accurate Likelihood
    # sort array using pandas.DataFrame(According to DESC 't' and ASC 'e')  
    df1 = pd.DataFrame({'t': t, 'e': e})
    df1.sort_values(['t', 'e'], ascending=[False, True], inplace=True)
    sort_idx = list(df1.index)
    x = x[sort_idx]
    e = e[sort_idx]
    t = t[sort_idx]

    return x, {'e': e, 't': t}

def parse_data(x, label):
    """
    Parse raw-data for survival analyze(Deep Surival).

    Parameters:
        x: np.array, two-dimension array indicating variables.
        label: Python dict contain 'e', 't'.

    Returns:
        Sorted (x, e, t) tuple, index of people who is failure or at risk, and type of ties.

    Examples:
        parse_data(data[x_cols].values, {'e': data['e'].values, 't': data['t'].values})
    """
    # sort data by t
    x, label = prepare_data(x, label)
    e, t = label['e'], label['t']

    failures = {}
    atrisk = {}
    n, cnt = 0, 0

    for i in range(len(e)):
        if e[i]:
            if t[i] not in failures:
                failures[t[i]] = [i]
                n += 1
            else:
                # ties occured
                cnt += 1
                failures[t[i]].append(i)

            if t[i] not in atrisk:
                atrisk[t[i]] = []
                for j in range(0, i+1):
                    atrisk[t[i]].append(j)
            else:
                atrisk[t[i]].append(i)
    # when ties occured frequently
    if cnt >= n / 2:
        ties = 'efron'
    elif cnt > 0:
        ties = 'breslow'
    else:
        ties = 'noties'

    return x, e, t, failures, atrisk, ties