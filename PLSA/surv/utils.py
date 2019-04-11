#coding=utf-8
"""Module for utilitize function of survival analyze.

The function of this Module is served as utility of survival analyze.

"""
import numpy as np
import pandas as pd
import pyper as pr
from PLSA.data.processing import parse_surv

def surv_ci(data, pred_col, duration_col, event_col):
    """Concordance Index

    Parameters
    ----------
    data : pandas.DataFrame
        Full survival data.
    pred_col : str
        Name of column indicating log hazard ratio.
    duration_col : str
        Name of column indicating time.
    event_col : str
        Name of column indicating event.

    Returns
    -------
    `dict`
        Object of dict include details about CI.

    Examples
    --------
    >>> surv_ci(data, 'Pred', 'T', 'E')
    """
    X = data[pred_col].values
    T = data[duration_col].values
    E = data[event_col].values
    r = pr.R(use_pandas=True)
    r("library('survival')")
    r("library('Hmisc')")
    r.assign("t", T)
    r.assign("e", E)
    r.assign("x", X)
    r("src <- rcorr.cens(-x, Surv(t, e))")
    return r.src

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

def surv_data_at_risk(data, duration_col, event_col, points=None):
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
    >>> surv_data_at_risk(data, 'T', 'E', points=[0, 10, 20, 30, 40, 50])
    """
    Td = data[duration_col].value_counts()
    TList = list(Td.index)
    TList.sort()
    N, deaths, S = len(data), 0, 0
    # Initial
    res = [(0, N, 0)]
    for idx in TList:
        data_at_time = data[data[duration_col] == idx]
        deaths += sum(data_at_time[event_col])
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
    return pd.DataFrame({"Time": T, "Obs": Obs, "Deaths": Deaths}, 
                       columns=['Time', 'Obs', 'Deaths'])

def survival_baseline(data_X, data_E, data_T, hazard_ratio, algo="bsl"):
    """Estimate base survival function S0(t) based on data(X, label).
    Parameters
    ----------
    data_X : pandas.DataFrame
        Patients' covariates.
    data_E : pandas.Series
        Patients' observed event.
    data_T : pandas.Series
        Patients' observed time.
    hazard_ratio : numpy.array
        Predicted patients' hazard ratio.
    algo : string
        algorithm for estimating survival function.
        The options includes "wwe", "kp" and "bsl".
    Returns
    -------
    tuple
        tuple is (T0, ST), T0 of it means time points of base survival function, 
        ST of it means survival rate of base survival function.
    Examples
    --------
    >>> survival_baseline(data[['x1', 'x2', 'x2']], data['e'], data['t'], hazard_ratio, algo='wwe')
    Notes
    -----
    Algorithm for estimating basel survival function:
    (1). wwe: WWE(with ties)
    (2). kp: Kalbfleisch & Prentice Estimator(without ties)
    (3). bsl: breslow(with ties, but exists negative value)
    """
    # Get data for estimating S0(t)
    X = data_X.values
    label = {'t': data_T.values,
             'e': data_E.values}
    X, E, T, failures, atrisk, ties = parse_surv(X, label)

    s0 = [1]
    t0 = [0]
    if algo == 'wwe':
        for t in T[::-1]:
            if t in t0:
                continue
            t0.append(t)
            if t in atrisk:
                # R(t_i) - D_i
                trisk = [j for j in atrisk[t] if j not in failures[t]]
                dt = len(failures[t]) * 1.0
                s = np.sum(hazard_ratio[trisk])
                cj = 1 - dt / (dt + s)
                s0.append(np.exp(cj - 1))
            else:
                s0.append(1)
    elif algo == 'kp':
        for t in T[::-1]:
            if t in t0:
                continue
            t0.append(t)
            if t in atrisk:
                # R(t_i)
                trisk = atrisk[t]
                s = np.sum(hazard_ratio[trisk])
                si = hazard_ratio[failures[t][0]]
                cj = (1 - si / s) ** (1 / si)
                s0.append(np.exp(cj - 1))
            else:
                s0.append(1)
    elif algo == 'bsl':
        for t in T[::-1]:
            if t in t0:
                continue
            t0.append(t)
            if t in atrisk:
                # R(t_i)
                trisk = atrisk[t]
                dt = len(failures[t]) * 1.0
                s = np.sum(hazard_ratio[trisk])
                cj = 1 - dt / s
                s0.append(np.exp(cj - 1))
            else:
                s0.append(1)
    else:
        raise NotImplementedError('tie breaking method not recognized')
    # base survival function
    S0 = np.cumprod(s0, axis=0)
    T0 = np.array(t0)
    return T0, S0

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