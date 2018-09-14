from sklearn import metrics
import numpy as np
import pandas as pd
import pyper as pr
from lifelines import CoxPHFitter
from PLSA.data import processing

def loss_dis(data, data_list, col):
    data0 = data_list[0][[col]].values 
    data1 = data_list[1][[col]].values
    data2 = data_list[2][[col]].values
    # zero-mean
    u  = np.mean(data[[col]].values, axis=0, keepdims=True)
    u0 = np.mean(data0, axis=0, keepdims=True)
    u1 = np.mean(data1, axis=0, keepdims=True)
    u2 = np.mean(data2, axis=0, keepdims=True)
    # calculate sw
    sw = np.dot(np.transpose(data0 - u0), data0 - u0) \
         + np.dot(np.transpose(data1 - u1), data1 - u1) \
         + np.dot(np.transpose(data2 - u2), data2 - u2)
    #print sw
    # calculate sb
    sb = data0.shape[0] * np.dot(np.transpose(u0 - u1), u0 - u1) \
         + data2.shape[0] * np.dot(np.transpose(u2 - u1), u2 - u1)
    sb = sb * data1.shape[0]
    #sb = data0.shape[0] * np.dot(np.transpose(u0 - u), u0 - u) + data1.shape[0] * np.dot(np.transpose(u1 - u), u1 - u) + data2.shape[0] * np.dot(np.transpose(u2 - u), u2 - u)
    #print sb
    return sb / sw

def coxph_coef(data, duration_col, event_col, silence=True):
    cph = CoxPHFitter()
    cph.fit(data, duration_col=duration_col, event_col=event_col, show_progress=(not silence))
    if not silence:
        cph.print_summary()
    return np.exp(cph.hazards_['div']['coef'])

def loss_hr(data_list, duration_col, event_col, base_val=0, silence=True):
    N_group = len(data_list)
    for i in range(N_group):
        data_list[i].loc[:, 'div'] = base_val + i
    data = pd.concat(data_list, axis=0)
    data.index = range(len(data))
    df = data[['div', event_col, duration_col]]
    return coxph_coef(df, duration_col, event_col, silence=silence)

def loss_bhr(data_list, duration_col, event_col, base_val=1, silence=True):
    N_group = len(data_list)
    L = []
    for i in range(N_group):
        data_list[i].loc[:, 'div'] = base_val + 2 * i
        L.append(data_list[i].copy())
        if i != N_group - 1:
            data_list[i].loc[:, 'div'] = base_val + 2 * i + 1
            data_list[i+1].loc[:, 'div'] = base_val + 2 * i + 1
            L.append(data_list[i].copy())
            L.append(data_list[i+1].copy())
    data = pd.concat(L, axis=0)
    data.index = range(len(data))
    df = data[['div', event_col, duration_col]]
    return coxph_coef(df, duration_col, event_col, silence=silence)

def stats_var(data, x_col, y_col, score_min=0, score_max=100):
    """
    Cutoff maximize distant between groups, minimize variance in group

    Parameters:
        data: pd.DataFrame, data set.
        x_col: Name of column to reference for dividing groups.
        y_col: Name of column to measure differences.
        score_min: Min value in x_col.
        score_max: Max value in x_col.

    Returns:
        Optimal cutoffs.

    Examples:
        stats_var(data, 'score', 'y')
    """
    max_val = -1
    cut_off = (0, 0)
    for i in range(score_min + 1, score_max):
        for j in range(i + 1, score_max):
            groups = processing.cut_groups(data, x_col, cutoffs=[score_min, i, j, score_max+1])
            loss = loss_dis(data, groups, y_col)
            if loss[0][0] > max_val:
                cut_off = (i, j)
                max_val = loss
    # print result
    print "____________Statistical Methods____________"
    print "Results of Maximize loss:"
    print "\tLoss :", max_val
    print "\tCutoff :", cut_off
    return cut_off

def hazards_ratio(data, pred_col, duration_col, event_col, score_min=0, score_max=100, balance=True):
    """
    Cutoff maximize HR or BHR.

    Parameters:
        data: DataFrame, full survival data.
        pred_col: Name of column to reference for dividing groups.
        duration_col: Name of column indicating time.
        event_col: Name of column indicating event.
        score_min: min value in pred_col.
        score_max: max value in pred_col.
        balance: True if using BHR as metrics, otherwise HR.

    Returns:
        Optimal cutoffs.

    Examples:
        hazards_ratio(data, 'score', 'T', 'E', balance=True)
    """
    # initialize
    max_val = -1
    cut_off = (0, 0)
    if balance:
        loss_func = loss_bhr
    else:
        loss_func = loss_hr
    # loop for all combinations of cutoff
    for i in range(score_min + 1, score_max):
        if i % 10 == 0:
            print "Proccessing:", str(i) + '%'
        for j in range(i + 1, score_max):
            groups = processing.cut_groups(data, pred_col, cutoffs=[score_min, i, j, score_max+1])
            flag = False
            for g in groups:
                if len(g) == 0:
                    flag = True
                    break
            if flag:
                continue
            loss = loss_func(groups, duration_col, event_col)
            if loss > max_val:
                cut_off = (i, j)
                max_val = loss
    # print result
    print "____________" + ("Balanced HR" if balance else "Unbalanced HR") + "____________"
    print "Results of Maximize HR:"
    print "\tHR :", max_val
    print "\tCutoff :", cut_off
    return cut_off

def youden_onecut(data, pred_col, duration_col, event_col, pt=None):
    """
    Cutoff maximize Youden Index.

    Parameters:
        data: DataFrame, full survival data.
        pred_col: Name of column to reference for dividing groups.
        duration_col: Name of column indicating time.
        event_col: Name of column indicating event.
        pt: Predicted time.

    Returns:
        value indicating cutoff for pred_col of data.

    Examples:
        youden_onecut(data, 'X', 'T', 'E')
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
    r("library(survivalROC)")
    r("src <- survivalROC(Stime = t, status = e, marker = mkr, predict.time = pt, span = 0.25*nobs^(-0.20))")
    r("Youden <- src$TP-src$FP")
    r("cutoff <- src$cut.values[which(Youden == max(Youden), arr.ind = T)]")
    r("abline(0,1)")
    return r.cutoff

def youden_twocut(data, pred_col, duration_col, event_col, pt=None):
    """
    Two values of cutoff maximize Youden Index.

    Parameters:
        data: DataFrame, full survival data.
        pred_col: Name of column to reference for dividing groups.
        duration_col: Name of column indicating time.
        event_col: Name of column indicating event.
        pt: Predicted time.

    Returns:
        (cutoff-1, cutoff-2) value indicating cutoff for pred_col of data.

    Examples:
        youden_twocut(data, 'X', 'T', 'E')
    """
    # Cut-off1
    cutoff = youden_onecut(data, pred_col, duration_col, event_col, pt=pt)
    Hp = data[pred_col] >= cutoff
    # cf1 cut for X1, cf2 cut for X2
    cutoff1 = youden_onecut(data[Hp], pred_col, duration_col, event_col, pt=pt)
    cutoff2 = youden_onecut(data[~Hp], pred_col, duration_col, event_col, pt=pt)
    return cutoff2, cutoff1