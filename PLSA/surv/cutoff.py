from sklearn import metrics
import numpy as np
import pandas as pd
from lifelines import CoxPHFitter
from data.proccessing import cut_groups

def loss_dis(data, c0, c1, c_min=0, c_max=100):
    data0 = data[(data['x'] >= c_min) & (data['x'] < c0)][['y']].values
    data1 = data[(data['x'] >= c0) & (data['x'] < c1)][['y']].values
    data2 = data[(data['x'] >= c1) & (data['x'] <= c_max)][['y']].values
    # zero-mean
    u  = np.mean(data[['y']].values, axis=0, keepdims=True)
    u0 = np.mean(data0, axis=0, keepdims=True)
    u1 = np.mean(data1, axis=0, keepdims=True)
    u2 = np.mean(data2, axis=0, keepdims=True)
    # calculate sw
    sw = np.dot(np.transpose(data0 - u0), data0 - u0) + np.dot(np.transpose(data1 - u1), data1 - u1) + np.dot(np.transpose(data2 - u2), data2 - u2)
    #print sw
    # calculate sb
    sb = data0.shape[0] * np.dot(np.transpose(u0 - u1), u0 - u1) + data2.shape[0] * np.dot(np.transpose(u2 - u1), u2 - u1)
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

def stats_var(data, score_min=0, score_max=100):
    """
    使用组内方差最小，组间间距最大的方法来进行数据分组.

    Parameters:
        args: description.

    Returns:
        args: description.

    Examples:
        f(a)
    """
    max_val = -1
    cut_off = (0, 0)
    for i in range(score_min + 1, score_max):
        for j in range(i + 1, score_max):
            loss = loss_dis(data, i, j, c_min=score_min, c_max=score_max)
            if loss[0][0] > max_val:
                cut_off = (i, j)
                max_val = loss
    return cut_off

def hazards_ratio(data, pred_col, duration_col, event_col, score_min=0, score_max=100, balance=True):
    """
    Cutoff maximize HR or BHR.

    Parameters:
        data: full survival data.
        col: Name of column to reference for dividing groups.
        score_min: min value of score.
        score_max: max value of score.
        balance: True if using BHR as metrics, otherwise HR.

    Returns:
        args: description.

    Examples:
        f(a)
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
            groups = cut_groups(data, pred_col, cutoffs=[score_min, i, j, score_max+1])
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