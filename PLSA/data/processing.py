#coding=utf-8
import pandas as pd
import numpy as np

def cut_groups(data, col, cutoffs):
    """
    Cut data into subsets according to cutoffs

    Parameters:
        data: pandas.DataFrame, data to split.
        col: columns in data to compare with cutoffs.
        cutoffs: cutoffs, like as [min-value, 30, 60, max-value].

    Returns:
        list of sub-data as DataFrame.

    Examples:
        cut_groups(data, "X", [0, 0.4, 0.6, 1.0])
    """
    res = []
    N = len(cutoffs)
    for i in range(N - 1):
        if i == N - 2:
            df = data[(data[col] >= cutoffs[i]) & (data[col] <= cutoffs[i+1])]
        else:
            df = data[(data[col] >= cutoffs[i]) & (data[col] < cutoffs[i+1])]
        res.append(df)
    return res

def prepare_surv(x, label):
    """
    Prepare data for survival analyze(Deep Surival).

    Parameters:
        x: np.array, two-dimension array indicating variables.
        label: Python dict contain 'e', 't'.

    Returns:
        Sorted (x, label) tuple.

    Examples:
        prepare_surv(data[x_cols].values, {'e': data['e'].values, 't': data['t'].values})
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

def parse_surv(x, label):
    """
    Parse raw-data for survival analyze(Deep Surival).

    Parameters:
        x: np.array, two-dimension array indicating variables.
        label: Python dict contain 'e', 't'.

    Returns:
        Sorted (x, e, t) tuple, index of people who is failure or at risk, and type of ties.

    Examples:
        parse_surv(data[x_cols].values, {'e': data['e'].values, 't': data['t'].values})
    """
    # sort data by t
    x, label = prepare_surv(x, label)
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