#coding=utf-8
"""Module for statistical test

The function of this Module is served for statistical test.

"""
from scipy import stats
import numpy as np
import math
import pandas as pd

def Hosmer_Lemeshow_Test(bins_true, bins_pred, bins_tot, n_bins=10, in_sample=False):
    """Hosmer-Lemeshow Test for testing calibration.

    Parameters
    ----------
    bins_true : numpy.array
        True Number of people in each group.
    bins_pred : numpy.array
        Pred Number of people in each group.
    bins_tot : numpy.array
        Totol Number of people in each group.
    n_bins : int
        Number of groups.
    in_sample : bool, default False
        Is Calibration-Test in sample.

    Returns
    -------
    tuple
        chi2 value and P value. 

    Examples
    --------
    >>> Hosmer_Lemeshow_Test(bins_true, bins_pred, bins_tot, n_bins=5)
    """
    v_chi2 = sum((bins_true - bins_pred)**2 / bins_pred / (1.0 - bins_pred / bins_tot))
    degree_of_freedom = n_bins - 2 if in_sample else n_bins
    p = stats.chi2.sf(v_chi2, degree_of_freedom)
    return v_chi2, p

def Delong_Test(y_true, pred_a, pred_b):
    """Delong-Test for comparing two predictive model.

    Parameters
    ----------
    y_true : numpy.array or pandas.Series.
        True label.
    pred_a : numpy.array or pandas.Series.
        Prediction of model A.
    pred_b : numpy.array or pandas.Series.
        Prediction of model B.

    Returns
    -------
    tuple
        chi2 value and P-value.

    Examples
    --------
    >>> # pred_proba1 = xgb1.predict_proba(test_X)
    >>> # pred_proba2 = xgb2.predict_proba(test_X)
    >>> Delong_test(test_y, pred_proba1[:, 1], pred_proba2[:, 1])
    """
    idx = 0
    a_x, v_ax = [], []
    a_y, v_ay = [], []
    b_x, v_bx = [], []
    b_y, v_by = [], []
    for label in y_true:
        if label == 0:
            a_y.append(pred_a[idx])
            b_y.append(pred_b[idx])
        else:
            a_x.append(pred_a[idx])
            b_x.append(pred_b[idx])
        idx += 1
    n1 = len(a_x)
    n2 = len(a_y)
    for x in a_x:
        cnt = .0
        for y in a_y:
            if y < x:
                cnt += 1
            elif y == x:
                cnt += 0.5
        v_ax.append(cnt)
    for y in a_y:
        cnt = .0
        for x in a_x:
            if y < x:
                cnt += 1
            elif y == x:
                cnt += 0.5
        v_ay.append(cnt)
    for x in b_x:
        cnt = .0
        for y in b_y:
            if y < x:
                cnt += 1
            elif y == x:
                cnt += 0.5
        v_bx.append(cnt)
    for y in b_y:
        cnt = .0
        for x in b_x:
            if y < x:
                cnt += 1
            elif y == x:
                cnt += 0.5
        v_by.append(cnt)
    theta_a = sum(v_ax) / (n1 * n2)
    theta_b = sum(v_bx) / (n1 * n2)
    theta = np.array([theta_a, theta_b]).reshape((1, 2))
    V = np.array([v_ax, v_bx]).T / n2
    Z = np.array([v_ay, v_by]).T / n1
    Sv = np.dot((V - theta).T, (V - theta)) / (n1 - 1)
    Sz = np.dot((Z - theta).T, (Z - theta))/ (n2 - 1)
    L = np.array([[1.0, -1.0]])
    u = np.dot(L, theta.T) / np.sqrt(np.dot(np.dot(L, (Sv / n1) + (Sz / n2)), L.T))
    pval = stats.norm.sf(np.abs(u))
    return u, 2.0 * pval

def VIF_Test(data, cols=None):
    """Variance Inflation Factors for each variable.

    Parameters
    ----------
    data : pandas.DataFrame
        Targeted data.
    cols : list(str), default `None`
        Given columns to calculate VIF.

    Returns
    -------
    pandas.Series
        Return VIF for each variable included in cols.

    Examples
    --------
    >>> VIF_Test(data[x_cols])
    """
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    from statsmodels.tools.tools import add_constant
    if cols is None:
        cols = list(data.columns)
    X = add_constant(data[cols])
    res = pd.Series([variance_inflation_factor(X.values, i) for i in range(X.shape[1])], 
                    index=X.columns)
    print(res)
    return res