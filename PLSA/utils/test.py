from scipy import stats
import numpy as np
import math
import pandas as pd

def Hosmer_Lemeshow_Test(bins_true, bins_pred, bins_tot, n_bins=10, in_sample=False):
    """
    Hosmer-Lemeshow Test for testing calibration.

    Parameters:
        bins_true: True Number of people in each group.
        bins_pred: Pred Number of people in each group.
        bins_tot: Totol Number of people in each group.
        n_bins: Number of groups.
        in_sample: Is Calibration-Test in sample.

    Returns:
        chi2 value and P. 

    Examples:
        Hosmer_Lemeshow_Test(bins_true, bins_pred, bins_tot, n_bins=5)
    """
    v_chi2 = sum((bins_true - bins_pred)**2 / bins_pred / (1.0 - bins_pred / bins_tot))
    degree_of_freedom = n_bins - 2 if in_sample else n_bins
    p = stats.chi2.sf(v_chi2, degree_of_freedom)
    print "__________Hosmer-Lemeshow-Test__________"
    print "\tChi2 =", v_chi2
    print "\tP = ", p
    return v_chi2, p

def Delong_Test(y_true, pred_a, pred_b):
    """
    Delong-Test for comparing two predictive model.

    Parameters:
        y_true: true label.
        pred_a: model A predict.
        pred_b: model B predict

    Returns:
        chi2 value, P

    Examples:
        # pred_proba1 = xgb1.predict_proba(test_X)
        # pred_proba2 = xgb2.predict_proba(test_X)
        Delong_test(test_y, pred_proba1[:, 1], pred_proba2[:, 1])
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