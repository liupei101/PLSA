#coding=utf-8
"""Module for evaluating model by many kinds of metrics 

The function of this Module is served for evaluating model by many kinds of metrics.

"""
from scipy import stats
import numpy as np
import math
import pandas as pd
from sklearn import metrics
from sklearn.utils import column_or_1d
from sklearn.metrics.classification import _check_binary_probabilistic_predictions
from PLSA.utils import test
from PLSA.utils import cutoff

def calibration_table(y_true, y_prob, normalize=False, n_bins=10):
    """Calibration table of predictive model.

    Parameters
    ----------
    y_true : `np.array` or `pandas.Series`
        True label.
    y_prob : `np.array` or `pandas.Series`
        Predicted label.
    n_bins : int
        Number of groups.

    Returns
    -------
    tuple(`numpy.array`)
        true, sum and total number of each group.

    Examples
    --------
    >>> calibration_table(y_test, y_pred, n_bins=5)
    """
    y_true = column_or_1d(y_true)
    y_prob = column_or_1d(y_prob)

    if normalize:  # Normalize predicted values into interval [0, 1]
        y_prob = (y_prob - y_prob.min()) / (y_prob.max() - y_prob.min())
    elif y_prob.min() < 0 or y_prob.max() > 1:
        raise ValueError("y_prob has values outside [0, 1] and normalize is "
                         "set to False.")

    y_true = _check_binary_probabilistic_predictions(y_true, y_prob)

    bins = np.linspace(0., 1. + 1e-8, n_bins + 1)        
    binids = np.digitize(y_prob, bins) - 1

    bin_sums = np.bincount(binids, weights=y_prob, minlength=len(bins))
    bin_true = np.bincount(binids, weights=y_true, minlength=len(bins))
    bin_total = np.bincount(binids, minlength=len(bins))

    nonzero = bin_total != 0
    # prob_true = (bin_true[nonzero] / bin_total[nonzero])
    # prob_pred = (bin_sums[nonzero] / bin_total[nonzero])

    return bin_true[nonzero], bin_sums[nonzero], bin_total[nonzero]

def calibration(y_true, pred_proba, n_bins=10, in_sample=False):
    """Calibration and test of predictive model.

    Parameters
    ----------
    y_true : `np.array` or `pandas.Series`
        True label.
    pred_proba : `np.array` or `pandas.Series`
        Predicted label.
    n_bins : int
        Number of groups.
    in_sample : bool, default `False`
        Is Calibration-Test in sample.

    Returns
    -------
    pandas.DataFrame
        Table of calibration.

    Examples
    --------
    >>> calibration(y_test, y_pred, n_bins=5)
    """
    prob_bin_true, prob_bin_pred, bin_tot = calibration_table(y_true, pred_proba, n_bins=n_bins)
    test.Hosmer_Lemeshow_Test(prob_bin_true, prob_bin_pred, bin_tot, n_bins=n_bins, in_sample=in_sample)
    return pd.DataFrame({"Total": bin_tot, "Obs": prob_bin_true, "Pred": prob_bin_pred})

def discrimination(y_true, y_pred_proba, threshold=None, name="Model X"):
    """Discrimination of classification model.

    Parameters
    ----------
    y_true : `np.array` or `pandas.Series`
        True label.
    pred_proba : `np.array` or `pandas.Series`
        Predicted label.
    threshold : float
        Cutoff value.
    name : str
        Title for printing.

    Returns
    -------
    dict
        Dict with kinds of metrics.

            {
                "points": threshold,
                "Sen": Re,
                "Spe": Spe,
                "Acc": Accuracy,
                "F1": F1
            }

    Examples
    --------
    >>> discrimination(y_true, y_pred_proba, threshold=0.21)
    """
    # default threshold
    if threshold == None:
        threshold, _ = cutoff.youden(y_true, y_pred_proba)
    y_pred = (y_pred_proba >= threshold).astype(np.int32)
    tn, fp, fn, tp = metrics.confusion_matrix(y_true, y_pred).ravel()
    print "-------------------------------"
    print "Metrics on %s:" % name
    print "Confusion Matrix"
    print "tp: %d    fp: %d\nfn: %d    tn: %d" % (tp, fp, fn, tn)
    Re = metrics.recall_score(y_true, y_pred)
    Spe = 1.0 * tn / (fp + tn)
    F1 = metrics.f1_score(y_true, y_pred)
    Accuracy = metrics.accuracy_score(y_true, y_pred)
    print "points\tSen\tSpe\tAcc\tF1"
    print "%f\t%f\t%f\t%f\t%f" % (threshold, Re, Spe, Accuracy, F1)
    return {
        "points": threshold,
        "Sen": Re,
        "Spe": Spe,
        "Acc": Accuracy,
        "F1": F1
    }

def discrimination_ver(y_true, y_pred_proba, threshold=None, option="accuracy", name="Model X"):
    """Discrimination of classification model in version 2.

    Parameters
    ----------
    y_true : `np.array` or `pandas.Series`
        True label.
    pred_proba : `np.array` or `pandas.Series`
        Predicted label.
    threshold : float
        Cutoff value.
    option: str
        "accuracy" or "youden".
    name : str
        Title for printing.

    Returns
    -------
    dict
        Dict with kinds of metrics.

            {
                "points": threshold,
                "Sen": Sen,
                "Spe": Spe,
                "PPV": ppv,
                "NPV": npv
            }

    Examples
    --------
    >>> discrimination_ver(y_true, y_pred_proba, threshold=0.21)
    """
    # default threshold
    if threshold == None:
        threshold, _ = cutoff.youden(y_true, y_pred_proba)
    y_pred = (y_pred_proba >= threshold).astype(np.int32)
    tn, fp, fn, tp = metrics.confusion_matrix(y_true, y_pred).ravel()
    if option == "accuracy":
        threshold_v = 1.0 * (tn + tp) / (tn + fp + fn + tp)
    elif option == "youden":
        threshold_v = 1.0 * tp / (tp + fn) + 1.0 * tn / (fp + tn) - 1.0
    else:
        threshold_v = -1
    print "-------------------------------"
    print "Metrics on %s:" % name
    print "Confusion Matrix"
    print "tp: %d    fp: %d\nfn: %d    tn: %d" % (tp, fp, fn, tn)
    Sen = 1.0 * tp / (tp + fn)
    Spe = 1.0 * tn / (fp + tn)
    ppv = 1.0 * tp / (tp + fp)
    npv = 1.0 * tn / (tn + fn)
    print "%.2f\n%.2f\n%.2f\n%.2f" % (100.0*Sen, 100.0*Spe, 100.0*ppv, 100.0*npv)
    return {
        "point": threshold,
        "point-v": threshold_v,
        "tp": tp,
        "fn": fn,
        "fp": fp,
        "tn": tn,
        "Sen": Sen,
        "Spe": Spe,
        "PPV": ppv,
        "NPV": npv
    }