import numpy as np
import math
import pandas as pd
from sklearn import metrics

def accuracy(y_true, y_prob):
    """
    Cutoff maximize accuracy.

    Parameters:
        y_true, y_prob: true and predicted value.

    Returns:
        optimal cutoff and max metrics. 

    Examples:
        accuracy(y_true, y_prob)
    """
    fpr, tpr, threshold = metrics.roc_curve(y_true, y_prob)
    max_acc = 0.0
    res_Cutoff = 0.0
    cut_off = []
    for i in range(len(tpr)):
        y_new_pred = (y_prob >= threshold[i]).astype(np.int32)
        Acc = metrics.accuracy_score(y_true, y_new_pred)
        if Acc > max_acc:
            max_acc = Acc
            res_Cutoff = threshold[i]
            cut_off = [fpr[i], tpr[i]]
    return res_Cutoff, max_acc

def youden(target, predicted):
    """
    Cutoff maximize Youden Index.

    Parameters:
        target, predicted: true and predicted value.

    Returns:
        optimal cutoff and max metrics.

    Examples:
        youden(y_true, y_prob)
    """
    fpr, tpr, threshold = metrics.roc_curve(target, predicted)
    max_yuden = 0.0
    res_Cutoff = 0.0
    cut_off = []
    for i in range(len(tpr)):
        if tpr[i] - fpr[i] > max_yuden:
            max_yuden = tpr[i] - fpr[i]
            res_Cutoff = threshold[i]
            cut_off = [fpr[i], tpr[i]]
    return res_Cutoff, max_yuden