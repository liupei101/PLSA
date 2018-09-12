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