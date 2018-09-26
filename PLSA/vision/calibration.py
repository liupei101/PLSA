#coding=utf-8
"""Module for visualizing curve of calibration test

The function of this Module is served for visualizing curve of calibration test.

"""
import matplotlib.pyplot as plt
from PLSA.utils import metrics
from PLSA.utils import test

def plot_DCalibration(y_true, pred_proba, n_bins=10, summary=True,
                      xlabel="Predicted value", ylabel="Observed average", 
                      title="Hosmer-Lemeshow Test", save_fig_as=""):
    """Plot calibration curve.

    Parameters
    ----------
    y_true : numpy.array
        True label.
    y_prob : numpy.array
        Predicted label.
    n_bins : int
        Number of groups.

    Returns
    -------
    None
        Summary table of result.

        Plot figure of calibration curve.

    Examples
    --------
    >>> plot_DCalibration(test_y, test_pred, n_bins=5)
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    lw = 2
    prob_bin_true, prob_bin_pred, bin_tot = metrics.calibration_table(y_true, pred_proba, n_bins=n_bins)
    v, p = test.Hosmer_Lemeshow_Test(prob_bin_true, prob_bin_pred, bin_tot, n_bins=n_bins)
    # summary
    if summary:
        print "__________Summary of Calibration__________"
        print "Hosmer Lemeshow Test:"
        print "\tchi2  =", v
        print "\tp     =", p
        print "Calibration Table:"
        print "\tTotal\tObs\tPred"
        for i in range(bin_tot.shape[0]):
            print "\t%d\t%d\t%.2f" % (bin_tot[i], prob_bin_true[i], prob_bin_pred[i])
    # plot
    plt.plot(prob_bin_pred / bin_tot, prob_bin_true / bin_tot, 
	         lw=lw, ls='-', marker='o',
	         label='$\chi^2$=%.2f, $p$=%.3f' % (v, p))
    plt.plot([0, 1], [0, 1], color='navy', lw=lw-1, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.01])
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend(loc="best", title=title)
    plt.show()
    if save_fig_as != "":
        fig.savefig(save_fig_as, format='png', dpi=600)