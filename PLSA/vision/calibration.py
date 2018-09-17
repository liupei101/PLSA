import matplotlib.pyplot as plt
from PLSA.utils import metrics
from PLSA.utils import test

def plot_DCalibration(y_true, pred_proba, n_bins=10, 
                      xlabel="Predicted value", ylabel="Observed average", title="Hosmer-Lemeshow Test", save_fig_as=""):
    """
    Plot calibration curve.

    Parameters:
        y_true, y_prob: True and predicted label.
        n_bins: Number of groups.

    Returns:

    Examples:
        plot_DCalibration(test_y, test_pred, n_bins=5)
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    lw = 2
    prob_bin_true, prob_bin_pred, bin_tot = metrics.calibration_table(y_true, pred_proba, n_bins=n_bins)
    v, p = test.Hosmer_Lemeshow_Test(prob_bin_true, prob_bin_pred, bin_tot, n_bins=n_bins)
    plt.plot(prob_bin_pred / bin_tot, prob_bin_true / bin_tot, 
	         lw=lw,
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

def plot_SCalibration(data_train, data_test, pred_col, duration_col, event_col, pt=None,
                      xlabel="Predicted Survival Probability", 
                      ylabel="Observed Survival Probability", 
                      title="Model Performance", save_fig_as=""):
    """
    Short description about your function.

    Parameters:
        args: description.

    Returns:
        args: description.

    Examples:
        f(a)
    """
    # TODO