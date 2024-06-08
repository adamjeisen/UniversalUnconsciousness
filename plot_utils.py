import matplotlib.pyplot as plt
import numpy as np

def clear_axes(ax):
    # Remove the top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    # Remove ticks on the top and right axes
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')

def plot_curve_with_se(curve, x_vals=None, label=None, linestyle=None, c=None, alpha=1, alpha_se=0.5, ax=None):
    if ax is None:
        ax = plt.gca()
    mean_vals = curve.mean(axis=0)
    se_vals = curve.std(axis=0)/np.sqrt(curve.shape[0])
    if x_vals is None:
        x_vals = np.arange(len(mean_vals))
    ln = ax.plot(x_vals, mean_vals, label=label, alpha=alpha, color=c, linestyle=linestyle)
    ax.fill_between(x_vals, mean_vals - se_vals, mean_vals + se_vals, alpha=alpha_se, color=c)

    return ln