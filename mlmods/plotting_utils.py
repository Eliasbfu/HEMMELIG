# plotting_utils.py
import matplotlib.pyplot as plt

def plot_convergence(ax, conv_dict, title, xlabel, ylabel="Train MSE"):
    """One axis: plot convergence curves from dict name -> [MSE_t]."""
    for name, series in conv_dict.items():
        ax.plot(series, label=name)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    ax.legend(ncol=2)

def plot_final_coeffs_with_cf(ax, family_dict, theta_cf, title, xlabel="Coeff index (0 = intercept)", ylabel="Value", cf_label="CF"):
    """
    One axis: plot final coefficient vectors for a family, plus a black solid CF reference.
    """
    for name, (th, _) in family_dict.items():
        ax.plot(th.ravel(), linewidth=1.2, alpha=0.9, label=name)
    ax.plot(theta_cf.ravel(), 'k-', linewidth=2.0, label=cf_label, zorder=10)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    ax.legend(ncol=2)