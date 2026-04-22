"""
spectral_analysis.py

Eigenspectral analysis and visualization for QL-bit adjacency matrices.

Two computation paths are provided:
  - Exact: full eigendecomposition via eigh on the actual coupled matrix.
  - Approximate: convolution of single-bit spectra, avoids building the
    exponentially large Cartesian product matrix.

In both paths the spectrum is split into:
  - Emergent (coherent) states  — the top eigenvalues, kept as sharp features.
  - Incoherent bulk             — the remainder, Gaussian-smoothed.

For an n-bit QL-bit network the expected number of emergent states is 2^n_bits.
The approximate path fixes this at 2 per bit and derives the product count via
convolution; the exact path takes num_emergent as an explicit argument.

Public API
----------
compute_eigenspectrum(adj_matrix)                        -> (eigenvalues, eigenvectors)
spectral_density_exact(adj_matrix, num_emergent, ...)    -> (x, rho_bulk, rho_emergent)
spectral_density_approx(ql_adj_matrix, n_bits, ...)      -> (x, rho_bulk, rho_emergent)
plot_eigenvalue_histogram(eigenvalues, ...)              -> Figure
plot_spectral_density(x, rho_bulk, rho_emergent, ...)   -> Figure
"""

import numpy as np
from scipy.ndimage import gaussian_filter1d
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


# ---------------------------------------------------------------------------
# Eigendecomposition
# ---------------------------------------------------------------------------

def compute_eigenspectrum(adj_matrix):
    """
    Compute eigenvalues and eigenvectors of a symmetric adjacency matrix.

    Uses numpy.linalg.eigh (exploits symmetry, guarantees real eigenvalues).
    Results are sorted by eigenvalue magnitude descending.

    Parameters
    ----------
    adj_matrix : np.ndarray  Square symmetric adjacency matrix

    Returns
    -------
    eigenvalues  : np.ndarray  shape (N,), sorted by |λ| descending
    eigenvectors : np.ndarray  shape (N, N), column i paired with eigenvalues[i]
    """
    eigenvalues, eigenvectors = np.linalg.eigh(adj_matrix)
    idx = np.abs(eigenvalues).argsort()[::-1]
    return eigenvalues[idx], eigenvectors[:, idx]


# ---------------------------------------------------------------------------
# Spectral density — exact path
# ---------------------------------------------------------------------------

def spectral_density_exact(adj_matrix, num_emergent, x_range=None, bins=8000, sigma=50.0):
    """
    Compute spectral density from the full exact eigendecomposition.

    The `num_emergent` largest eigenvalues are treated as emergent (coherent)
    states; the remainder form the incoherent bulk.  Gaussian smoothing
    (width `sigma` bins) is applied only to the bulk.

    For an n-bit QL-bit network pass num_emergent = 2 ** n_bits.

    Parameters
    ----------
    adj_matrix   : np.ndarray       Symmetric adjacency matrix of the coupled system
    num_emergent : int              Number of top eigenvalues treated as emergent
    x_range      : (float, float) | None
                   Eigenvalue axis range.  Auto-derived from data if None.
    bins         : int              Number of histogram bins (default 8000)
    sigma        : float            Gaussian smoothing width in bins for bulk
                                    (default 50.0; set 0 to disable)

    Returns
    -------
    x_axis       : np.ndarray  Bin centres
    rho_bulk     : np.ndarray  Smoothed incoherent bulk density (normalised)
    rho_emergent : np.ndarray  Emergent state density as delta spikes (normalised)
    """
    eigenvalues = np.linalg.eigvalsh(adj_matrix)
    n_total = len(eigenvalues)
    sorted_evs = np.sort(eigenvalues)

    if x_range is None:
        spread = sorted_evs[-1] - sorted_evs[0]
        pad = spread * 0.1
        x_range = (sorted_evs[0] - pad, sorted_evs[-1] + pad)

    emergent_evs = sorted_evs[-num_emergent:]
    bulk_evs = sorted_evs[:-num_emergent]

    bin_edges = np.linspace(x_range[0], x_range[1], bins + 1)
    x_axis = (bin_edges[:-1] + bin_edges[1:]) / 2
    dx = x_axis[1] - x_axis[0]

    rho_bulk, _ = np.histogram(bulk_evs, bins=bin_edges)
    rho_bulk = rho_bulk.astype(float) / n_total / dx

    if sigma > 0:
        rho_bulk = gaussian_filter1d(rho_bulk, sigma=sigma)

    rho_emergent = np.zeros_like(x_axis)
    for ev in emergent_evs:
        i = int(np.argmin(np.abs(x_axis - ev)))
        rho_emergent[i] += (1.0 / n_total) / dx

    return x_axis, rho_bulk, rho_emergent


# ---------------------------------------------------------------------------
# Spectral density — approximate (convolution) path
# ---------------------------------------------------------------------------

def spectral_density_approx(ql_adj_matrix, n_bits, x_range=None, bins=8000, sigma=50.0):
    """
    Approximate spectral density for an n-bit Cartesian-product QL-bit network
    via convolution of single-bit spectra.

    The single-bit spectrum is split into 2 emergent states and a bulk.
    Convolution for each additional bit follows the product rule:
      new_emergent = emergent ⊛ emergent
      new_bulk     = (bulk ⊛ bulk) + (bulk ⊛ emergent) + (emergent ⊛ bulk)

    This keeps emergent states from contaminating the smooth bulk hump and
    avoids constructing the full (2N)^(2*n_bits) coupled matrix.

    For large n_bits the output length grows with each convolution pass;
    the returned x_axis is automatically rescaled to cover [x_range[0]*n_bits,
    x_range[1]*n_bits].

    Parameters
    ----------
    ql_adj_matrix : np.ndarray  Adjacency matrix of a single QL-bit
    n_bits        : int         Number of QL-bits coupled via Cartesian product (≥ 1)
    x_range       : (float, float) | None
                    Eigenvalue range for the single QL-bit.  Auto-derived if None.
                    The output axis is n_bits × this range.
    bins          : int         Histogram bins for the single-bit base (default 8000)
    sigma         : float       Gaussian smoothing width in bins for bulk
                                (default 50.0; set 0 to disable)

    Returns
    -------
    x_axis       : np.ndarray  Eigenvalue axis (length grows with n_bits)
    rho_bulk     : np.ndarray  Smoothed incoherent bulk density
    rho_emergent : np.ndarray  Emergent state density
    """
    _NUM_EMERGENT_PER_BIT = 2

    eigenvalues = np.linalg.eigvalsh(ql_adj_matrix)
    sorted_evs = np.sort(eigenvalues)
    n_total = len(sorted_evs)

    if x_range is None:
        spread = sorted_evs[-1] - sorted_evs[0]
        pad = spread * 0.1
        x_range = (sorted_evs[0] - pad, sorted_evs[-1] + pad)

    emergent_evs = sorted_evs[-_NUM_EMERGENT_PER_BIT:]
    bulk_evs = sorted_evs[:-_NUM_EMERGENT_PER_BIT]

    bin_edges = np.linspace(x_range[0], x_range[1], bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    dx = bin_centers[1] - bin_centers[0]

    base_bulk, _ = np.histogram(bulk_evs, bins=bin_edges)
    base_bulk = base_bulk.astype(float) / n_total / dx

    base_em = np.zeros_like(bin_centers)
    for ev in emergent_evs:
        i = int(np.argmin(np.abs(bin_centers - ev)))
        base_em[i] += (1.0 / n_total) / dx

    curr_bulk = base_bulk.copy()
    curr_em = base_em.copy()

    for _ in range(1, n_bits):
        new_em = np.convolve(curr_em, base_em, mode='full') * dx

        term1 = np.convolve(curr_bulk, base_bulk, mode='full') * dx
        term2 = np.convolve(curr_bulk, base_em,   mode='full') * dx
        term3 = np.convolve(curr_em,   base_bulk, mode='full') * dx
        new_bulk = term1 + term2 + term3

        curr_bulk = new_bulk
        curr_em   = new_em

    if sigma > 0:
        curr_bulk = gaussian_filter1d(curr_bulk, sigma=sigma)

    x_axis = np.linspace(x_range[0] * n_bits, x_range[1] * n_bits, len(curr_bulk))

    return x_axis, curr_bulk, curr_em


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_eigenvalue_histogram(
    eigenvalues,
    x_range=None,
    bins=200,
    color='steelblue',
    show_plot=False,
    save_path=None,
):
    """
    Histogram of all eigenvalues — no emergent/bulk separation.

    Parameters
    ----------
    eigenvalues : np.ndarray         1-D array of eigenvalues
    x_range     : (float, float) | None  Axis limits; auto if None
    bins        : int                Number of histogram bins (default 200)
    color       : str                Bar colour (default 'steelblue')
    show_plot   : bool               Display interactively (default False)
    save_path   : str | None         PNG path; defaults to 'eigenvalue_histogram.png'

    Returns
    -------
    matplotlib.figure.Figure
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.hist(eigenvalues, bins=bins, color=color, alpha=0.8, edgecolor='none')

    if x_range is not None:
        ax.set_xlim(x_range)

    ax.set_xlabel(r'$\lambda$', fontsize=36)
    ax.set_ylabel('Count', fontsize=28)

    for spine in ax.spines.values():
        spine.set_linewidth(2)
    ax.tick_params(axis='both', which='major', labelsize=20, length=12, width=2, direction='inout')

    plt.tight_layout()

    if save_path is None:
        save_path = 'eigenvalue_histogram.png'
    _save_and_show(fig, save_path, show_plot)
    return fig


def plot_spectral_density(
    x,
    rho_bulk,
    rho_emergent,
    x_range=None,
    y_range=(1e-6, 10),
    color='forestgreen',
    label=None,
    log_scale=True,
    show_plot=False,
    save_path=None,
):
    """
    Publication-style spectral density plot.

    Emergent (coherent) states are drawn as vertical lines from the y-axis
    floor to their density value; the incoherent bulk is drawn as a smooth
    continuous curve.  Matches the aesthetic of the graph_spectra_viz notebook
    (log y-scale, large tick marks, LaTeX axis labels, thick border).

    Parameters
    ----------
    x            : np.ndarray  Eigenvalue axis
    rho_bulk     : np.ndarray  Bulk (incoherent) spectral density
    rho_emergent : np.ndarray  Emergent state density (delta spikes)
    x_range      : (float, float) | None  x-axis limits; auto if None
    y_range      : (float, float)          y-axis limits (default (1e-6, 10))
    color        : str                     Colour for both bulk curve and emergent lines
    label        : str | None              Legend label; no legend if None
    log_scale    : bool                    Use log y-axis (default True)
    show_plot    : bool                    Display interactively (default False)
    save_path    : str | None              PNG path; defaults to 'spectral_density.png'

    Returns
    -------
    matplotlib.figure.Figure
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    y_floor = y_range[0] if log_scale else 0.0

    # Bulk as a smooth curve; small floor prevents log(0) issues
    ax.plot(x, np.maximum(rho_bulk, y_floor * 0.1), color=color, lw=4, alpha=0.85, label=label)

    # Emergent states as vertical lines at non-zero bins
    em_mask = rho_emergent > 0
    for xi, yi in zip(x[em_mask], rho_emergent[em_mask]):
        ax.vlines(xi, y_floor, yi, colors=color, lw=2.5, alpha=0.9)

    if log_scale:
        ax.set_yscale('log')
        ax.set_ylim(y_range)

    if x_range is not None:
        ax.set_xlim(x_range)

    ax.set_xlabel(r'$\lambda$', fontsize=48)
    ax.set_ylabel(r'$\rho(\lambda)$', fontsize=48)

    for spine in ax.spines.values():
        spine.set_linewidth(2)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(20))
    ax.tick_params(axis='both', which='major', labelsize=30, length=20, width=4, direction='inout')

    if label is not None:
        ax.legend(fontsize=16)

    plt.tight_layout()

    if save_path is None:
        save_path = 'spectral_density.png'
    _save_and_show(fig, save_path, show_plot)
    return fig


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _save_and_show(fig, save_path, show_plot):
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f'Saved: {save_path}')
    if show_plot:
        plt.show()
    else:
        plt.close(fig)
