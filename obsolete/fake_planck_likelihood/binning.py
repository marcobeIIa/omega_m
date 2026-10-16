"""
binning.py

Shared bandpower-binning utilities for the fake Planck high-l TT/TE/EE
likelihood.

Both `generate_fake_data.py` (which builds the mock "observed" bandpowers
once, offline) and `fake_planck_highl_TTTEEE/__init__.py` (which bins the
theory spectrum on every MCMC step) import this module, so the two binning
operations are guaranteed to stay identical. If you change the binning here,
both sides pick it up automatically -- just remember to re-run
generate_fake_data.py afterwards.
"""
import numpy as np


def make_bin_edges(lmin, lmax, delta_l):
    """
    Return an array of bin edges lmin, lmin+delta_l, ..., covering up to
    (at least) lmax.
    """
    edges = np.arange(lmin, lmax + delta_l, delta_l, dtype=float)
    if edges[-1] < lmax:
        edges = np.append(edges, float(lmax))
    return edges


def bin_spectrum(ell, dl, bin_edges):
    """
    Flat (uninverse-variance-weighted) average of dl(ell) inside each
    [edge_i, edge_i+1) bin. That's a fine approximation for a toy/fake
    likelihood -- a real analysis would inverse-variance weight and/or
    apply a bandpower window function.

    Parameters
    ----------
    ell, dl : 1-D arrays, same length
    bin_edges : 1-D array of bin edges, e.g. from make_bin_edges()

    Returns
    -------
    centers : bin centers (only for bins that actually contained data)
    binned  : binned dl values
    """
    ell = np.asarray(ell)
    dl = np.asarray(dl)
    centers, binned = [], []
    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        mask = (ell >= lo) & (ell < hi)
        if not np.any(mask):
            continue
        centers.append(0.5 * (lo + hi))
        binned.append(np.mean(dl[mask]))
    return np.array(centers), np.array(binned)
