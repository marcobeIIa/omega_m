#!/usr/bin/env python
"""
generate_fake_data.py

Builds a mock "observed" Planck-like high-l TT/TE/EE bandpower data set
using your classy cosmopower shim (the same one imported in
test_classy_shim.py), for a fiducial model that is standard LCDM
EVERYWHERE except that Omega_m is allowed to step to a slightly different
value above l=1250 vs below l=1250.

This is a toy injection test: it lets you check whether a pipeline built
around the fake_planck_highl_TTTEEE likelihood can recover (or correctly
fail to find, if you set OMEGA_M_LOW_L == OMEGA_M_HIGH_L) a scale-dependent
Omega_m step.

It assumes your shim accepts "Om_bin<i>" parameters that apply to
different ell ranges, switching at the ell value(s) given in the L_SWITCH
env var -- exactly as in test_classy_shim.py. Here we only need ONE switch
point (1250), giving two bins: Om_bin0 (l < 1250) and Om_bin1 (l >= 1250).

Usage (run in the same environment you'd use for test_classy_shim.py):
    export COSMOPOWER_TT_PATH=/path/to/cmb_TT_NN
    # export COSMOPOWER_TE_PATH=/path/to/cmb_TE_NN   # if your shim needs these
    # export COSMOPOWER_EE_PATH=/path/to/cmb_EE_NN
    export L_SWITCH=1250
    python generate_fake_data.py --outdir fake_planck_data --seed 42

Output: fake_planck_data/fake_planck_highl_TTTEEE_bandpowers.dat
    Columns: l_center  Dl_TT  sigma_TT  Dl_TE  sigma_TE  Dl_EE  sigma_EE
    (values in uK^2; a spectrum is filled with NaN if your shim didn't
    return it -- the likelihood will just skip NaN columns)
"""
import argparse
import os
import sys
import numpy as np

from binning import make_bin_edges, bin_spectrum

try:
    from classy import Class  # must resolve to the shim, not real CLASS
except ImportError:
    sys.exit(
        "Could not import `classy`. Run this script from wherever your "
        "montepython_shim (or its PYTHONPATH entry) lives, exactly as you "
        "would for test_classy_shim.py."
    )

# --------------------------------------------------------------------------
# Fiducial LCDM parameters (Planck-2018-ish), shared everywhere in ell.
# --------------------------------------------------------------------------
FIDUCIAL_PARAMS = {
    "omega_b": 0.0224,
    "h": 0.674,
    "tau_reio": 0.054,
    "n_s": 0.965,
    "ln10^{10}A_s": 3.044,
}

# The injected step: Omega_m-like bin parameters below vs above L_SWITCH.
# "Slightly different" -> a few-percent step by default; tune to taste.
OMEGA_M_LOW_L = 0.12   # applies to l  < L_SWITCH  (Om_bin0)
OMEGA_M_HIGH_L = 0.17 # applies to l >= L_SWITCH  (Om_bin1)

L_SWITCH_DEFAULT = 1250
LMIN, LMAX, DELTA_L = 30, 2508, 30       # bandpower binning
FSKY = 0.6                                # toy sky fraction
NOISE_FLOOR_UK2 = {"tt": 1.5e-5, "te": 3.0e-5, "ee": 6.0e-5}  # toy, uK^2 units
NOISE_KNEE_L = 1600.0                     # toy scale where noise starts to bite


def toy_sigma_dl(l_centers, dl_signal, spec):
    """
    Very simple toy error model: cosmic-variance-like scaling of the signal
    itself, growing with an instrumental-noise-like term past a knee scale.
    This is NOT a real Planck covariance -- it's just enough structure
    (bigger errors at low l and at high l) to make the fake data set behave
    like a real one for pipeline-testing purposes. Replace freely if you
    have (or want to fake) something more realistic.
    """
    cosmic_var = np.abs(dl_signal) * np.sqrt(2.0 / ((2 * l_centers + 1) * FSKY))
    noise_term = NOISE_FLOOR_UK2[spec] * (1.0 + (l_centers / NOISE_KNEE_L) ** 4)
    return np.sqrt(cosmic_var ** 2 + noise_term ** 2)


def get_dl(cl, ell, key):
    """Convert a raw_cl() Cl column (K^2) to Dl in uK^2, if present."""
    if key not in cl:
        return None
    c = np.asarray(cl[key])
    return c * ell * (ell + 1) / (2 * np.pi) * 1e12


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir", default="fake_planck_data")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--noiseless", action="store_true",
                         help="Write the noiseless model instead of a noisy realization "
                              "(sigma column is still filled in either way).")
    parser.add_argument("--l-switch", type=float, default=L_SWITCH_DEFAULT,
                         help="ell at which omega_m__0 -> omega_m__1 (default: 1250).")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    # Make sure the shim sees the same switch point we're telling the user
    # to export -- if it's already set in the environment we leave it alone
    # so `export L_SWITCH=...` still takes precedence, but warn if they differ.
    params = dict(FIDUCIAL_PARAMS)
    params["omega_m0"] = OMEGA_M_LOW_L
    params["omega_m1"] = OMEGA_M_HIGH_L
    params["l_switch1"] = args.l_switch   # first (and only interior) edge: 1250
    params["l_switch2"] = LMAX            # final edge
    cosmo = Class()
    cosmo.set(params)
    cosmo.compute()
    cl = cosmo.raw_cl()
    ell = np.asarray(cl["ell"], dtype=float)

    bin_edges = make_bin_edges(LMIN, LMAX, DELTA_L)

    rows = None
    header_cols = ["l_center"]
    columns = {}
    for spec in ("tt", "te", "ee"):
        dl = get_dl(cl, ell, spec)
        if dl is None:
            print(f"NOTE: shim did not return '{spec}' -- filling with NaN "
                  f"(disable use_{spec} in the .data file, or check your shim).")
            continue
        centers, dl_binned = bin_spectrum(ell, dl, bin_edges)
        sigma = toy_sigma_dl(centers, dl_binned, spec)
        if not args.noiseless:
            dl_obs = dl_binned + rng.normal(0.0, sigma)
        else:
            dl_obs = dl_binned
        if rows is None:
            rows = centers
        columns[spec] = (dl_obs, sigma)
        header_cols += [f"Dl_{spec.upper()}", f"sigma_{spec.upper()}"]

    if rows is None:
        sys.exit("Shim returned none of tt/te/ee -- nothing to write.")

    n = len(rows)
    out = np.full((n, 1 + 2 * 3), np.nan)
    out[:, 0] = rows
    for i, spec in enumerate(("tt", "te", "ee")):
        if spec in columns:
            dl_obs, sigma = columns[spec]
            out[:, 1 + 2 * i] = dl_obs
            out[:, 2 + 2 * i] = sigma

    outfile = os.path.join(args.outdir, "fake_planck_highl_TTTEEE_bandpowers.dat")
    full_header = ("l_center Dl_TT sigma_TT Dl_TE sigma_TE Dl_EE sigma_EE "
                    f"(uK^2; generated with Om_bin0={OMEGA_M_LOW_L}, "
                    f"Om_bin1={OMEGA_M_HIGH_L}, L_SWITCH={args.l_switch}, "
                    f"seed={args.seed}, noiseless={args.noiseless})")
    np.savetxt(outfile, out, header=full_header)
    print(f"saved {outfile}  ({n} bandpowers, {LMIN}<=l<={LMAX}, delta_l={DELTA_L})")


if __name__ == "__main__":
    main()
