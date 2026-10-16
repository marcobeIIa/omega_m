"""
make_fisher_covmat.py -- starting proposal covmat for MontePython, from the
Fisher matrix of the fake_planck_realistic likelihood.

This is the script that produced models/two_bins_fisher.covmat (2026-10-06),
cleaned up so it runs from this folder. Same numbers, same method.

What it computes
  Likelihood: MontePython's mock-CMB likelihood as set up in two_bins.param:
    TT, EE, TE + lensing extraction (dd = l(l+1) C_l^phiphi), T-d cross
    neglected (neglect_TD = True), f_sky = 0.57, l = 2..2508, noise from
    montepython_fake/data/noise_fake_planck_realistic_two.dat.
  Per multipole the data covariance is the 3x3 matrix
        | TT+N_T   TE       0      |
        | TE       EE+N_P   0      |
        | 0        0        dd+N_dd|
  and the Fisher matrix is the standard Gaussian one
        F_ij = sum_l (2l+1)/2 f_sky Tr[C^-1 dC/dp_i C^-1 dC/dp_j],
  which is also the curvature of MontePython's mock-CMB likelihood at the
  fiducial. Derivatives: central finite differences, one CLASS (binned) run
  at p +- step for each parameter.
  Spectra: real CLASS through with_class/class/python/classy.py (the binned
  wrapper), at the fiducial below.

Output: a MontePython .covmat = inverse Fisher matrix, written in PHYSICAL
units (MontePython divides each entry by the parameter's 'scale' when it reads
the file, so omega_b is stored as omega_b, not 100*omega_b).

Usage (from omega_m/, in an environment with real classy):
    python models/make_fisher_covmat.py
Edit FIDUCIAL / STEP / L_SWITCH below for other setups.
"""

import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # omega_m/
sys.path.insert(0, os.path.join(ROOT, "with_class", "class", "python"))
from classy import Class  # noqa: E402  (binned real-CLASS wrapper)

NOISE_FILE = os.path.join(ROOT, "montepython_fake", "data",
                          "noise_fake_planck_realistic_two.dat")
OUT = os.path.join(ROOT, "models", "two_bins_fisher.covmat")

L_MIN, L_MAX, F_SKY = 2, 2508, 0.57
L_SWITCH = [1250, 2508]

# Parameters in MontePython chain units (omega_b sampled as 100*omega_b),
# with their 'scale' from the .param file.
NAMES = ["omega_b", "h", "tau_reio", "n_s", "ln10^{10}A_s", "omega_m0", "omega_m1"]
SCALE = {"omega_b": 0.01}
FIDUCIAL = {"omega_b": 2.24, "h": 0.674, "tau_reio": 0.054, "n_s": 0.965,
            "ln10^{10}A_s": 3.044, "omega_m0": 0.15, "omega_m1": 0.15}
STEP = {"omega_b": 0.01, "h": 0.002, "tau_reio": 0.002, "n_s": 0.002,
        "ln10^{10}A_s": 0.004, "omega_m0": 0.001, "omega_m1": 0.001}

T_MUK = 2.7255e6  # T_cmb in muK, same conversion MontePython applies


def main():
    ell = np.arange(L_MIN, L_MAX + 1)
    noise = np.loadtxt(NOISE_FILE)
    rows = {int(r[0]): r for r in noise}
    N_T = np.array([rows[l][1] for l in ell])
    N_P = np.array([rows[l][2] for l in ell])
    N_dd = np.array([rows[l][3] for l in ell]) / (ell * (ell + 1) / 2 / np.pi)

    fixed = {"output": "tCl lCl pCl", "lensing": "yes", "l_max_scalars": L_MAX}
    for i, ls in enumerate(L_SWITCH):
        fixed["l_switch%d" % (i + 1)] = ls

    cosmo = Class()

    def spectra(p):
        cosmo.struct_cleanup()
        cosmo.empty()
        cosmo.set(dict(fixed, **{k: v * SCALE.get(k, 1) for k, v in p.items()}))
        cosmo.compute(["lensing"])
        cl = cosmo.lensed_cl(L_MAX)
        return np.array([cl["tt"][ell] * T_MUK**2, cl["ee"][ell] * T_MUK**2,
                         cl["te"][ell] * T_MUK**2, ell * (ell + 1) * cl["pp"][ell]])

    def cov3(s, with_noise):
        tt, ee, te, dd = s
        C = np.zeros((len(ell), 3, 3))
        C[:, 0, 0] = tt + (N_T if with_noise else 0)
        C[:, 1, 1] = ee + (N_P if with_noise else 0)
        C[:, 0, 1] = C[:, 1, 0] = te
        C[:, 2, 2] = dd + (N_dd if with_noise else 0)
        return C

    t0 = time.time()
    C_inv = np.linalg.inv(cov3(spectra(FIDUCIAL), True))
    dC = {}
    for k in NAMES:
        up, dn = dict(FIDUCIAL), dict(FIDUCIAL)
        up[k] += STEP[k]
        dn[k] -= STEP[k]
        dC[k] = cov3((spectra(up) - spectra(dn)) / (2 * STEP[k]), False)
    print("CLASS runs done in %.0f s" % (time.time() - t0))

    n = len(NAMES)
    F = np.zeros((n, n))
    for i, a in enumerate(NAMES):
        A = C_inv @ dC[a]
        for j, b in enumerate(NAMES):
            B = C_inv @ dC[b]
            F[i, j] = np.sum((2 * ell + 1) / 2 * F_SKY * np.einsum("lij,lji->l", A, B))

    cov_chain = np.linalg.inv(F)                     # chain units
    sig = np.sqrt(np.diag(cov_chain))
    print("Fisher 1-sigma (chain units):")
    for k, s in zip(NAMES, sig):
        print("  %-14s %.4g" % (k, s))
    print("correlation matrix:")
    np.set_printoptions(precision=2, suppress=True, linewidth=150)
    print(cov_chain / np.outer(sig, sig))

    S = np.diag([SCALE.get(k, 1) for k in NAMES])
    cov_phys = S @ cov_chain @ S                     # what MontePython expects on disk
    with open(OUT, "w") as f:
        f.write("# " + ", ".join(NAMES) + "\n")
        for row in cov_phys:
            f.write(" ".join("%.6e" % x for x in row) + "\n")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
