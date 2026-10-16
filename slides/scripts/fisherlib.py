import sys, numpy as np
sys.path.insert(0, '/tmp/claude-0/-home-claude/292e6737-93c7-5f4e-80bb-a872aa2b0bd3/scratchpad/deck/work/ob/models')
import optimal_bins as ob
U = "/mnt/user-data/uploads/omega_m"
d = np.load(U + "/models/fisher_derivatives.npz")
centre = {k: d["c_" + k] for k in ("tt", "ee", "te", "dd")}
deriv = {k: d["d_" + k] for k in ("tt", "ee", "te", "dd")}
LMIN, LMAX, FSKY = 2, 2508, 0.57
noise = ob.read_noise(U + "/montepython_fake/data/noise_fake_planck_realistic_two.dat", LMIN, LMAX)
Fl = ob.per_ell_fisher(centre, deriv, noise, LMIN, LMAX, FSKY)
bf = ob.BinnedFisher(Fl, LMIN, LMAX)
NT = 5

def omega_true(steps, om_fid=0.15):
    w = np.zeros(LMAX + 1)
    for lo, hi, v in steps:
        w[lo:hi + 1] = v - om_fid
    return w

def linear_fit(steps, edges):
    """Linear-response best fit of the binned model (edges = l_switch list w/o l_max) to a
    noiseless mock with omega_m(l) steps; returns (delta theta, omega_m bins, chi2_min)."""
    dw = omega_true(steps)
    F = bf.full_fisher(list(edges))
    b = np.zeros(F.shape[0])
    b[:NT] = (Fl[:, :NT, NT] * dw[:, None]).sum(axis=0)
    B = bf.boundaries(list(edges)).astype(int)
    for i in range(len(B) - 1):
        sl = slice(B[i] + 1, B[i + 1] + 1)
        b[NT + i] = (Fl[sl, NT, NT] * dw[sl]).sum()
    p = np.linalg.solve(F, b)
    chi2_full = (Fl[:, NT, NT] * dw ** 2).sum()
    chi2_min = chi2_full - b @ p
    return p, chi2_min
