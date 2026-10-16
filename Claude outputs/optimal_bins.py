"""
optimal_bins.py -- omega_m bin edges at a fixed number of bins
==============================================================

What it does
  For a fixed number of bins N, find the edges l_switch1 < ... < l_switch{N-1}
  that MINIMISE THE LARGEST forecast error on adjacent differences:

        minimise   max_k  sigma(omega_m,k - omega_m,k+1),     k = 0 .. N-2

  sigma comes from the Fisher matrix of MontePython's fake_planck_realistic
  likelihood at a fiducial cosmology, marginalised over omega_b, h, tau_reio,
  n_s, ln10^{10}A_s and the other omega_m bins. No data enters: the edges
  depend only on the fiducial model and the noise.

Why it is cheap
  In the binned model omega_m,i only changes the spectra inside bin i, so at
  the fiducial (all bins equal)  dC_l/d omega_m,i = dC_l/d omega_m * 1[l in bin i].
  The derivatives are computed ONCE with real CLASS (6 parameters, 25 CLASS
  runs, ~1-2 min) and cached in a .npz file. After that, the Fisher matrix of
  any binning is a sum of per-l Fisher blocks over the bins (cumulative sums),
  so one binning costs ~tens of microseconds.

Likelihood: same as montepython Likelihood_mock_cmb with the fake_planck_realistic
settings (TT TE EE + lensing extraction, neglect_TD = True, f_sky = 0.57,
noise_fake_planck_realistic_two.dat, l = 2..2508):
      F_ab = sum_l (2l+1) f_sky/2 Tr[ C_l^-1 dC_l/da C_l^-1 dC_l/db ]
      C_l  = [[TT+N_T, TE, 0], [TE, EE+N_P, 0], [0, 0, l(l+1)C_l^pp + N_dd]]  (muK^2)
Spectra come from with_class/class/python/classy.py (the binned real-CLASS
wrapper) run with ONE bin, so the parametrisation is exactly the one MontePython
samples (omega_cdm = omega_m - omega_b).

Optimisation, for each N
  N = 2: every integer edge is evaluated (exact optimum).
  N > 2: differential evolution on the edges (several seeds), then Nelder-Mead,
         then an integer local search. If the seeds end on different edges,
         this is printed.
  Bins narrower than --min_width multipoles are not allowed.

Output, for each N
  the edges, sigma(omega_m,i), sigma of each adjacent difference and its
  correlation,
  the best edges restricted to multiples of 10, 25 and 50 with what that costs,
  the same numbers for equal fixed-parameter-information edges (for comparison),
  and the l_switch / omega_m lines for a MontePython .param file.

Usage, from omega_m/ (needs the real classy installed, like with_class/):
  python models/optimal_bins.py --n_bins 3
  python models/optimal_bins.py --n_bins 2 3 4 5 6
  python models/optimal_bins.py --edges 1250             # evaluate given edges (N=2)
  python models/optimal_bins.py --n_bins 4 --covmat_out models/four_bins_fisher.covmat
  python models/optimal_bins.py --n_bins 3 --brute 5     # brute-force check, grid step 5
  python models/optimal_bins.py --n_bins 3 --recompute --step_scale 0.5   # derivative-step check

Files written: only --deriv_file (derivative cache, default
models/fisher_derivatives.npz), and --out / --covmat_out if you ask for them.
"""

import argparse
import json
import math
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # omega_m/
WRAPPER_DIR = os.path.join(ROOT, "with_class", "class", "python")
NOISE_FILE = os.path.join(ROOT, "montepython_fake", "data",
                          "noise_fake_planck_realistic_two.dat")
DERIV_FILE = os.path.join(ROOT, "models", "fisher_derivatives.npz")

# Parameter order everywhere: the 5 "theta" parameters, then omega_m.
PARAMS = ["omega_b", "h", "tau_reio", "n_s", "ln10^{10}A_s", "omega_m"]
N_THETA = 5
# Derivative steps (5-point stencil), times --step_scale. Each is ~0.5-1 sigma.
STEPS = {"omega_b": 1e-4, "h": 0.005, "tau_reio": 0.004, "n_s": 0.004,
         "ln10^{10}A_s": 0.008, "omega_m": 0.001}


# =============================================================================
# 1. Spectra and derivatives (real CLASS through the binned wrapper, 1 bin)
# =============================================================================

def read_noise(path, l_min, l_max):
    """Same parsing as MontePython's Likelihood_mock_cmb (noise_from_file).
    Returns arrays indexed by l (0..l_max), zero below l_min."""
    nT, nP, ndd = (np.zeros(l_max + 1) for _ in range(3))
    seen = set()
    with open(path) as f:
        for line in f:
            if "#" in line or not line.strip():
                continue
            c = line.split()
            l = int(float(c[0]))
            if l_min <= l <= l_max:
                nT[l] = float(c[1])
                nP[l] = float(c[2])
                ndd[l] = float(c[3]) / (l * (l + 1) / 2. / math.pi)
                seen.add(l)
    missing = [l for l in range(l_min, l_max + 1) if l not in seen]
    if missing:
        raise ValueError("noise file has no entry for l = %s ..." % missing[:5])
    return nT, nP, ndd


def make_spectra_function(l_max):
    """Returns spectra(p) -> dict tt, ee, te (muK^2) and dd = l(l+1)C_l^pp,
    indexed by l = 0..l_max, from the binned wrapper with a single bin."""
    sys.path.insert(0, WRAPPER_DIR)
    import classy
    if not hasattr(classy, "REAL_CLASSY_FILE"):
        sys.exit("imported %s, which is not the with_class binned wrapper" % classy.__file__)
    cosmo = classy.Class()
    ell = np.arange(l_max + 1, dtype=float)

    def spectra(p):
        cosmo.empty()
        cosmo.set({"omega_b": p["omega_b"], "h": p["h"], "tau_reio": p["tau_reio"],
                   "n_s": p["n_s"], "ln10^{10}A_s": p["ln10^{10}A_s"],
                   "omega_m0": p["omega_m"], "l_switch1": l_max,
                   "l_max_scalars": l_max, "output": "tCl lCl pCl", "lensing": "yes"})
        cosmo.compute()
        cl = cosmo.lensed_cl(l_max)
        T = cosmo.T_cmb() * 1e6      # same conversion as MontePython's get_cl
        return {"tt": cl["tt"] * T**2, "ee": cl["ee"] * T**2, "te": cl["te"] * T**2,
                "dd": ell * (ell + 1.) * cl["pp"]}

    return spectra, classy.REAL_CLASSY_FILE


def compute_derivatives(fid, l_max, step_scale):
    """Central spectra and 5-point-stencil derivatives for the 6 parameters."""
    spectra, real_file = make_spectra_function(l_max)
    t0 = time.time()
    centre = spectra(fid)
    keys = ("tt", "ee", "te", "dd")
    deriv = {k: np.zeros((len(PARAMS), l_max + 1)) for k in keys}
    steps = {}
    for a, name in enumerate(PARAMS):
        h = STEPS[name] * step_scale
        steps[name] = h
        s = {}
        for m in (-2, -1, 1, 2):
            p = dict(fid)
            p[name] = fid[name] + m * h
            s[m] = spectra(p)
        for k in keys:
            deriv[k][a] = (-s[2][k] + 8 * s[1][k] - 8 * s[-1][k] + s[-2][k]) / (12 * h)
        print("  derivative %-14s step %.2e  done (%.0f s)" % (name, h, time.time() - t0))
    return centre, deriv, steps, real_file


def load_or_compute_derivatives(a, fid):
    """Load the cache if it matches the fiducial / l_max / step scale, else compute."""
    if os.path.exists(a.deriv_file) and not a.recompute:
        d = np.load(a.deriv_file, allow_pickle=False)
        same = (int(d["l_max"]) == a.l_max and float(d["step_scale"]) == a.step_scale
                and np.allclose(d["fid_values"], [fid[p] for p in PARAMS], rtol=0, atol=1e-12))
        if same:
            print("derivatives: loaded %s" % a.deriv_file)
            centre = {k: d["c_" + k] for k in ("tt", "ee", "te", "dd")}
            deriv = {k: d["d_" + k] for k in ("tt", "ee", "te", "dd")}
            return centre, deriv
        print("derivatives: cache %s was made for a different fiducial, l_max or "
              "step scale -> recomputing" % a.deriv_file)
    print("derivatives: computing with real CLASS (25 runs) ...")
    centre, deriv, steps, real_file = compute_derivatives(fid, a.l_max, a.step_scale)
    os.makedirs(os.path.dirname(os.path.abspath(a.deriv_file)), exist_ok=True)
    np.savez(a.deriv_file, l_max=a.l_max, step_scale=a.step_scale,
             params=np.array(PARAMS), fid_values=np.array([fid[p] for p in PARAMS]),
             steps=np.array([steps[p] for p in PARAMS]), real_classy=real_file,
             **{"c_" + k: v for k, v in centre.items()},
             **{"d_" + k: v for k, v in deriv.items()})
    print("derivatives: saved to %s" % a.deriv_file)
    return centre, deriv


# =============================================================================
# 2. Per-l Fisher blocks
# =============================================================================

def per_ell_fisher(centre, deriv, noise, l_min, l_max, f_sky):
    """F_l[a, b] for every l (shape (l_max+1, 6, 6), zero below l_min)."""
    nT, nP, ndd = noise
    sl = slice(l_min, l_max + 1)
    ell = np.arange(l_min, l_max + 1, dtype=float)
    # TE block: M = [[TT+N_T, TE], [TE, EE+N_P]]
    a_ = centre["tt"][sl] + nT[sl]
    b_ = centre["ee"][sl] + nP[sl]
    c_ = centre["te"][sl]
    det = a_ * b_ - c_ ** 2
    Minv = np.empty((len(ell), 2, 2))
    Minv[:, 0, 0], Minv[:, 1, 1] = b_ / det, a_ / det
    Minv[:, 0, 1] = Minv[:, 1, 0] = -c_ / det
    dM = np.empty((len(PARAMS), len(ell), 2, 2))
    dM[:, :, 0, 0] = deriv["tt"][:, sl]
    dM[:, :, 1, 1] = deriv["ee"][:, sl]
    dM[:, :, 0, 1] = dM[:, :, 1, 0] = deriv["te"][:, sl]
    X = np.einsum("lij,pljk->plik", Minv, dM)           # M^-1 dM_p
    F_te = np.einsum("plij,qlji->lpq", X, X)              # Tr[M^-1 dM_p M^-1 dM_q]
    # Deflection block (TD neglected, as in fake_planck_realistic)
    D = centre["dd"][sl] + ndd[sl]
    dD = deriv["dd"][:, sl]
    F_dd = np.einsum("pl,ql->lpq", dD, dD) / (D ** 2)[:, None, None]
    Fl = np.zeros((l_max + 1, len(PARAMS), len(PARAMS)))
    Fl[sl] = ((2 * ell + 1) * f_sky / 2.)[:, None, None] * (F_te + F_dd)
    return Fl


# =============================================================================
# 3. Fisher of any binning
# =============================================================================

class BinnedFisher:
    """Fisher matrix of the N-bin model for any edges, from cumulative sums.

    Edges are the l_switch values: bin 0 = [l_min, e_1], bin i = (e_i, e_{i+1}],
    last bin ends at l_max (same convention as classy.py)."""

    def __init__(self, Fl, l_min, l_max, prior_sigma=None):
        self.l_min, self.l_max = l_min, l_max
        A = Fl[:, :N_THETA, :N_THETA].sum(axis=0)
        if prior_sigma:
            for name, s in prior_sigma.items():
                A[PARAMS.index(name), PARAMS.index(name)] += 1. / s ** 2
        self.A = A
        self.Ainv = np.linalg.inv(A)
        # S[l] = sum_{l' <= l} F_l'  (F is zero below l_min, so S[l_min-1] = 0)
        self.Sb = np.cumsum(Fl[:, :N_THETA, N_THETA], axis=0)   # (l_max+1, 5)
        self.Sw = np.cumsum(Fl[:, N_THETA, N_THETA])            # (l_max+1,)
        self.x = np.arange(l_max + 1, dtype=float)

    def boundaries(self, edges):
        return np.concatenate([[self.l_min - 1], np.asarray(edges, float), [self.l_max]])

    def _sums(self, b):
        """Per-bin sums. b may be fractional (linear interpolation of the
        cumulative sums: a smooth stand-in used only inside the optimiser)."""
        if np.all(b == np.round(b)):
            bi = b.astype(int)
            Sb, Sw = self.Sb[bi], self.Sw[bi]
        else:
            Sb = np.stack([np.interp(b, self.x, self.Sb[:, j]) for j in range(N_THETA)], axis=-1)
            Sw = np.interp(b, self.x, self.Sw)
        return Sb[1:] - Sb[:-1], Sw[1:] - Sw[:-1]

    def omega_cov(self, edges):
        """Marginalised covariance of (omega_m,0 .. omega_m,N-1):
        inverse of  diag(F_ww,i) - B^T A^-1 B  (Schur complement)."""
        B, Dw = self._sums(self.boundaries(edges))
        Fm = np.diag(Dw) - B @ self.Ainv @ B.T
        return np.linalg.inv(Fm)

    def full_fisher(self, edges):
        """(5+N)x(5+N) Fisher in the order omega_b, h, tau_reio, n_s, lnAs, omega_m0.."""
        B, Dw = self._sums(self.boundaries(edges))
        n = len(Dw)
        F = np.zeros((N_THETA + n, N_THETA + n))
        F[:N_THETA, :N_THETA] = self.A
        F[:N_THETA, N_THETA:] = B.T
        F[N_THETA:, :N_THETA] = B
        F[N_THETA:, N_THETA:] = np.diag(Dw)
        return F

    @staticmethod
    def diff_sigmas(C):
        k = np.arange(C.shape[0] - 1)
        var = C[k, k] + C[k + 1, k + 1] - 2 * C[k, k + 1]
        return np.sqrt(var), C[k, k + 1] / np.sqrt(C[k, k] * C[k + 1, k + 1])

    def widths(self, edges):
        return np.diff(self.boundaries(edges))

    def objective(self, edges, min_width):
        """max_k sigma(Delta_k); a large penalty for invalid edges."""
        e = np.sort(np.asarray(edges, float))
        w = self.widths(e)
        if np.any(w < min_width):
            return 1e3 + float(np.sum(np.clip(min_width - w, 0, None)))
        try:
            s, _ = self.diff_sigmas(self.omega_cov(e))
        except np.linalg.LinAlgError:
            return 1e3
        if not np.all(np.isfinite(s)):
            return 1e3
        return float(s.max())

    # -- batched version, used by the brute-force check ----------------------
    def objective_batch(self, E, min_width):
        """E: (M, N-1) integer edges. Returns max_k sigma(Delta_k) per row."""
        M = E.shape[0]
        b = np.concatenate([np.full((M, 1), self.l_min - 1), E,
                            np.full((M, 1), self.l_max)], axis=1)
        Sb, Sw = self.Sb[b], self.Sw[b]
        B = Sb[:, 1:] - Sb[:, :-1]                          # (M, N, 5)
        Dw = Sw[:, 1:] - Sw[:, :-1]                         # (M, N)
        Fm = -np.einsum("mia,ab,mjb->mij", B, self.Ainv, B)
        n = Dw.shape[1]
        Fm[:, np.arange(n), np.arange(n)] += Dw
        out = np.full(M, np.inf)
        ok = np.all(np.diff(b, axis=1) >= min_width, axis=1)
        C = np.linalg.inv(Fm[ok])
        k = np.arange(n - 1)
        var = C[:, k, k] + C[:, k + 1, k + 1] - 2 * C[:, k, k + 1]
        out[ok] = np.sqrt(var).max(axis=1)
        return out


# =============================================================================
# 4. Optimiser
# =============================================================================

def equal_information_edges(bf, n_bins):
    """Edges that split the fixed-parameter omega_m information (sum of F_ww,l)
    into n_bins equal parts -- the comparison binning and a starting point."""
    S = bf.Sw / bf.Sw[-1]
    return np.array([int(np.searchsorted(S, k / n_bins)) for k in range(1, n_bins)])


def integer_polish(bf, e, min_width, unit=1):
    """Integer local search on multiples of `unit`: single-edge and adjacent-pair
    moves with shrinking steps, accepting only improvements."""
    e = np.array(sorted(int(unit * round(v / unit)) for v in e))
    best = bf.objective(e, min_width)
    steps = [unit * s for s in (256, 128, 64, 32, 16, 8, 4, 2, 1)]
    improved = True
    while improved:
        improved = False
        for d in steps:
            moves = []
            for j in range(len(e)):
                for s in (d, -d):
                    m = np.zeros(len(e), int); m[j] = s; moves.append(m)
            for j in range(len(e) - 1):
                for s1, s2 in ((d, d), (-d, -d), (d, -d), (-d, d)):
                    m = np.zeros(len(e), int); m[j] = s1; m[j + 1] = s2; moves.append(m)
            for m in moves:
                t = e + m
                if np.any(np.diff(t) <= 0) or t[0] < bf.l_min or t[-1] >= bf.l_max:
                    continue
                v = bf.objective(t, min_width)
                if v < best - 1e-15:
                    e, best, improved = t, v, True
    return e, best


def optimise(bf, n_bins, min_width, seeds):
    """Minimise max_k sigma(Delta_k) over the N-1 edges. Returns edges, value,
    and the per-seed results (to check that the optimum is reproducible)."""
    lo, hi = bf.l_min - 1 + min_width, bf.l_max - min_width
    if n_bins == 2:   # exact: try every edge
        E = np.arange(lo, hi + 1)[:, None]
        v = bf.objective_batch(E, min_width)
        i = int(np.argmin(v))
        return E[i], float(v[i]), [(E[i], float(v[i]))]

    from scipy.optimize import differential_evolution, minimize
    f = lambda x: bf.objective(x, min_width)
    x0 = equal_information_edges(bf, n_bins).astype(float)
    runs = []
    for seed in range(seeds):
        r = differential_evolution(f, [(lo, hi)] * (n_bins - 1), seed=seed,
                                   popsize=25, maxiter=3000, tol=1e-10,
                                   mutation=(0.5, 1.0), recombination=0.7,
                                   init="sobol", x0=np.clip(x0, lo, hi), polish=False)
        r2 = minimize(f, np.sort(r.x), method="Nelder-Mead",
                      options={"xatol": 0.05, "fatol": 1e-14, "maxiter": 20000})
        x = r2.x if r2.fun <= r.fun else r.x
        e, v = integer_polish(bf, x, min_width)
        runs.append((e, v))
    i = int(np.argmin([v for _, v in runs]))
    return runs[i][0], runs[i][1], runs


def brute_force(bf, n_bins, min_width, step):
    """N = 3 only: every pair of edges on a grid of spacing `step`, then
    integer polish of the best. Independent check of optimise()."""
    if n_bins != 3:
        raise ValueError("--brute is implemented for N = 3 only")
    lo, hi = bf.l_min - 1 + min_width, bf.l_max - min_width
    g = np.arange(lo, hi + 1, step)
    i, j = np.triu_indices(len(g), k=1)
    best_v, best_e = np.inf, None
    for c in range(0, len(i), 200000):
        E = np.stack([g[i[c:c + 200000]], g[j[c:c + 200000]]], axis=1)
        v = bf.objective_batch(E, min_width)
        k = int(np.argmin(v))
        if v[k] < best_v:
            best_v, best_e = float(v[k]), E[k]
    return integer_polish(bf, best_e, min_width)


def rounded_edges(bf, e, min_width, units=(10, 25, 50)):
    """Best edges restricted to multiples of 10, 25, 50 (local search from the
    optimum) and how much the objective worsens: what round numbers cost."""
    v0 = bf.objective(e, min_width)
    out = []
    for u in units:
        er, vr = integer_polish(bf, e, min_width, unit=u)
        out.append({"unit": u, "edges": [int(x) for x in er],
                    "max_sigma_diff": vr, "increase_pct": 100 * (vr / v0 - 1)})
    return out


# =============================================================================
# 5. Reporting
# =============================================================================

def describe(bf, e):
    C = bf.omega_cov(e)
    sd, rho = bf.diff_sigmas(C)
    return {"edges": [int(x) for x in e],
            "bins": [[int(lo) + 1, int(hi)] for lo, hi in
                     zip(bf.boundaries(e)[:-1], bf.boundaries(e)[1:])],
            "sigma_omega_m": np.sqrt(np.diag(C)).tolist(),
            "sigma_diff": sd.tolist(), "rho_adjacent": rho.tolist(),
            "max_sigma_diff": float(sd.max())}


def print_result(bf, n_bins, res, eq, runs, tol, fid_om):
    print("\n" + "=" * 72)
    print("N = %d bins: minimise the largest sigma(omega_m,i - omega_m,i+1)" % n_bins)
    print("=" * 72)
    print("  bin   l range        width   sigma(omega_m,i)")
    for i, ((lo, hi), s) in enumerate(zip(res["bins"], res["sigma_omega_m"])):
        print("  %2d   %4d - %-4d     %5d   %.5f" % (i, lo, hi, hi - lo + 1, s))
    if n_bins > 1:
        print("  pair   sigma(Delta)   rho")
        for k, (s, r) in enumerate(zip(res["sigma_diff"], res["rho_adjacent"])):
            print("  %d-%d    %.5f        %+.3f" % (k, k + 1, s, r))
        print("  largest sigma(Delta) = %.5f" % res["max_sigma_diff"])
    if tol:
        print("  edges rounded to multiples of 10 / 25 / 50 (re-optimised on that grid):")
        for r in tol:
            print("    %2d: %s   largest sigma(Delta) %.5f  (+%.1f%%)"
                  % (r["unit"], r["edges"], r["max_sigma_diff"], r["increase_pct"]))
    if len(runs) > 1:
        distinct = {tuple(int(x) for x in e) for e, _ in runs}
        if len(distinct) == 1:
            print("  all %d optimiser runs (different seeds) end on the same edges" % len(runs))
        else:
            print("  WARNING: optimiser runs ended on different edges:")
            for e, v in runs:
                print("    %s  max sigma(Delta) = %.6f" % (list(map(int, e)), v))
    if eq is not None:
        print("  for comparison, equal fixed-parameter information edges %s:"
              % eq["edges"])
        print("    sigma(Delta) = %s, largest = %.5f"
              % (", ".join("%.5f" % s for s in eq["sigma_diff"]), eq["max_sigma_diff"]))
    print("  .param lines:")
    for k, e in enumerate(res["edges"] + [bf.l_max]):
        print("    data.parameters['l_switch%d'] = [%d, %d, %d, 0, 1, 'cosmo']"
              % (k + 1, e, e, e))
    for i, s in enumerate(res["sigma_omega_m"]):
        print("    data.parameters['omega_m%d']  = [%.2f, 0.05, 0.25, %.4f, 1, 'cosmo']"
              % (i, fid_om, s))


def write_covmat(path, bf, edges):
    """Inverse Fisher in MontePython covmat format, physical units
    (same format as models/two_bins_fisher.covmat)."""
    C = np.linalg.inv(bf.full_fisher(edges))
    names = PARAMS[:N_THETA] + ["omega_m%d" % i for i in range(len(edges) + 1)]
    with open(path, "w") as f:
        f.write("# " + ", ".join(names) + "\n")
        for row in C:
            f.write(" ".join("%.6e" % v for v in row) + "\n")
    print("  wrote %s" % path)


# =============================================================================

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n_bins", type=int, nargs="+", default=[3],
                    help="number(s) of bins, e.g. 3 or 2 3 4 5")
    ap.add_argument("--edges", type=int, nargs="+",
                    help="evaluate these edges (l_switch1 ..) instead of optimising")
    ap.add_argument("--min_width", type=int, default=20,
                    help="smallest allowed bin, in multipoles (default 20)")
    ap.add_argument("--seeds", type=int, default=3,
                    help="independent optimiser runs per N (default 3)")
    ap.add_argument("--brute", type=int, default=0, metavar="STEP",
                    help="N=3 only: also brute-force all edge pairs on a grid of this step")
    ap.add_argument("--tau_prior", type=float, default=0,
                    help="Gaussian prior sigma on tau_reio (default: none, as in MontePython)")
    # fiducial (defaults: two_bins.param means, omega_m = 0.15)
    ap.add_argument("--omega_b", type=float, default=0.0224)
    ap.add_argument("--h", type=float, default=0.674)
    ap.add_argument("--tau_reio", type=float, default=0.054)
    ap.add_argument("--n_s", type=float, default=0.965)
    ap.add_argument("--lnAs", type=float, default=3.044)
    ap.add_argument("--omega_m", type=float, default=0.15)
    # likelihood (defaults: fake_planck_realistic with l_max = 2508)
    ap.add_argument("--l_min", type=int, default=2)
    ap.add_argument("--l_max", type=int, default=2508)
    ap.add_argument("--f_sky", type=float, default=0.57)
    ap.add_argument("--noise_file", default=NOISE_FILE)
    # derivatives
    ap.add_argument("--deriv_file", default=DERIV_FILE)
    ap.add_argument("--recompute", action="store_true", help="ignore the derivative cache")
    ap.add_argument("--step_scale", type=float, default=1.0,
                    help="multiply all derivative steps (stability check: 0.5)")
    # output
    ap.add_argument("--out", help="also save the results as JSON here")
    ap.add_argument("--covmat_out",
                    help="write the inverse Fisher (MontePython covmat format) for the "
                         "optimal edges of the LAST N here")
    a = ap.parse_args()

    fid = {"omega_b": a.omega_b, "h": a.h, "tau_reio": a.tau_reio, "n_s": a.n_s,
           "ln10^{10}A_s": a.lnAs, "omega_m": a.omega_m}
    centre, deriv = load_or_compute_derivatives(a, fid)
    noise = read_noise(a.noise_file, a.l_min, a.l_max)
    Fl = per_ell_fisher(centre, deriv, noise, a.l_min, a.l_max, a.f_sky)
    bf = BinnedFisher(Fl, a.l_min, a.l_max,
                      {"tau_reio": a.tau_prior} if a.tau_prior else None)

    results = {"fiducial": fid, "l_min": a.l_min, "l_max": a.l_max, "f_sky": a.f_sky,
               "min_width": a.min_width, "objective": "max_k sigma(omega_m,k - omega_m,k+1)",
               "results": {}}

    if a.edges:
        e = np.array(sorted(a.edges))
        res = describe(bf, e)
        print_result(bf, len(e) + 1, res, None, [], None, a.omega_m)
        results["results"]["given"] = res
        last_edges = e
    else:
        for n in a.n_bins:
            if n < 2:
                sys.exit("--n_bins must be >= 2")
            t0 = time.time()
            e, v, runs = optimise(bf, n, a.min_width, a.seeds)
            res = describe(bf, e)
            eq = describe(bf, equal_information_edges(bf, n))
            tol = rounded_edges(bf, e, a.min_width)
            print_result(bf, n, res, eq, runs, tol, a.omega_m)
            if a.brute and n != 3:
                print("  (--brute skipped: implemented for N = 3 only)")
            elif a.brute:
                eb, vb = brute_force(bf, n, a.min_width, a.brute)
                print("  brute force (grid step %d + polish): edges %s, largest sigma(Delta) %.6f %s"
                      % (a.brute, list(map(int, eb)), vb,
                         "(same)" if list(eb) == list(e) else "(DIFFERENT)"))
            print("  (%.1f s)" % (time.time() - t0))
            res["equal_information"] = eq
            res["rounded"] = tol
            results["results"][str(n)] = res
            last_edges = e

    if a.out:
        with open(a.out, "w") as f:
            json.dump(results, f, indent=1)
        print("\nwrote %s" % a.out)
    if a.covmat_out:
        write_covmat(a.covmat_out, bf, last_edges)


if __name__ == "__main__":
    main()
