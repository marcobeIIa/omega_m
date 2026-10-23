"""Deck 3 figures: EDE mocks fitted with binned omega_m and with the smooth omega_m(l) (poly) model.
Bands = weighted 2.5/16/50/84/97.5% quantiles of omega_m(l) (as analyse_poly.ipynb); prior = uniform draws within the log.param bounds.
Run with OMEGA_M_ROOT = omega_m root. Writes figs/ede2_bins, ede2_poly_prior, ede1_poly, ede1_bins_prior (+ .png)."""
import json, os, re, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import style
from style import *
from ede_runs import *
FIG = style.FIG = str(Path(__file__).resolve().parents[1] / "figs") + "/"
NUM = json.load(open(Path(__file__).resolve().parents[1] / "data" / "ede_runs.json"))
ELL = np.linspace(2, 2508, 400)
Q = [0.025, 0.16, 0.5, 0.84, 0.975]
rng = np.random.default_rng(0)
COL = {"one_bin": C1, "five_bins": C2, "ten_bins": C3, "poly": "#c0392b"}


def wquant(x, w):
    i = np.argsort(x); cw = np.cumsum(w[i]) / w.sum()
    return np.interp(Q, cw, x[i])


def poly_band(samples, index, weights, logp):
    x = (ELL - logp["l_c"][0]) / 1000.0
    om = samples[:, index["omega_m0"]][:, None] + np.outer(samples[:, index["dwm_b"]], x) + np.outer(samples[:, index["dwm_c"]], x ** 2)
    return np.array([wquant(om[:, j], weights) for j in range(len(ELL))]).T


def poly_prior(logp, n=50_000):
    names = ["omega_m0", "dwm_b", "dwm_c"]
    X = np.column_stack([rng.uniform(logp[k][1], logp[k][2], n) for k in names])
    return poly_band(X, {k: i for i, k in enumerate(names)}, np.ones(n), logp)


def bin_bands(s, logp):
    """per-bin quantiles, as step functions on ELL"""
    out = np.zeros((5, len(ELL)))
    for i, (lo, hi) in enumerate(bins(logp)):
        q = wquant(s.samples[:, s.index[f"omega_m{i}"]], s.weights)
        m = (ELL >= lo - 0.5) & (ELL <= hi + 0.5)
        out[:, m] = np.array(q)[:, None]
    return out


def band(ax, b, col, label, a=(0.15, 0.35), lw=1.4, step=False):
    ax.fill_between(ELL, b[0], b[4], color=col, alpha=a[0], lw=0)
    ax.fill_between(ELL, b[1], b[3], color=col, alpha=a[1], lw=0)
    ax.plot(ELL, b[2], color=col, lw=lw, label=label)


# ---------------- 1. 2-axion EDE mock: 1, 5, 10 bins
fig, ax = plt.subplots(figsize=(8.6, 3.5))
runs = [("one_bin", "1 bin"), ("five_bins", "5 bins"), ("ten_bins", "10 bins")]
for k, (kind, lab) in enumerate(runs):
    s, logp, root = load(EDE2 + kind)
    for i, (lo, hi) in enumerate(bins(logp)):
        q = wquant(s.samples[:, s.index[f"omega_m{i}"]], s.weights)
        x = 0.5 * (lo + hi) + (k - 1) * 30
        ax.plot([lo, hi], [q[2]] * 2, color=COL[kind], lw=1.0, alpha=0.6)
        ax.vlines(x, q[0], q[4], color=COL[kind], lw=1.1)
        ax.vlines(x, q[1], q[3], color=COL[kind], lw=4)
        ax.plot(x, q[2], "o", color=COL[kind], mec="w", ms=5, label=lab if i == 0 else None, zorder=3)
    if kind != "one_bin":
        t = NUM[EDE2 + kind]["lcdm_test"]
        lab = f"{lab}: all bins equal, $\\chi^2={t['chi2']:.2f}$ for {t['dof']} dof"
        ax.plot([], [], " ", label=lab)
ax.set_xlim(0, 2560); ax.set_ylim(0.136, 0.148)
ax.set_xlabel(r"multipole $\ell$"); ax.set_ylabel(r"$\omega_m$ per bin")
ax.annotate(r"mock $\omega_m=0.1563$ (off scale, above)", xy=(0.99, 0.97), xycoords="axes fraction", ha="right", va="top", fontsize=8, color=MUTED)
ax.legend(fontsize=7.5, ncol=5, loc="lower center", bbox_to_anchor=(0.5, 1.0), handlelength=1.2, columnspacing=1.0)
save(fig, "ede2_bins")

# ---------------- poly prior vs posterior (+ zoom vs fixed omega_m), for a mock
def poly_figure(pre, name, om_true):
    s, logp, _ = load(pre + "poly")
    post = poly_band(s.samples, s.index, s.weights, logp)
    prior = poly_prior(logp)
    sf, lf, _ = load(pre + "one_bin")
    fixed = np.repeat(np.array(wquant(sf.samples[:, sf.index["omega_m0"]], sf.weights))[:, None], len(ELL), 1)
    fig, (a, b) = plt.subplots(1, 2, figsize=(10, 3.5), gridspec_kw={"wspace": 0.22})
    band(a, post, COL["poly"], r"posterior $\omega_m(\ell)$")
    for k, ls in [(0, "--"), (4, "--"), (1, "-"), (3, "-")]:
        a.plot(ELL, prior[k], color=COL["poly"], ls=ls, lw=0.9)
    a.plot([], [], color=COL["poly"], lw=0.9, label="prior 68%"); a.plot([], [], color=COL["poly"], lw=0.9, ls="--", label="prior 95%")
    a.axhline(om_true, color=INK, ls=":", lw=1, label=rf"mock $\omega_m={om_true:g}$")
    a.set_title("prior vs posterior", loc="left")
    band(b, fixed, C2, r"fixed $\omega_m$ (\texttt{one\_bin})".replace(r"\texttt{one\_bin}", "one_bin"))
    band(b, post, COL["poly"], r"$\omega_m(\ell)$ (poly)")
    b.axhline(om_true, color=INK, ls=":", lw=1)
    lo = min(post[0].min(), fixed[0].min()); hi = max(post[4].max(), fixed[4].max(), om_true)
    b.set_ylim(lo - 0.002, hi + 0.003)
    b.set_title("posterior, zoomed", loc="left")
    for x in (a, b):
        x.set_xlim(2, 2508); x.set_xlabel(r"multipole $\ell$")
    a.set_ylabel(r"$\omega_m(\ell)$")
    a.legend(fontsize=7.5, loc="lower left"); b.legend(fontsize=7.5, loc="upper left")
    save(fig, name)
    return post, prior

p2, pr2 = poly_figure(EDE2, "ede2_poly_prior", NUM["mock_ede2ax"]["omega_m0"])

# ---------------- 3. 1-axion EDE: fixed vs poly (as analyse_poly figure 1)
s, logp, _ = load(EDE1 + "poly"); post1 = poly_band(s.samples, s.index, s.weights, logp)
sf, lf, _ = load(EDE1 + "one_bin")
fixed1 = np.repeat(np.array(wquant(sf.samples[:, sf.index["omega_m0"]], sf.weights))[:, None], len(ELL), 1)
fig, ax = plt.subplots(figsize=(7.6, 3.5))
band(ax, fixed1, C2, r"fixed $\omega_m$ ($\Lambda$CDM fit)")
band(ax, post1, COL["poly"], r"$\ell$-dependent $\omega_m(\ell)$ (poly)")
om1 = NUM["mock_ede1ax"]["omega_m0"]
ax.axhline(om1, color=INK, ls=":", lw=1, label=rf"mock $\omega_m={om1:g}$")
ax.set_xlim(2, 2508); ax.set_xlabel(r"multipole $\ell$"); ax.set_ylabel(r"$\omega_m(\ell)$")
ax.legend(fontsize=8, loc="upper left")
save(fig, "ede1_poly")

# ---------------- 4. 1-axion EDE, 10 bins: prior vs posterior (+ zoom vs fixed and poly)
s, logp, _ = load(EDE1 + "ten_bins")
pb = bin_bands(s, logp)
lo_, hi_ = logp["omega_m0"][1], logp["omega_m0"][2]
pri = np.array([lo_ + q * (hi_ - lo_) for q in Q])[:, None] * np.ones(len(ELL))
fig, (a, b) = plt.subplots(1, 2, figsize=(10, 3.5), gridspec_kw={"wspace": 0.22})
band(a, pb, C3, "posterior, 10 bins")
for k, ls in [(0, "--"), (4, "--"), (1, "-"), (3, "-")]:
    a.plot(ELL, pri[k], color=C3, ls=ls, lw=0.9)
a.plot([], [], color=C3, lw=0.9, label="prior 68% (each bin)"); a.plot([], [], color=C3, lw=0.9, ls="--", label="prior 95%")
a.axhline(om1, color=INK, ls=":", lw=1, label=rf"mock $\omega_m={om1:g}$")
for l0, _ in bins(logp)[1:]:
    a.axvline(l0 - 0.5, color=GRID, lw=0.6); b.axvline(l0 - 0.5, color=GRID, lw=0.6)
a.set_title("prior vs posterior", loc="left"); a.set_ylim(0.105, 0.215)
band(b, fixed1, C2, r"fixed $\omega_m$ (one_bin)", a=(0.12, 0.3))
band(b, pb, C3, "10 bins")
b.plot(ELL, post1[2], color=COL["poly"], lw=1.0, ls="--", label=r"poly $\omega_m(\ell)$ median")
b.axhline(om1, color=INK, ls=":", lw=1)
b.set_ylim(0.133, 0.152); b.set_title("posterior, zoomed", loc="left")
for x in (a, b):
    x.set_xlim(2, 2508); x.set_xlabel(r"multipole $\ell$")
a.set_ylabel(r"$\omega_m(\ell)$")
a.legend(fontsize=7.5, loc="center right", bbox_to_anchor=(1.0, 0.62)); b.legend(fontsize=7.5, loc="upper left")
save(fig, "ede1_bins_prior")

# numbers for the slides
x = (ELL - 400) / 1000
for nm, p, pr in [("ede2", p2, pr2)]:
    for l in (30, 400, 1000, 1500, 2500):
        j = np.argmin(abs(ELL - l))
        print(nm, l, "post 68% half-width", round((p[3][j] - p[1][j]) / 2, 4), "prior", round((pr[3][j] - pr[1][j]) / 2, 4))
for l in (30, 1000, 2000, 2500):
    j = np.argmin(abs(ELL - l)); print("ede1 poly", l, "median", round(post1[2][j], 4), "68%", round(post1[1][j], 4), round(post1[3][j], 4))
