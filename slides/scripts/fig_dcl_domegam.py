"""dC_l/domega_m at the benchmark point, and how much it changes for a 1-sigma shift of each other LCDM parameter.

Writes figs/dcl_domegam.pdf (the derivative) and figs/dcl_domegam_stability.pdf (absolute change, relative to the
derivative, relative to C_l). Input: derivs.npz from dcl_derivs_compute.py (CLASS 3.4.1, lensed, l = 2..2508).
"""
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import style
from style import *
from pathlib import Path
FIG = style.FIG = str(Path(__file__).resolve().parents[1] / "figs") + "/"
DATA = Path(__file__).resolve().parents[1] / "data"

z = np.load(DATA / "dcl_derivs.npz"); l = z["l"]; s = slice(2, None); ll = l[s].astype(float)
C0, D0 = z["C0"][:, s], z["D0"][:, s]                       # rows: TT, TE, EE, phiphi
Dl = ll * (ll + 1) / (2 * np.pi); Dpp = (ll * (ll + 1))**2 / (2 * np.pi)
# 1-sigma = 68% errors of the CLASS-model fit to the CLASS mock (data/bench_mocks.json); fraction of a 3% shift
SIG = {"h": 0.00550 / (0.03 * 0.674), "omega_b": 0.000149 / (0.03 * 0.0224), "A_s": 0.00847 / np.log(1.03),
       "n_s": 0.00358 / (0.03 * 0.965), "tau_reio": 0.00441 / (0.03 * 0.054)}
LAB = {"h": "$h$", "omega_b": r"$\omega_b$", "A_s": r"$A_s$", "n_s": r"$n_s$", "tau_reio": r"$\tau$"}
COLS = {"h": C2, "omega_b": C1, "A_s": C3, "n_s": "#8a5cc2", "tau_reio": MUTED}
dD = {n: SIG[n] * (z["Dp_" + n][:, s] - z["Dm_" + n][:, s]) / 2 for n in SIG}   # change of dC/domega_m, 1-sigma shift
TITLES = ["TT", "TE", "EE", r"$\phi\phi$"]
amp = [Dl, Dl, Dl, Dpp * 1e7]                                  # D_l units for plotting
norm = [C0[0], np.sqrt(C0[0] * C0[2]), C0[2], C0[3]]           # TE normalised by sqrt(TT EE)

def axes(nrow, h):
    fig, ax = plt.subplots(nrow, 4, figsize=(10.4, h), sharex=True, gridspec_kw={"hspace": 0.22, "wspace": 0.32})
    for a in ax.flat:
        a.set_xscale("log"); a.set_xlim(2, 2508); a.axhline(0, color=MUTED, lw=0.6)
    for j in range(4):
        ax[0, j].set_title(TITLES[j]); ax[-1, j].set_xlabel(r"$\ell$")
    return fig, ax

# ---- 1. the derivative itself
fig, ax = axes(2, 3.9)
for j in range(4):
    ax[0, j].plot(ll, amp[j] * D0[j], color=INK, lw=1.0)
    ax[1, j].plot(ll, D0[j] / norm[j], color=INK, lw=1.0)
ax[0, 0].set_ylabel(r"$\partial D_\ell/\partial\omega_m\ [\mu{\rm K}^2]$")
ax[0, 3].set_ylabel(r"$10^7\,\partial[\ell(\ell+1)]^2C_\ell^{\phi\phi}/2\pi\,/\partial\omega_m$", fontsize=7.5)
ax[1, 0].set_ylabel(r"$\partial C_\ell/\partial\omega_m\ /\ C_\ell$")
ax[1, 1].set_ylabel(r"$/\sqrt{C^{TT}_\ell C^{EE}_\ell}$", fontsize=8)
save(fig, "dcl_domegam")

# ---- 2. its change for a 1-sigma shift of each other parameter
fig, ax = axes(3, 4.9)
for n in SIG:
    for j in range(4):
        ax[0, j].plot(ll, amp[j] * dD[n][j], color=COLS[n], lw=0.9, label=LAB[n])
        rel = 100 * dD[n][j] / np.abs(D0[j])
        ax[1, j].plot(ll, rel, color=COLS[n], lw=0.9)
        ax[2, j].plot(ll, dD[n][j] / norm[j], color=COLS[n], lw=0.9)
for j in range(4):
    ax[1, j].set_ylim(-12, 12)
ax[0, 0].set_ylabel(r"$\delta(\partial D_\ell/\partial\omega_m)\ [\mu{\rm K}^2]$")
ax[1, 0].set_ylabel(r"$\delta(\partial C_\ell/\partial\omega_m)\,/\,|\partial C_\ell/\partial\omega_m|$ [%]", fontsize=8)
ax[2, 0].set_ylabel(r"$\delta(\partial C_\ell/\partial\omega_m)\,/\,C_\ell$")
ax[0, 0].legend(ncol=5, loc="lower left", bbox_to_anchor=(0.0, 1.22), fontsize=8, title=r"$1\sigma$ shift of:",
                title_fontsize=8, alignment="left")
save(fig, "dcl_domegam_stability")

# numbers for the slide: noise-weighted change per 3% (reproduces the table), and typical sizes
print("max |dlnC/domega_m| (l>=30):", [float(np.max(np.abs(D0[j] / norm[j])[28:])) for j in range(4)])
for n in SIG:
    r = [float(np.median(np.abs(dD[n][j] / norm[j])[28:])) for j in range(4)]
    print(n, "median |delta(dC/dom)/C| l>=30:", np.round(r, 3))
