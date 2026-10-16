import numpy as np, sys; sys.path.insert(0, '.')
from style import *
from fisherlib import bf, Fl, LMAX
import json
fw = Fl[:, 5, 5]
l = np.arange(LMAX + 1)
fig, ax = plt.subplots(1, 3, figsize=(10.2, 3.2), gridspec_kw={"width_ratios": [1.05, 1, 1.05], "wspace": 0.38})
# (a) information density on omega_m (fixed other params), smoothed
k = 25; sm = np.convolve(fw, np.ones(k) / k, mode="same")
a = ax[0]
a.plot(l, sm / sm.max(), color=INK, lw=1.2)
a.fill_between(l, 0, sm / sm.max(), color=INK, alpha=0.08, lw=0)
for e in [968]:
    a.axvline(e, color=C2, lw=1.2, label="2 bins: 968")
for j, e in enumerate([871, 1230]):
    a.axvline(e, color=C3, lw=1.2, ls="--", label="3 bins: 871, 1230" if j == 0 else None)
a.set_xlabel(r"$\ell$"); a.set_ylabel(r"$F_{\omega_m\omega_m}(\ell)$, normalised")
a.set_title(r"(a) where the $\omega_m$ information sits", loc="left")
cum = np.cumsum(fw) / fw.sum()
a.legend(loc="upper right", fontsize=7, bbox_to_anchor=(1.03, 1.02), frameon=True, framealpha=0.9, edgecolor="none")
a.set_xlim(0, 2508); a.set_ylim(0, 1.1)
# (b) N=2: sigma(Delta) vs edge
E = np.arange(30, 2480)[:, None]
v = bf.objective_batch(E, 20)
b = ax[1]
b.plot(E[:, 0], v * 1e4, color=C2, lw=1.6)
i = np.argmin(v)
b.plot(E[i, 0], v[i] * 1e4, "o", color=C2, ms=6, mec="white", mew=1.5, zorder=3)
b.annotate(r"$\ell_s=968$" + "\n" + r"$\sigma(\Delta)=%.2f\times10^{-4}$" % (v[i] * 1e4), (E[i, 0], v[i] * 1e4), xytext=(120, 2.0), fontsize=8,
           arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.6))
b.axvline(1250, color=MUTED, ls=":", lw=0.8); b.text(1290, 4.2, "2-bin mock\nstep (1250)", fontsize=7, color=MUTED, va="top")
b.set_ylim(0, 16); b.set_xlim(0, 2508)
b.set_xlabel(r"edge $\ell_s$"); b.set_ylabel(r"$\sigma(\omega_{m,1}-\omega_{m,0})\ [10^{-4}]$")
b.set_title("(b) 2 bins: scan of the edge", loc="left")
# (c) N=3: max sigma(Delta) over (l1, l2)
g = np.arange(30, 2481, 10)
I, J = np.meshgrid(g, g, indexing="ij")
m = J > I + 20
Eg = np.stack([I[m], J[m]], axis=1)
vv = np.full(I.shape, np.nan); vv[m] = bf.objective_batch(Eg, 20)
c = ax[2]
lev = np.array([6.5, 7, 8, 10, 12, 15, 20])
im = c.contourf(I, J, vv * 1e4, levels=np.concatenate([[6.4], lev, [40]]), cmap="Blues_r", extend="max")
c.contour(I, J, vv * 1e4, levels=lev, colors="white", linewidths=0.4)
c.plot(871, 1230, "*", color=C3, ms=12, mec="white", mew=1.0, label="optimum 871, 1230")
c.plot(800, 1600, "X", color=INK, ms=7, mec="white", mew=0.8, label="mock steps 800, 1600")
c.set_xlabel(r"$\ell_{s,1}$"); c.set_ylabel(r"$\ell_{s,2}$"); c.grid(False)
c.set_xlim(0, 2508); c.set_ylim(0, 2508)
c.set_title(r"(c) 3 bins: max$_k\,\sigma(\Delta_k)\ [10^{-4}]$", loc="left")
cb = fig.colorbar(im, ax=c, shrink=0.85, pad=0.03); cb.ax.tick_params(labelsize=7)
c.legend(loc="lower right", fontsize=6.5, frameon=False)
save(fig, "binning_fisher")
print("min N=2", E[i], v[i]); k2 = np.nanargmin(vv); print("grid min N=3", I.flat[k2], J.flat[k2], np.nanmin(vv))
print("cum info at 968, 871, 1230:", cum[968], cum[871], cum[1230])
