import numpy as np, sys; sys.path.insert(0, '.')
from style import *
from fisherlib import centre, deriv
l = np.arange(2509).astype(float); s = slice(2, 2509)
tt, ee, te, dd = centre["tt"], centre["ee"], centre["te"], centre["dd"]
with np.errstate(divide="ignore", invalid="ignore"):
    dl = {"TT": deriv["tt"][5] / tt, "EE": deriv["ee"][5] / ee, "TE": deriv["te"][5] / np.sqrt(tt * ee), r"$\phi\phi$": deriv["dd"][5] / dd}
cols = {"TT": C2, "EE": C1, "TE": C3, r"$\phi\phi$": MUTED}
fig, ax = plt.subplots(1, 2, figsize=(7.4, 3.0), gridspec_kw={"wspace": 0.3, "width_ratios": [1.15, 1]})
a = ax[0]
for k, v in dl.items():
    a.plot(l[s], v[s], color=cols[k], lw=1.1 if k != r"$\phi\phi$" else 1.4, label=k + (r" (norm. $\sqrt{C^{TT}C^{EE}}$)" if k == "TE" else ""))
a.axhline(0, color=MUTED, lw=0.6)
a.set_ylim(-40, 40); a.set_xlim(0, 2508)
a.set_xlabel(r"$\ell$"); a.set_title(r"(a) $\partial\ln C_\ell/\partial\omega_m$", loc="left")
a.legend(fontsize=7, loc="lower left", ncol=2)
c = ax[1]
E = np.load("edges_class.npz")
T = (2.7255e6) ** 2
D = lambda om, k: l * (l + 1) / (2 * np.pi) * E[f"{k}_{om}"] * T
r = slice(1150, 1351)
binned = np.where(l <= 1250, D(0.14, "tt"), D(0.16, "tt"))
c.plot(l[r], D(0.14, "tt")[r], color=INK, lw=1.6, ls=(0,(4,2)), label=r"ΛCDM, $\omega_m=0.14$", zorder=3)
c.plot(l[r], D(0.16, "tt")[r], color=INK, lw=1.6, ls=(0,(1,1.5)), label=r"ΛCDM, $\omega_m=0.16$", zorder=3)
c.plot(l[r], binned[r], color=C2, lw=1.4, alpha=0.9, label="2-bin mock (step at 1250)")
c.axvline(1250.5, color=MUTED, lw=0.5)
c.set_xlabel(r"$\ell$"); c.set_ylabel(r"$D_\ell^{TT}\ [\mu\mathrm{K}^2]$")
c.set_title(r"(b) 2-bin mock edge: TT jumps $-10.6\%$", loc="left")
c.legend(fontsize=7, loc="upper right")
save(fig, "edges_derivatives")
