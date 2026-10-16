import numpy as np, sys, json; sys.path.insert(0, '.')
from style import *
from matplotlib.patches import Rectangle
from fisherlib import linear_fit
S = json.load(open("summary.json"))
FIDS = {"lcdm": [(2, 2508, 0.15)], "2bins": [(2, 1250, 0.14), (1251, 2508, 0.16)],
        "3bins": [(2, 800, 0.14), (801, 1600, 0.16), (1601, 2508, 0.15)]}
EDGES = {"one_bin": [], "two_bins": [968], "three_bins": [871, 1230]}
ROWLAB = {"one_bin": "1 bin\n(ΛCDM)", "two_bins": "2 bins\n968", "three_bins": "3 bins\n871, 1230"}
def figure(model, ylim, name, width=5.8, height=3.6):
    fig, axes = plt.subplots(3, 1, figsize=(width, height), sharex=True, sharey=True, gridspec_kw={"hspace": 0.08})
    for ax, f in zip(axes, ["one_bin", "two_bins", "three_bins"]):
        r = S[f"{model}|{f}"]; c = FITCOL[f]
        p_lin, _ = linear_fit(FIDS[model], EDGES[f])
        for i, (lo, hi) in enumerate(r["bins"]):
            p = r["params"][f"omega_m{i}"]
            ax.add_patch(Rectangle((lo, p["lo95"]), hi - lo, p["hi95"] - p["lo95"], color=c, alpha=0.18, lw=0))
            ax.add_patch(Rectangle((lo, p["lo68"]), hi - lo, p["hi68"] - p["lo68"], color=c, alpha=0.45, lw=0))
            ax.plot([lo, hi], [p["mean"]] * 2, color=c, lw=1.6, solid_capstyle="butt")
            ax.plot([lo, hi], [0.15 + p_lin[5 + i]] * 2, color=INK, lw=0.9, ls=(0, (1, 1.5)))
            if i > 0: ax.axvline(lo - 0.5, color=c, lw=0.6, alpha=0.7)
        for j, (lo, hi, v) in enumerate(FIDS[model]):
            ax.plot([lo, hi], [v, v], color=INK, ls="--", lw=1.2)
        ax.text(1.01, 0.5, ROWLAB[f], transform=ax.transAxes, fontsize=7.5, va="center", color=c)
        ax.set_ylim(*ylim); ax.set_xlim(0, 2508)
        chi2 = r["chi2_min"]
        txt = r"$\chi^2_\mathrm{min}=%.0f$" % chi2 if chi2 > 1 else r"$\chi^2_\mathrm{min}=%.1f$" % chi2
        if "lcdm_test" in r:
            t = r["lcdm_test"]; txt += r"  |  all $\omega_{m,i}$ equal: %.1f$\sigma$" % t["nsigma"]
        ax.text(0.99, 0.06, txt, transform=ax.transAxes, ha="right", fontsize=7, color=INK,
                bbox=dict(fc="white", ec="none", alpha=0.8, pad=1))
    axes[1].set_ylabel(r"$\omega_m$")
    axes[-1].set_xlabel(r"multipole $\ell$")
    from matplotlib.lines import Line2D
    h = [Line2D([], [], color=INK, ls="--", lw=1.2), Rectangle((0, 0), 1, 1, color=MUTED, alpha=0.45),
         Rectangle((0, 0), 1, 1, color=MUTED, alpha=0.18), Line2D([], [], color=INK, lw=0.9, ls=(0, (1, 1.5)))]
    fig.legend(h, ["mock (truth)", "68%", "95%", "linear Fisher prediction"], loc="upper center", ncol=4,
               bbox_to_anchor=(0.5, 0.99), fontsize=7.5)
    fig.subplots_adjust(top=0.9)
    save(fig, name)
figure("lcdm", (0.1455, 0.1545), "omega_ell_lcdm")
figure("2bins", (0.134, 0.166), "omega_ell_2bins")
figure("3bins", (0.132, 0.166), "omega_ell_3bins")
