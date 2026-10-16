import numpy as np, sys, json, pickle; sys.path.insert(0, '.')
from style import *
from runs import load, run_dir
from getdist import plots
TH = ["omega_b", "h", "tau_reio", "n_s", "ln10^{10}A_s"]
THL = [r"$\omega_b$", r"$h$", r"$\tau_\mathrm{reio}$", r"$n_s$", r"$\ln(10^{10}A_s)$"]
FID = {"omega_b": 0.0224, "h": 0.674, "tau_reio": 0.054, "n_s": 0.965, "ln10^{10}A_s": 3.044}
FITS = ["one_bin", "two_bins", "three_bins"]
S = {}
for m in ["lcdm", "2bins", "3bins"]:
    for f in FITS:
        S[(m, f)] = load(run_dir(m, f))
S[("class", "one_bin")] = load(run_dir("lcdm", "one_bin", "with_class"))
# ---- theta posteriors: rows = mocks, cols = params
ROWT = {"lcdm": "ΛCDM mock", "2bins": "2-bin mock", "3bins": "3-bin mock"}
fig, ax = plt.subplots(3, 5, figsize=(10.0, 4.6), gridspec_kw={"hspace": 0.55, "wspace": 0.12})
for i, m in enumerate(["lcdm", "2bins", "3bins"]):
    for j, p in enumerate(TH):
        a = ax[i, j]
        for f in FITS:
            d = S[(m, f)].get1DDensity(p)
            x = np.linspace(d.x.min(), d.x.max(), 400)
            y = d.Prob(x); y = y / y.max()
            a.plot(x, y, color=FITCOL[f], lw=1.5 if f != "one_bin" else 1.8, label=FITLAB[f] if (i == 0 and j == 0) else None,
                   ls="-" if f != "three_bins" else (0, (5, 1.5)))
        a.axvline(FID[p], color=INK, ls="--", lw=0.9, label="truth" if (i == 0 and j == 0) else None)
        a.set_yticks([]); a.set_ylim(0, 1.08); a.grid(False)
        a.tick_params(axis="x", labelsize=6.5)
        a.xaxis.set_major_locator(plt.MaxNLocator(3, prune="both"))
        if i == 2 or True: a.set_xlabel(THL[j], fontsize=8.5, labelpad=1)
        if j == 0: a.set_ylabel(ROWT[m], fontsize=8.5)
fig.legend(loc="upper center", ncol=4, bbox_to_anchor=(0.5, 1.0), fontsize=7.5)
fig.subplots_adjust(top=0.9)
save(fig, "theta_posteriors")
# ---- shifts table data (in sigma of the 1-bin LCDM-mock fit) for text
shift = {}
for m in ["lcdm", "2bins", "3bins"]:
    for f in FITS:
        ms = S[(m, f)].getMargeStats()
        shift[f"{m}|{f}"] = {p: (ms.parWithName(p).mean - FID[p]) / ms.parWithName(p).err for p in TH}
json.dump(shift, open("theta_pulls.json", "w"), indent=1)
for k, v in shift.items(): print(k, " ".join(f"{p}:{x:+.2f}" for p, x in v.items()))
plt.rcParams["axes.grid"] = False
# ---- omega_m bin triangle (3-bin fit), LCDM mock vs 2-bin mock
OM = ["omega_m0", "omega_m1", "omega_m2"]
g = plots.get_subplot_plotter(width_inch=4.3)
g.settings.axes_fontsize = 8; g.settings.axes_labelsize = 10; g.settings.legend_fontsize = 8
g.settings.figure_legend_frame = False
g.triangle_plot([S[("lcdm", "three_bins")], S[("2bins", "three_bins")]], OM, filled=True,
                contour_colors=[MUTED, C3], legend_labels=["ΛCDM mock", "2-bin mock (step at 1250)"])
for i in range(1, 3):
    for j in range(i):
        a = g.subplots[i, j]
        lo = 0.135; hi = 0.165
        a.plot([lo, hi], [lo, hi], color=INK, ls=":", lw=0.9)
g.export("/tmp/claude-0/-home-claude/292e6737-93c7-5f4e-80bb-a872aa2b0bd3/scratchpad/deck/figs/omega_triangle.pdf")
g.export("/tmp/claude-0/-home-claude/292e6737-93c7-5f4e-80bb-a872aa2b0bd3/scratchpad/deck/figs/omega_triangle.png", dpi=170)
plt.rcParams["axes.grid"] = True
# ---- Delta 1D posteriors: 2-bin fit and 3-bin fit, on LCDM and 2-bin mocks
fig, ax = plt.subplots(1, 3, figsize=(7.4, 2.2), gridspec_kw={"wspace": 0.15})
items = [("two_bins", "domega_m1", r"$\omega_{m,1}-\omega_{m,0}$ (2 bins)"),
         ("three_bins", "domega_m1", r"$\omega_{m,1}-\omega_{m,0}$ (3 bins)"),
         ("three_bins", "domega_m2_1", r"$\omega_{m,2}-\omega_{m,1}$ (3 bins)")]
for a, (f, p, lab) in zip(ax, items):
    for m, c, ls in [("lcdm", MUTED, "-"), ("2bins", FITCOL[f], "-"), ("3bins", FITCOL[f], (0, (3, 1.5)))]:
        d = S[(m, f)].get1DDensity(p); x = np.linspace(d.x.min(), d.x.max(), 400); y = d.Prob(x); y /= y.max()
        a.plot(x * 1e3, y, color=c, ls=ls, lw=1.6, label={"lcdm": "ΛCDM mock", "2bins": "2-bin mock", "3bins": "3-bin mock"}[m])
    a.axvline(0, color=INK, lw=0.8, ls="--")
    a.set_yticks([]); a.set_xlabel(lab + r" $[10^{-3}]$", fontsize=8); a.grid(False); a.set_ylim(0, 1.1)
from matplotlib.lines import Line2D
ax[0].legend([Line2D([], [], color=MUTED, lw=1.6), Line2D([], [], color=INK, lw=1.6), Line2D([], [], color=INK, lw=1.6, ls=(0, (3, 1.5)))],
             ["ΛCDM mock", "2-bin mock", "3-bin mock"], fontsize=6.8, loc="upper left", bbox_to_anchor=(0.32, 1.0))
save(fig, "delta_posteriors")
# ---- correlation matrix (3-bin fit, LCDM mock)
s = S[("lcdm", "three_bins")]
names = s.getParamNames().list()
P = TH + OM
idx = [names.index(n) for n in P]
corr = s.getCorrelationMatrix()[np.ix_(idx, idx)]
fig, a = plt.subplots(figsize=(4.0, 3.4))
im = a.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
lab = THL + [r"$\omega_{m,0}$", r"$\omega_{m,1}$", r"$\omega_{m,2}$"]
a.set_xticks(range(len(P)), lab, rotation=45, ha="right", fontsize=7.5); a.set_yticks(range(len(P)), lab, fontsize=7.5)
for i in range(len(P)):
    for j in range(len(P)):
        a.text(j, i, f"{corr[i, j]:.2f}", ha="center", va="center", fontsize=6, color="white" if abs(corr[i, j]) > 0.6 else INK)
a.grid(False)
cb = fig.colorbar(im, ax=a, shrink=0.8); cb.ax.tick_params(labelsize=7)
save(fig, "correlation_3bins")
print("corr omega bins:", np.round(corr[5:, 5:], 3)); print("corr h-om:", np.round(corr[1, 5:], 3), "n_s-om:", np.round(corr[3, 5:], 3))
plt.rcParams["axes.grid"] = False
# ---- benchmark posteriors: CLASS vs CosmoPower, LCDM mock, 1 bin (each pipeline with its own mock)
g = plots.get_subplot_plotter(width_inch=5.0)
g.settings.axes_fontsize = 7; g.settings.axes_labelsize = 9; g.settings.legend_fontsize = 8; g.settings.figure_legend_frame = False
g.triangle_plot([S[("class", "one_bin")], S[("lcdm", "one_bin")]], TH + ["omega_m0"], filled=[True, False],
                contour_colors=["#9a9993", CPC], contour_lws=[1, 1.4], legend_labels=["CLASS pipeline (own mock)", "CosmoPower pipeline (own mock)"])
g.export("/tmp/claude-0/-home-claude/292e6737-93c7-5f4e-80bb-a872aa2b0bd3/scratchpad/deck/figs/bench_posteriors.pdf")
g.export("/tmp/claude-0/-home-claude/292e6737-93c7-5f4e-80bb-a872aa2b0bd3/scratchpad/deck/figs/bench_posteriors.png", dpi=170)
a, b = S[("class", "one_bin")].getMargeStats(), S[("lcdm", "one_bin")].getMargeStats()
bench = {p: dict(class_mean=a.parWithName(p).mean, class_err=a.parWithName(p).err, cp_mean=b.parWithName(p).mean, cp_err=b.parWithName(p).err,
                 shift_sigma=(b.parWithName(p).mean - a.parWithName(p).mean) / a.parWithName(p).err, err_ratio=b.parWithName(p).err / a.parWithName(p).err) for p in TH + ["omega_m0"]}
json.dump(bench, open("bench_mcmc.json", "w"), indent=1)
for p, v in bench.items(): print(p, "shift %.3f sigma, err ratio %.3f" % (v["shift_sigma"], v["err_ratio"]))
