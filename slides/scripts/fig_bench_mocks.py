"""CLASS vs CosmoPower on both mocks (LCDM, one bin): 1D marginals, triangle plots, numbers.

Run from anywhere:  python slides/scripts/fig_bench_mocks.py
U = omega_m root. Same chain conventions as benchmark_class_vs_cosmopower.ipynb (burn-in 0.3).
Writes figs/bench_mocks_1d.pdf, figs/bench_triangle_{class,cosmopower}_mock.pdf, data/bench_mocks.json.
"""
import ast, glob, json, os, re, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import style
from style import *
FIG = style.FIG = str(Path(__file__).resolve().parents[1] / "figs") + "/"   # slides/figs, whatever style.py says
from getdist import loadMCSamples, plots

U = Path(os.environ.get("OMEGA_M_ROOT", Path(__file__).resolve().parents[2]))
DATA = Path(__file__).resolve().parents[1] / "data"
RUNS = {("CLASS mock", "CLASS"): "with_class/chains/model_lcdm__param_one_bin",
        ("CLASS mock", "CosmoPower"): "with_cosmopower/chains/model_lcdm_from_class__param_one_bin",
        ("CosmoPower mock", "CLASS"): "with_class/chains/model_lcdm_from_cosmopower__param_one_bin",
        ("CosmoPower mock", "CosmoPower"): "with_cosmopower/chains/model_lcdm__param_one_bin"}
TRUTH = {"omega_b": 0.0224, "h": 0.674, "tau_reio": 0.054, "n_s": 0.965, "ln10^{10}A_s": 3.044, "omega_m0": 0.15}
TEX = {"omega_b": r"\omega_b", "h": "h", "tau_reio": r"\tau", "n_s": "n_s", "ln10^{10}A_s": r"\ln(10^{10}A_s)",
       "omega_m0": r"\omega_m"}
COL = {"CLASS": CLASSC, "CosmoPower": CPC}
BURN = 0.3
TICKS = {"omega_b": [0.0222, 0.0226], "omega_m0": [0.147, 0.150, 0.153], "ln10^{10}A_s": [3.02, 3.04, 3.06]}


def load(d):
    d = U / d
    roots = [str(f)[:-len(".paramnames")] for f in d.glob("*.paramnames")]
    root = max(roots, key=lambda r: len(glob.glob(glob.escape(r) + "_[0-9]*.txt")))
    s = loadMCSamples(root, settings={"ignore_rows": BURN}, no_cache=True)
    logp = {n: ast.literal_eval(r) for n, r in re.findall(
        r"^data\.parameters\[['\"](.+?)['\"]\]\s*=\s*(\[.*?\])", (d / "log.param").read_text(), re.M)}
    ranges = {}
    for i, n in enumerate(s.getParamNames().list()):
        if n in logp:
            _, lo, hi, _, sc, role = logp[n]
            s.samples[:, i] *= sc
            if role != "derived":
                ranges[n] = (None if lo in (None, -1) else lo * sc, None if hi in (None, -1) else hi * sc)
    s.setRanges(ranges)
    for p in s.paramNames.names:
        p.label = TEX.get(p.name, p.label)
    s.updateBaseStatistics()
    return s


S = {k: load(v) for k, v in RUNS.items()}
P = list(TRUTH)
mocks = ["CLASS mock", "CosmoPower mock"]

# ---- numbers
out = {}
for mock in mocks:
    out[mock] = {}
    for n in P:
        st = {m: S[(mock, m)].getMargeStats().parWithName(n) for m in ("CLASS", "CosmoPower")}
        c, p = st["CLASS"], st["CosmoPower"]
        out[mock][n] = dict(class_mean=c.mean, class_err=c.err, cp_mean=p.mean, cp_err=p.err,
                            shift_sigma=(p.mean - c.mean) / c.err, err_ratio=p.err / c.err,
                            class_pull=(c.mean - TRUTH[n]) / c.err, cp_pull=(p.mean - TRUTH[n]) / p.err)
fisher = json.load(open(DATA / "emubias.json"))["one_bin"]["shift_sigma"]
out["fisher_forecast_shift_sigma"] = dict(zip(P, fisher))
out["chains"] = {f"{m} | {k}": v for (m, k), v in RUNS.items()}
json.dump(out, open(DATA / "bench_mocks.json", "w"), indent=1)
for mock in mocks:
    print(mock)
    for n in P:
        r = out[mock][n]
        print(f"  {n:<13} shift {r['shift_sigma']:+.2f}  errratio {r['err_ratio']:.2f}  pulls CLASS {r['class_pull']:+.2f} CP {r['cp_pull']:+.2f}"
              f"   Fisher {out['fisher_forecast_shift_sigma'][n]:+.2f}")

# ---- 1D marginals: rows = mocks, columns = parameters
fig, ax = plt.subplots(2, 6, figsize=(10.2, 3.6), gridspec_kw={"hspace": 0.95, "wspace": 0.12})
for i, mock in enumerate(mocks):
    for j, n in enumerate(P):
        a = ax[i, j]
        for m in ("CLASS", "CosmoPower"):
            d = S[(mock, m)].get1DDensity(n)
            x = np.linspace(d.x[0], d.x[-1], 300); y = d.Prob(x); y = y / y.max()
            if m == "CLASS":
                a.fill_between(x, y, color=COL[m], alpha=0.15, lw=0); a.plot(x, y, color=COL[m], lw=1.2)
            else:
                a.plot(x, y, color=COL[m], lw=1.6)
        a.axvline(TRUTH[n], color=MUTED, ls="--", lw=0.8)
        a.set_yticks([]); a.set_ylim(0, 1.08); a.grid(False)
        a.spines["left"].set_visible(False)
        a.tick_params(axis="x", labelsize=7)
        a.locator_params(axis="x", nbins=3)
        if i == 1:
            a.set_xlabel(rf"${TEX[n]}$")
        r = out[mock][n]
        a.set_title(rf"$\Delta={r['shift_sigma']:+.2f}\sigma$", fontsize=7.5, color=CPC, loc="center")
    ax[i, 0].text(-0.08, 1.32, f"{mock} (fake data made with {mock.split()[0]})", transform=ax[i, 0].transAxes,
                  fontsize=9, fontweight="bold", ha="left")
for j, n in enumerate(P):     # same x range in both rows: input value +- 4 sigma (CLASS model, CLASS mock)
    w = 4 * out["CLASS mock"][n]["class_err"]
    for i in (0, 1):
        ax[i, j].set_xlim(TRUTH[n] - w, TRUTH[n] + w)
        if n in TICKS:
            ax[i, j].set_xticks(TICKS[n])
from matplotlib.lines import Line2D
fig.legend([Line2D([], [], color=CLASSC, lw=1.2), Line2D([], [], color=CPC, lw=1.6), Line2D([], [], color=MUTED, ls="--", lw=0.8)],
           ["CLASS model", "CosmoPower model", r"input value     ($\Delta$ = CosmoPower $-$ CLASS mean, in $\sigma$)"], loc="upper right", ncol=3, fontsize=8, bbox_to_anchor=(0.91, 1.06))
save(fig, "bench_mocks_1d")

# ---- full triangles (backup)
for mock in mocks:
    g = plots.get_subplot_plotter(width_inch=7.5)
    g.settings.legend_fontsize = 11; g.settings.axes_labelsize = 12
    g.triangle_plot([S[(mock, "CLASS")], S[(mock, "CosmoPower")]], P, filled=[True, False],
                    contour_colors=[CLASSC, CPC], contour_lws=[1.0, 1.5],
                    legend_labels=["CLASS model", "CosmoPower model"], legend_loc="upper right",
                    markers=TRUTH, marker_args={"color": MUTED, "lw": 0.8, "ls": "--"})
    g.fig.text(0.60, 0.80, f"Fit to the {mock}\nΛCDM, one bin\ndashed: input values", fontsize=12, va="top")
    tag = mock.split()[0].lower()
    g.export(FIG + f"bench_triangle_{tag}_mock.pdf")
