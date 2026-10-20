"""Cross fits, old vs new emulator: CosmoPower model on the CLASS mock and CLASS model on the CosmoPower mock.

Row 1: CP_paper emulators (Spurio Mancini+22) vs CLASS with massless nu, linear lensing (with_cosmopower / with_class).
Row 2: lcdm v1 emulators (Bolliet+23, arXiv:2303.01591) vs CLASS with 1x0.06 eV + HMcode (with_cp_lcdm / with_class_mnu).
Writes figs/bench_mnu_cross.pdf and data/bench_mnu_cross.json. Run with OMEGA_M_ROOT = omega_m root.
"""
import ast, json, os, re, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import style
from style import *
from getdist import loadMCSamples
FIG = style.FIG = str(Path(__file__).resolve().parents[1] / "figs") + "/"
DATA = Path(__file__).resolve().parents[1] / "data"
U = Path(os.environ.get("OMEGA_M_ROOT", Path(__file__).resolve().parents[2]))
TRUTH = {"omega_b": 0.0224, "h": 0.674, "tau_reio": 0.054, "n_s": 0.965, "ln10^{10}A_s": 3.044, "omega_m0": 0.15}
TEX = {"omega_b": r"\omega_b", "h": "h", "tau_reio": r"\tau", "n_s": "n_s", "ln10^{10}A_s": r"\ln(10^{10}A_s)", "omega_m0": r"\omega_m"}
TICKS = {"omega_b": [0.0222, 0.0226], "omega_m0": [0.147, 0.150, 0.153], "ln10^{10}A_s": [3.02, 3.04, 3.06]}
ROWS = [("CP_paper emulator  vs  CLASS (massless $\\nu$, linear lensing)",
         "with_cosmopower/chains/model_lcdm_from_class__param_one_bin",
         "with_class/chains/model_lcdm_from_cosmopower__param_one_bin"),
        ("lcdm v1 emulator (Bolliet+23)  vs  CLASS ($m_\\nu=0.06$ eV, HMcode)",
         "with_cp_lcdm/chains/model_lcdm_from_class_mnu__param_one_bin",
         "with_class_mnu/chains/model_lcdm_from_cp_lcdm__param_one_bin")]


def load(d):
    d = U / d
    root = [str(f)[:-len(".paramnames")] for f in d.glob("*.paramnames")][0]
    s = loadMCSamples(root, settings={"ignore_rows": 0.3}, no_cache=True)
    logp = {n: ast.literal_eval(r) for n, r in re.findall(
        r"^data\.parameters\[['\"](.+?)['\"]\]\s*=\s*(\[.*?\])", (d / "log.param").read_text(), re.M)}
    for i, n in enumerate(s.getParamNames().list()):
        if n in logp:
            s.samples[:, i] *= logp[n][4]
    s.updateBaseStatistics()
    return s


fig, ax = plt.subplots(2, 6, figsize=(10.2, 3.7), gridspec_kw={"hspace": 1.0, "wspace": 0.12})
out = {}
for i, (title, cp_on_class, class_on_cp) in enumerate(ROWS):
    S = {"cp": load(cp_on_class), "class": load(class_on_cp)}
    out[title] = {}
    for j, n in enumerate(TRUTH):
        a = ax[i, j]
        pulls = {}
        for k, col, lw in (("class", CLASSC, 1.2), ("cp", CPC, 1.6)):
            st = S[k].getMargeStats().parWithName(n)
            pulls[k] = (st.mean - TRUTH[n]) / st.err
            d = S[k].get1DDensity(n)
            x = np.linspace(d.x[0], d.x[-1], 300); y = d.Prob(x); y = y / y.max()
            if k == "class":
                a.fill_between(x, y, color=col, alpha=0.15, lw=0)
            a.plot(x, y, color=col, lw=lw)
        out[title][n] = {"pull_cp_model_on_class_mock": pulls["cp"], "pull_class_model_on_cp_mock": pulls["class"]}
        w = 4 * S["class"].getMargeStats().parWithName(n).err
        a.set_xlim(TRUTH[n] - w, TRUTH[n] + w)
        if n in TICKS:
            a.set_xticks(TICKS[n])
        else:
            a.locator_params(axis="x", nbins=3)
        a.axvline(TRUTH[n], color=MUTED, ls="--", lw=0.8)
        a.set_yticks([]); a.set_ylim(0, 1.08); a.grid(False); a.spines["left"].set_visible(False)
        a.tick_params(axis="x", labelsize=7)
        a.set_title(rf"${pulls['cp']:+.2f}$ / ${pulls['class']:+.2f}$", fontsize=7.5, color=INK)
        if i == 1:
            a.set_xlabel(rf"${TEX[n]}$")
    ax[i, 0].text(-0.08, 1.38, title, transform=ax[i, 0].transAxes, fontsize=9, fontweight="bold", ha="left")
from matplotlib.lines import Line2D
fig.legend([Line2D([], [], color=CPC, lw=1.6), Line2D([], [], color=CLASSC, lw=1.2), Line2D([], [], color=MUTED, ls="--", lw=0.8)],
           ["emulator model, CLASS mock", "CLASS model, emulator mock", "input value     (titles: pulls in $\\sigma$, same order)"],
           loc="upper right", ncol=3, fontsize=8, bbox_to_anchor=(0.91, 1.07))
save(fig, "bench_mnu_cross")
json.dump(out, open(DATA / "bench_mnu_cross.json", "w"), indent=1)
for t, v in out.items():
    print(t); print("  max |pull|:", max(max(abs(x) for x in d.values()) for d in v.values()))
