"""Smooth omega_m(l) (poly) vs fixed omega_m (one bin), on the mAxiCLASS EDE mock and on the LCDM CLASS mock.

Sampler: with_cp_lcdm (CosmoPower lcdm v1 emulators); likelihood fake_planck_realistic (TT TE EE + lensing, l <= 2508).
  poly:    omega_m(l) = omega_m0 + b x + c x^2, x = (l - 400)/1000   (with_cp_lcdm/poly.param, dwm_a = 0)
  one_bin: omega_m(l) = omega_m0                                     (with_cp_lcdm/one_bin.param)
Writes figs/poly_omega_ell.pdf and data/poly_runs.json. Burn-in 0.3. Run with OMEGA_M_ROOT = omega_m root.
"""
import ast, glob, json, os, re, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import style
from style import *
from getdist import loadMCSamples
FIG = style.FIG = str(Path(__file__).resolve().parents[1] / "figs") + "/"
DATA = Path(__file__).resolve().parents[1] / "data"
U = Path(os.environ.get("OMEGA_M_ROOT", Path(__file__).resolve().parents[2]))
CH = "with_cp_lcdm/chains/"
MOCKS = [  # key, panel title, mock file, one_bin folder, poly folder
    ("ede", "mAxiCLASS mock: 2-axion EDE", "with_maxiclass/model/ede2ax_pdh0_mean.dat",
     CH + "model_ede2ax_pdh0_mean_from_maxiclass__param_one_bin", CH + "model_ede2ax_pdh0_mean_from_maxiclass__param_poly"),
    ("lcdm", "CLASS mock: $\\Lambda$CDM (null test)", "with_class_mnu/model/lcdm.dat",
     CH + "model_lcdm_from_class_mnu__param_one_bin", CH + "model_lcdm_from_class_mnu__param_poly"),
]
ELL = np.linspace(2, 2508, 300)
QS = [0.025, 0.16, 0.5, 0.84, 0.975]


def header(p):
    return {k: float(v) for k, v in re.findall(r"([\w^{}]+)\s*=\s*([-\d.eE+]+)", (U / p).open().readline())}


def load(d):
    d = U / d
    logp = {n: ast.literal_eval(r) for n, r in re.findall(
        r"^data\.parameters\[['\"](.+?)['\"]\]\s*=\s*(\[.*?\])", (d / "log.param").read_text(), re.M)}
    root = [str(f)[:-len(".paramnames")] for f in d.glob("*.paramnames")][0]
    raw = [np.loadtxt(f, ndmin=2) for f in sorted(glob.glob(glob.escape(root) + "_[0-9]*.txt")) if Path(f).stat().st_size]
    s = loadMCSamples(root, settings={"ignore_rows": 0.3}, no_cache=True)
    for i, n in enumerate(s.getParamNames().list()):
        if n in logp:
            s.samples[:, i] *= logp[n][4]
    s.updateBaseStatistics()
    return s, logp, raw


def gelman_rubin(raw, col):
    m, v, n = [], [], []
    for c in raw:
        c = c[int(0.3 * len(c)):]
        w, x = c[:, 0], c[:, col]
        mu = np.average(x, weights=w)
        m.append(mu); v.append(np.average((x - mu) ** 2, weights=w)); n.append(w.sum())
    W, B, nn = np.mean(v), np.var(m, ddof=1), np.mean(n)
    return ((nn - 1) / nn * W + B) / W - 1


def bands(s, logp):
    om0 = s.samples[:, s.index["omega_m0"]]
    om = np.repeat(om0[:, None], len(ELL), axis=1)
    if "l_c" in logp:
        x = (ELL - logp["l_c"][0]) / (logp["l_norm"][0] if "l_norm" in logp else 1000.0)
        for k, n in enumerate(["dwm_a", "dwm_b", "dwm_c"]):
            if n in s.index:
                om += np.outer(s.samples[:, s.index[n]], x ** k)
            elif n in logp:
                om += logp[n][0] * x ** k
    out = np.empty((len(QS), len(ELL)))
    for j in range(len(ELL)):
        i = np.argsort(om[:, j]); cw = np.cumsum(s.weights[i]) / s.weights.sum()
        out[:, j] = np.interp(QS, cw, om[i, j])
    return out


def summary(s, logp, raw, mock):
    ms = s.getMargeStats()
    allc = np.vstack(raw)
    r = {"chi2_min": float(2 * allc[:, 1].min()), "n_chains": len(raw),
         "steps": [float(c[:, 0].sum()) for c in raw], "acceptance": [len(c) / float(c[:, 0].sum()) for c in raw],
         "R-1": {}, "params": {}}
    names = s.getParamNames().list()
    for i, n in enumerate(names):
        p = ms.parWithName(n)
        t = 100 * mock["h"] if n == "H0" else mock.get(n)
        r["params"][n] = {"mean": p.mean, "sigma": p.err, "mock": t,
                          "pull": None if t is None else (p.mean - t) / p.err}
        if logp.get(n, [0, 0, 0, 0, 1, "derived"])[5] != "derived":
            r["R-1"][n] = float(gelman_rubin(raw, 2 + i))
    if "dwm_b" in names:
        X = np.column_stack([s.samples[:, s.index[n]] for n in ("dwm_b", "dwm_c")])
        mu = np.average(X, axis=0, weights=s.weights)
        C = np.cov(X.T, aweights=s.weights)
        r["bc_joint_chi2_vs_0"] = float(mu @ np.linalg.solve(C, mu))
        r["corr_bc"] = float(C[0, 1] / np.sqrt(C[0, 0] * C[1, 1]))
    return r


out = {}
fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.3), sharey=True)
for ax, (key, title, mock_file, fixed_dir, poly_dir) in zip(axes, MOCKS):
    mock = header(mock_file)
    out[key] = {"mock_file": mock_file, "mock": mock}
    for tag, d, col, lab in [("one_bin", fixed_dir, C1, "fixed $\\omega_m$ ($\\Lambda$CDM fit)"),
                             ("poly", poly_dir, C2, "$\\omega_m(\\ell)=\\omega_{m0}+b\\,x+c\\,x^2$")]:
        s, logp, raw = load(d)
        out[key][tag] = dict(folder=d, **summary(s, logp, raw, mock))
        b = bands(s, logp)
        ax.fill_between(ELL, b[0], b[4], color=col, alpha=0.13, lw=0)
        ax.fill_between(ELL, b[1], b[3], color=col, alpha=0.32, lw=0)
        ax.plot(ELL, b[2], color=col, lw=1.4, label=lab)
    ax.axhline(mock["omega_m0"], color=INK, ls="--", lw=1, label=f"mock $\\omega_m$ = {mock['omega_m0']:g}")
    dchi2 = out[key]["one_bin"]["chi2_min"] - out[key]["poly"]["chi2_min"]
    out[key]["delta_chi2_min"] = dchi2
    ax.set_title(f"{title}\n$\\Delta\\chi^2_{{\\min}}$ (fixed $-$ $\\ell$-dependent) = {dchi2:.2f}, 2 extra params")
    ax.set_xlabel("multipole $\\ell$")
    ax.set_xlim(ELL[0], ELL[-1])
axes[0].set_ylabel("$\\omega_m$")
axes[0].set_ylim(0.105, 0.19)
axes[0].legend(loc="upper left", fontsize=7.5)
fig.tight_layout()
save(fig, "poly_omega_ell")
json.dump(out, open(DATA / "poly_runs.json", "w"), indent=1)
for key in out:
    o = out[key]
    print(key, "chi2_min one_bin %.3f poly %.3f  delta %.3f" % (o["one_bin"]["chi2_min"], o["poly"]["chi2_min"], o["delta_chi2_min"]))
