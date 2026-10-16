"""
compare_class_cosmopower.py -- debug check #1

Real CLASS (class_public, lensed_cl) vs our CosmoPower-backed classy stand-in
(class/python/classy.py) for lensed TT, TE, EE at one LCDM point
(single omega_m bin, l_switch1 = 2508, so the stand-in is plain LCDM).

Both codes run in one process: real CLASS is imported as `classy`
(compiled extension), the stand-in is loaded from its file under the name
`classy_cp`.

    python compare_class_cosmopower.py \
        --standin /Users/bellamarco01/uni/omega_m/class/python/classy.py \
        --outdir  class_vs_cosmopower

Outputs (in --outdir):
    class_vs_cosmopower.png        spectra, difference, difference / cosmic variance
    class_vs_cosmopower_tables.md  selected multipoles + summary per ell range
    cl_{tt,te,ee}_comparison.csv   every multipole 2..2508
"""
import argparse
import importlib.util
import os

import numpy as np

# Planck-2018-like LCDM point (omega_m = omega_b + omega_cdm = 0.1424)
PARAMS = {
    "omega_b": 0.0224,
    "omega_cdm": 0.1200,
    "h": 0.674,
    "tau_reio": 0.054,
    "n_s": 0.965,
    "ln10^{10}A_s": 3.044,
}
LMAX = 2508
T_UK2 = (2.7255e6) ** 2
SPECS = ("tt", "te", "ee")
TABLE_ELLS = [2, 10, 30, 100, 220, 400, 540, 800, 1000, 1250, 1500, 2000, 2500]
ELL_RANGES = [(2, 29), (30, 799), (800, 1599), (1600, 2508)]


def run_class():
    # Make sure `import classy` finds the REAL (compiled) CLASS wrapper, not a
    # classy.py stand-in sitting next to this script or on PYTHONPATH.
    import sys
    here = os.path.dirname(os.path.abspath(__file__))
    sys.path[:] = [p for p in sys.path
                   if os.path.abspath(p or ".") != here
                   and not os.path.isfile(os.path.join(p or ".", "classy.py"))]
    import classy
    if hasattr(classy, "_EMU"):
        raise SystemExit("`import classy` resolved to the CosmoPower stand-in (%s); "
                         "remove it from PYTHONPATH for this script" % classy.__file__)
    print("real CLASS:", classy.__file__)
    Class = classy.Class
    c = Class()
    c.set(PARAMS)
    # l_max_scalars above LMAX so CLASS's lensed spectra are accurate up to LMAX
    c.set({"output": "tCl,pCl,lCl", "lensing": "yes", "l_max_scalars": 3000})
    c.compute()
    cl = c.lensed_cl(LMAX)
    return {k: np.array(cl[k][: LMAX + 1]) for k in SPECS}


def run_standin(path):
    spec = importlib.util.spec_from_file_location("classy_cp", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    c = mod.Class()
    p = {k: v for k, v in PARAMS.items() if k != "omega_cdm"}
    p.update(omega_m0=PARAMS["omega_b"] + PARAMS["omega_cdm"], l_switch1=LMAX,
             l_max_scalars=LMAX)
    c.set(p)
    c.compute(["lensing"])
    cl = c.lensed_cl(LMAX)
    return {k: np.array(cl[k]) for k in SPECS}


def cosmic_variance_sigma(cl, ell):
    """Full-sky Gaussian error on a single C_l (no noise, f_sky = 1)."""
    nu = 2 * ell + 1
    return {
        "tt": np.sqrt(2.0 / nu) * np.abs(cl["tt"]),
        "ee": np.sqrt(2.0 / nu) * np.abs(cl["ee"]),
        "te": np.sqrt((cl["tt"] * cl["ee"] + cl["te"] ** 2) / nu),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--standin", required=True, help="path to the CosmoPower classy.py")
    ap.add_argument("--outdir", default="class_vs_cosmopower")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    ell = np.arange(LMAX + 1)
    ref = run_class()
    cp = run_standin(args.standin)
    sig = cosmic_variance_sigma(ref, ell)
    dl = ell * (ell + 1) / (2 * np.pi) * T_UK2  # C_l (dimensionless) -> D_l [uK^2]

    rows = {}
    for k in SPECS:
        d_ref, d_cp = dl * ref[k], dl * cp[k]
        with np.errstate(divide="ignore", invalid="ignore"):
            rel = (cp[k] - ref[k]) / ref[k]
            ncv = (cp[k] - ref[k]) / sig[k]
        rows[k] = dict(d_ref=d_ref, d_cp=d_cp, diff=d_cp - d_ref, rel=rel, ncv=ncv)
        np.savetxt(
            os.path.join(args.outdir, "cl_%s_comparison.csv" % k),
            np.column_stack([ell, d_ref, d_cp, d_cp - d_ref, rel, ncv])[2:],
            delimiter=",", fmt=["%d"] + ["%.8e"] * 5,
            header="ell,Dl_CLASS_uK2,Dl_CosmoPower_uK2,Dl_diff_uK2,rel_diff,diff_over_cosmic_variance",
            comments="")

    # ---------------- tables ----------------
    lines = ["# CLASS vs CosmoPower stand-in -- lensed TT, TE, EE", "",
             "Cosmology: " + ", ".join("%s = %g" % kv for kv in PARAMS.items()) + ".",
             "D_l = l(l+1)C_l/2pi in uK^2 (T_cmb = 2.7255 K). Diff = CosmoPower - CLASS. "
             "sigma_CV = full-sky cosmic-variance error on a single C_l.", ""]
    lines += ["## Summary by multipole range", "",
              "| spectrum | ell range | median abs rel diff | max abs rel diff | median abs diff/sigma_CV | max abs diff/sigma_CV |",
              "|---|---|---|---|---|---|"]
    for k in SPECS:
        r = rows[k]
        for lo, hi in ELL_RANGES:
            s = slice(lo, hi + 1)
            rel = np.abs(r["rel"][s])
            if k == "te":  # relative diff is meaningless at TE zero crossings
                rel_txt = ("n/a", "n/a")
            else:
                rel_txt = ("%.2e" % np.median(rel), "%.2e" % rel.max())
            n = np.abs(r["ncv"][s])
            lines.append("| %s | %d-%d | %s | %s | %.3f | %.3f |"
                         % (k.upper(), lo, hi, rel_txt[0], rel_txt[1], np.median(n), n.max()))
    # ---- ell-alignment check: an indexing bug shows up as a best-fit shift of
    # exactly +-1, +-2, ... multipoles; physics differences give |d| << 1.
    lines += ["", "## ell-alignment check", "",
              "Fit emulator(l) = [smooth cubic distortion] x CLASS(l + d), scanning d in [-3, 3]. "
              "The smooth factor absorbs amplitude/tilt differences, so only the peak positions constrain d. "
              "An indexing bug gives |d| = 1, 2, ...; PASS requires |d| < 0.5 and d = 0 beating d = +-1.", "",
              "| spectrum | ell range | best d | rms at best d | rms at d=0 | rms at d=-1 / +1 | verdict |",
              "|---|---|---|---|---|---|---|"]
    ds = np.round(np.linspace(-3, 3, 601), 3)
    for k in SPECS:
        for lo, hi in [(30, 900), (30, 2400)]:
            L = np.arange(lo, hi)
            s = sig[k][L]
            x = (L - L.mean()) / L.std()
            rms = []
            for d in ds:
                ci = np.interp(L + d, ell, ref[k])
                A = np.vstack([ci * x**p for p in range(4)]).T / s[:, None]
                y = (cp[k][L] - ci) / s
                coef = np.linalg.lstsq(A, y, rcond=None)[0]
                rms.append(np.sqrt(np.mean((y - A @ coef) ** 2)))
            rms = np.array(rms)
            at = lambda v: rms[np.argmin(np.abs(ds - v))]
            best = ds[np.argmin(rms)]
            ok = abs(best) < 0.5 and at(0) < min(at(-1), at(1))
            lines.append("| %s | %d-%d | %+.2f | %.3f | %.3f | %.3f / %.3f | %s |"
                         % (k.upper(), lo, hi - 1, best, rms.min(), at(0), at(-1), at(1),
                            "PASS" if ok else "FAIL"))
            if not ok:
                print("WARNING: ell-alignment check FAILED for", k, lo, hi)

    for k in SPECS:
        r = rows[k]
        lines += ["", "## %s at selected multipoles" % k.upper(), "",
                  "| ell | D_l CLASS | D_l CosmoPower | diff | rel diff | diff / sigma_CV |",
                  "|---|---|---|---|---|---|"]
        for l in TABLE_ELLS:
            rel = "n/a" if k == "te" else "%+.2e" % r["rel"][l]
            lines.append("| %d | %.4g | %.4g | %+.3e | %s | %+.3f |"
                         % (l, r["d_ref"][l], r["d_cp"][l], r["diff"][l], rel, r["ncv"][l]))
    with open(os.path.join(args.outdir, "class_vs_cosmopower_tables.md"), "w") as f:
        f.write("\n".join(lines) + "\n")

    # ---------------- figure ----------------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    C_REF, C_CP, INK, MUTED, GRID = "#1a1a19", "#2a78d6", "#1a1a19", "#6b6a63", "#e5e4de"
    plt.rcParams.update({"font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                         "xtick.color": MUTED, "ytick.color": MUTED, "axes.grid": True,
                         "grid.color": GRID, "grid.linewidth": 0.6,
                         "axes.spines.top": False, "axes.spines.right": False})
    fig, ax = plt.subplots(3, 3, figsize=(13, 8.5), sharex=True,
                           gridspec_kw={"height_ratios": [2, 1, 1]})
    L = ell[2:]
    for j, k in enumerate(SPECS):
        r = rows[k]
        a = ax[0, j]
        a.plot(L, r["d_ref"][2:], color=C_REF, lw=2.4, alpha=0.35, label="CLASS v3.4.0 (class_public)")
        a.plot(L, r["d_cp"][2:], color=C_CP, lw=1.0, label="CosmoPower stand-in")
        a.set_title(r"$D_\ell^{%s}$ [$\mu$K$^2$]" % k.upper(), loc="left", color=INK)
        if k == "te":
            a.axhline(0, color=MUTED, lw=0.6)
        a = ax[1, j]
        a.plot(L, r["diff"][2:], color=C_CP, lw=0.8)
        a.axhline(0, color=MUTED, lw=0.6)
        a.set_title(r"$\Delta D_\ell$ [$\mu$K$^2$]  (CosmoPower $-$ CLASS)", loc="left",
                    color=INK, fontsize=8.5)
        a = ax[2, j]
        a.plot(L, r["ncv"][2:], color=C_CP, lw=0.8)
        a.axhline(0, color=MUTED, lw=0.6)
        for y in (-0.1, 0.1):
            a.axhline(y, color=MUTED, lw=0.6, ls=":")
        a.set_title(r"$\Delta C_\ell / \sigma_{\rm CV}$  (dotted: $\pm 0.1$)", loc="left",
                    color=INK, fontsize=8.5)
        a.set_xlabel(r"$\ell$")
    ax[0, 0].legend(frameon=False, loc="upper right")
    fig.suptitle("Lensed CMB spectra: CLASS vs CosmoPower stand-in, single LCDM point "
                 r"($\omega_m=0.1424$)", x=0.01, ha="left", fontsize=11, color=INK)
    fig.tight_layout()
    fig.savefig(os.path.join(args.outdir, "class_vs_cosmopower.png"), dpi=140)
    print("wrote", args.outdir)


if __name__ == "__main__":
    main()
