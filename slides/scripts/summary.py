import sys, json, re, pickle
import numpy as np
from scipy import stats as sps
sys.path.insert(0, '.')
from runs import load, run_dir
out = {}
cache = {}
for model in ["lcdm", "2bins", "3bins"]:
    for param in ["one_bin", "two_bins", "three_bins"]:
        s = load(run_dir(model, param))
        ms = s.getMargeStats()
        r = {"bins": s.bins, "fid_steps": s.steps, "chi2_min": s.chi2_min, "acc": s.acc, "params": {}}
        for n in s.getParamNames().list():
            p = ms.parWithName(n)
            r["params"][n] = dict(mean=p.mean, err=p.err, lo68=p.limits[0].lower, hi68=p.limits[0].upper,
                                  lo95=p.limits[1].lower, hi95=p.limits[1].upper)
        dn = [n for n in s.getParamNames().list() if re.fullmatch(r"domega_m\d+", n)]
        if dn:
            X = np.column_stack([s.samples[:, s.index[n]] for n in dn]); w = s.weights
            mu = np.average(X, axis=0, weights=w); C = np.atleast_2d(np.cov(X.T, aweights=w))
            chi2 = float(mu @ np.linalg.solve(C, mu)); pte = sps.chi2.sf(chi2, len(dn))
            r["lcdm_test"] = dict(chi2=chi2, dof=len(dn), pte=pte, nsigma=float(sps.norm.isf(pte / 2)) if pte > 0 else float(np.sqrt(chi2)))
        # R-1 per param (weighted), as in the notebook
        def gr(col):
            means, vars_, ns = [], [], []
            for c in s.raw:
                c = c[int(0.3 * len(c)):]; w, x = c[:, 0], c[:, col]
                m = np.average(x, weights=w); means.append(m); vars_.append(np.average((x - m) ** 2, weights=w)); ns.append(w.sum())
            W = np.mean(vars_); B = np.var(means, ddof=1); n = np.mean(ns)
            return ((n - 1) / n * W + B) / W - 1
        names = [n for n in s.getParamNames().list() if not n.startswith("domega")]
        r["Rm1_max"] = max(gr(2 + i) for i in range(len(names)))
        out[f"{model}|{param}"] = r
        print(f"{model:6s} {param:10s} bins={s.bins} chi2_min={s.chi2_min:8.2f} acc={s.acc:.3f} R-1max={r['Rm1_max']:.4f}",
              " ".join(f"{n}={r['params'][n]['mean']:.5f}±{r['params'][n]['err']:.5f}" for n in s.oms),
              ("| LCDM test %.1f sigma (chi2 %.1f/%d)" % (r['lcdm_test']['nsigma'], r['lcdm_test']['chi2'], r['lcdm_test']['dof'])) if dn else "")
        print("        theta:", " ".join(f"{n}={r['params'][n]['mean']:.5g}±{r['params'][n]['err']:.2g}" for n in ["omega_b","h","tau_reio","n_s","ln10^{10}A_s"]))
s = load(run_dir("lcdm", "one_bin", "with_class"))
ms = s.getMargeStats()
out["class|lcdm|one_bin"] = {"chi2_min": s.chi2_min, "acc": s.acc, "params": {n: dict(mean=ms.parWithName(n).mean, err=ms.parWithName(n).err) for n in s.getParamNames().list()}}
print("CLASS lcdm one_bin chi2_min %.3f acc %.3f" % (s.chi2_min, s.acc), " ".join(f"{n}={ms.parWithName(n).mean:.5g}±{ms.parWithName(n).err:.2g}" for n in s.getParamNames().list()))
json.dump(out, open("summary.json", "w"), indent=1, default=float)
