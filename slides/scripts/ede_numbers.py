import json, os, sys, re
import numpy as np
from scipy import stats as sps
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ede_runs import *

out = {}
for pre, kinds in [(EDE2, ["one_bin", "five_bins", "ten_bins", "poly"]), (EDE1, ["one_bin", "ten_bins", "poly"])]:
    for k in kinds:
        s, logp, root = load(pre + k)
        raw, steps, acc, chi2 = chain_info(root)
        nvar = sum(1 for n, v in logp.items() if v[3] != 0 and v[5] != "derived")
        ms = s.getMargeStats()
        r = dict(root=os.path.basename(root), steps=steps, acceptance=acc, chi2_min=chi2, maxR1=max_gr(raw, nvar))
        for n in ["omega_b", "h", "tau_reio", "n_s", "ln10^{10}A_s"] + [p for p in s.getParamNames().list() if re.fullmatch(r"omega_m\d+|dwm_[bc]", p)]:
            p = ms.parWithName(n); r[n] = [p.mean, p.err]
        om = [p for p in s.getParamNames().list() if re.fullmatch(r"omega_m\d+", p)]
        if len(om) > 1:   # LCDM test on Delta_i = omega_m_i - omega_m0
            X = np.column_stack([s.samples[:, s.index[n]] - s.samples[:, s.index["omega_m0"]] for n in om[1:]])
            mu = np.average(X, axis=0, weights=s.weights); C = np.atleast_2d(np.cov(X.T, aweights=s.weights))
            c2 = float(mu @ np.linalg.solve(C, mu)); pte = sps.chi2.sf(c2, len(mu))
            r["lcdm_test"] = dict(chi2=c2, dof=len(mu), pte=pte, nsigma=float(sps.norm.isf(pte / 2)))
            r["bins"] = bins(logp)
        if "dwm_b" in s.index:
            X = np.column_stack([s.samples[:, s.index[n]] for n in ["dwm_b", "dwm_c"]])
            mu = np.average(X, axis=0, weights=s.weights); C = np.cov(X.T, aweights=s.weights)
            c2 = float(mu @ np.linalg.solve(C, mu)); pte = sps.chi2.sf(c2, 2)
            r["bc_zero_test"] = dict(chi2=c2, pte=pte, nsigma=float(sps.norm.isf(pte / 2)), corr=float(C[0, 1] / np.sqrt(C[0, 0] * C[1, 1])))
        out[pre + k] = r
        print(f"{pre[6:12]} {k:<10} steps/chain {min(steps)}-{max(steps)} acc {acc:.3f} chi2min {chi2:.2f} maxR-1 {r['maxR1']:.4f}",
              {kk: v for kk, v in r.items() if kk in ("lcdm_test", "bc_zero_test")})
out["mock_ede2ax"] = mock_header(EDE2); out["mock_ede1ax"] = mock_header(EDE1)
json.dump(out, open(Path(__file__).resolve().parents[1] / "data" / "ede_runs.json", "w"), indent=1, default=float)
print(out["mock_ede2ax"]); print(out["mock_ede1ax"])
