import numpy as np, sys, json; sys.path.insert(0, '.')
from fisherlib import *
a, b = np.load("spec_cp.npz"), np.load("spec_class.npz")
T = (b["T"] * 1e6) ** 2; l = np.arange(2509).astype(float)
def spec(z): return {"tt": z["tt"] * T, "ee": z["ee"] * T, "te": z["te"] * T, "dd": l * (l + 1) * z["pp"]}
cp, cl = spec(a), spec(b)
dC = {k: cl[k] - cp[k] for k in cp}            # data (CLASS) - model (CosmoPower)
nT, nP, ndd = noise
sl = slice(LMIN, LMAX + 1); ell = l[sl]
A_ = centre["tt"][sl] + nT[sl]; B_ = centre["ee"][sl] + nP[sl]; C_ = centre["te"][sl]
det = A_ * B_ - C_ ** 2
Minv = np.empty((len(ell), 2, 2)); Minv[:, 0, 0], Minv[:, 1, 1] = B_ / det, A_ / det; Minv[:, 0, 1] = Minv[:, 1, 0] = -C_ / det
dM = np.empty((6, len(ell), 2, 2)); dM[:, :, 0, 0] = deriv["tt"][:, sl]; dM[:, :, 1, 1] = deriv["ee"][:, sl]; dM[:, :, 0, 1] = dM[:, :, 1, 0] = deriv["te"][:, sl]
E = np.empty((len(ell), 2, 2)); E[:, 0, 0] = dC["tt"][sl]; E[:, 1, 1] = dC["ee"][sl]; E[:, 0, 1] = E[:, 1, 0] = dC["te"][sl]
X = np.einsum("lij,pljk->plik", Minv, dM); Y = np.einsum("lij,ljk->lik", Minv, E)
pre = (2 * ell + 1) * FSKY / 2
D = centre["dd"][sl] + ndd[sl]
bl = pre[:, None] * (np.einsum("plij,lji->lp", X, Y) + (deriv["dd"][:, sl] * dC["dd"][sl] / D ** 2).T)   # (n_ell, 6)
chi2_full = (pre * (np.einsum("lij,lji->l", Y, Y) + (dC["dd"][sl] / D) ** 2)).sum()
blf = np.zeros((LMAX + 1, 6)); blf[sl] = bl
out = {}
for name, edges in [("one_bin", []), ("two_bins", [968]), ("three_bins", [871, 1230])]:
    F = bf.full_fisher(edges); n = F.shape[0]
    bv = np.zeros(n); bv[:5] = blf[:, :5].sum(axis=0)
    Bd = bf.boundaries(edges).astype(int)
    for i in range(len(Bd) - 1): bv[5 + i] = blf[Bd[i] + 1:Bd[i + 1] + 1, 5].sum()
    Cov = np.linalg.inv(F); dp = Cov @ bv; sig = np.sqrt(np.diag(Cov))
    res = {"shift_sigma": (dp / sig).tolist(), "shift": dp.tolist(), "chi2_min": float(chi2_full - bv @ dp)}
    if n > 6:
        k = np.arange(n - 6)
        dd = dp[6 + k] - dp[5 + k]; sd = np.sqrt(Cov[5 + k, 5 + k] + Cov[6 + k, 6 + k] - 2 * Cov[5 + k, 6 + k])
        res["dDelta_sigma"] = (dd / sd).tolist(); res["dDelta"] = dd.tolist()
    out[name] = res
    print(name, "chi2_min %.2f" % res["chi2_min"], "shifts/sigma:", np.round(dp / sig, 2), "Delta shifts/sigma:", np.round(res.get("dDelta_sigma", []), 2), np.round(res.get("dDelta", []), 6))
print("chi2 at fixed params (no refit):", chi2_full)
json.dump(out, open("emubias.json", "w"), indent=1)
np.savez("emudiff.npz", l=l, **{"cp_" + k: v for k, v in cp.items()}, **{"cl_" + k: v for k, v in cl.items()})
