import numpy as np, classy, json
def run(om):
    c = classy.Class()
    c.set({"omega_b": 0.0224, "omega_cdm": om - 0.0224, "h": 0.674, "tau_reio": 0.054, "n_s": 0.965, "ln10^{10}A_s": 3.044,
           "l_max_scalars": 2508, "output": "tCl lCl pCl", "lensing": "yes"})
    c.compute(); cl = c.lensed_cl(2508); return {k: np.array(cl[k]) for k in ("tt", "ee", "te", "pp")}
oms = [0.14, 0.14972, 0.15, 0.15028, 0.16]
C = {om: run(om) for om in oms}
np.savez("edges_class.npz", **{f"{k}_{om}": C[om][k] for om in oms for k in ("tt", "ee", "te", "pp")})
def jump(ls, w1, w2):
    """binned model: C(l)=C_w1 for l<=ls, C_w2 for l>ls. Report jump at ls->ls+1 vs smooth step, and dC/dl error."""
    out = {}
    for k in ("tt", "ee", "te", "pp"):
        a, b = C[w1][k], C[w2][k]
        norm = np.sqrt(C[w1]["tt"] * C[w1]["ee"]) if k == "te" else np.abs(a)
        J = (b[ls + 1] - a[ls + 1]) / norm[ls + 1]              # jump in C at the edge, relative
        nat = (a[ls + 1] - a[ls]) / norm[ls]                     # natural l -> l+1 change (smooth model)
        dsm = (a[ls + 1] - a[ls - 1]) / 2                        # smooth central derivative at ls
        dbin = (b[ls + 1] - a[ls - 1]) / 2                       # binned-model central derivative at ls
        out[k] = dict(jump_pct=100 * J, natural_pct=100 * nat, dCdl_err_pct_of_C=100 * (dbin - dsm) / norm[ls],
                      dCdl_err_rel=float((dbin - dsm) / dsm) if dsm != 0 else None)
    return out
res = {"1250 (0.14->0.16)": jump(1250, 0.14, 0.16), "800 (0.14->0.16)": jump(800, 0.14, 0.16),
       "1600 (0.16->0.15)": jump(1600, 0.16, 0.15), "968 (1 sigma: 0.14972->0.15028)": jump(968, 0.14972, 0.15028),
       "1250 (1 sigma)": jump(1250, 0.14972, 0.15028), "871 (1 sigma)": jump(871, 0.14972, 0.15028), "1230 (1 sigma)": jump(1230, 0.14972, 0.15028)}
for e, r in res.items():
    print(e)
    for k, v in r.items():
        print("   %s jump %+7.2f%%  natural l->l+1 %+6.2f%%  dC/dl error %+7.2f%% of C (rel. to smooth dC/dl: %s)" % (k, v["jump_pct"], v["natural_pct"], v["dCdl_err_pct_of_C"], "%+.0f%%" % (100 * v["dCdl_err_rel"]) if v["dCdl_err_rel"] is not None else "-"))
json.dump(res, open("edges.json", "w"), indent=1)
