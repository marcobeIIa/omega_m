import numpy as np, sys, json; sys.path.insert(0, '.')
from style import *
from fisherlib import noise
z = np.load("emudiff.npz"); l = z["l"]; s = slice(2, 2509)
nT, nP, ndd = noise; fsky = 0.57; nu = (2 * l + 1) * fsky
cp = {k: z["cp_" + k] for k in ("tt", "ee", "te", "dd")}; cl = {k: z["cl_" + k] for k in ("tt", "ee", "te", "dd")}
sig_lik = {"tt": np.sqrt(2 / nu) * (cl["tt"] + nT), "ee": np.sqrt(2 / nu) * (cl["ee"] + nP),
           "te": np.sqrt(((cl["tt"] + nT) * (cl["ee"] + nP) + cl["te"] ** 2) / nu)}
sig_cv = {"tt": np.sqrt(2 / (2 * l + 1)) * cl["tt"], "ee": np.sqrt(2 / (2 * l + 1)) * cl["ee"],
          "te": np.sqrt((cl["tt"] * cl["ee"] + cl["te"] ** 2) / (2 * l + 1))}
D = l * (l + 1) / (2 * np.pi)
fig, ax = plt.subplots(2, 3, figsize=(9.6, 4.0), sharex=True, gridspec_kw={"hspace": 0.18, "wspace": 0.28})
for j, k in enumerate(["tt", "te", "ee"]):
    a = ax[0, j]
    if k == "te":
        a.plot(l[s], (D * (cp[k] - cl[k]))[s], color=CPC, lw=0.9); a.set_title(r"$\Delta D_\ell^{TE}\ [\mu\mathrm{K}^2]$ (CosmoPower − CLASS)", loc="left")
    else:
        a.plot(l[s], 100 * (cp[k] / cl[k] - 1)[s], color=CPC, lw=0.9); a.set_title(r"$C_\ell^{%s}$: CosmoPower/CLASS − 1 [%%]" % k.upper(), loc="left")
    a.axhline(0, color=MUTED, lw=0.6)
    b = ax[1, j]
    b.plot(l[s], ((cp[k] - cl[k]) / sig_cv[k])[s], color=MUTED, lw=0.8, label=r"cosmic variance only, full sky")
    b.plot(l[s], ((cp[k] - cl[k]) / sig_lik[k])[s], color=CPC, lw=1.0, label=r"our likelihood ($f_\mathrm{sky}$ 0.57 + Planck noise)")
    b.axhline(0, color=MUTED, lw=0.6); b.set_xlabel(r"$\ell$")
    if j == 0: b.set_ylabel(r"$\Delta C_\ell/\sigma_\ell$")
    b.set_ylim(-0.5, 0.35)
ax[0, 0].set_ylim(-1, 0.4); ax[0, 2].set_ylim(-1.5, 1.0)
ax[1, 0].legend(fontsize=6.8, loc="lower left")
for a in ax.flat: a.set_xlim(0, 2508)
save(fig, "bench_spectra")
for k in ["tt", "ee"]:
    r = 100 * (cp[k] / cl[k] - 1)
    print(k, " ".join(f"{x}:{r[x]:+.2f}" for x in [30, 500, 1000, 1500, 2000, 2500]), "max|lik|:", np.abs((cp[k] - cl[k]) / sig_lik[k])[30:].max(), "max|cv|:", np.abs((cp[k] - cl[k]) / sig_cv[k])[30:].max())
print("te max|lik|", np.abs((cp['te'] - cl['te']) / sig_lik['te'])[30:].max())
# ---- Fisher bias bars
B = json.load(open("emubias.json"))
labs = [r"$\omega_b$", r"$h$", r"$\tau$", r"$n_s$", r"$\ln A_s$", r"$\omega_{m,0}$", r"$\omega_{m,1}$", r"$\omega_{m,2}$"]
fig, a = plt.subplots(figsize=(5.4, 2.5))
w = 0.26
for k, f in enumerate(["one_bin", "two_bins", "three_bins"]):
    v = B[f]["shift_sigma"]; x = np.arange(len(v)) + (k - 1) * w
    a.bar(x, v, width=w * 0.9, color=FITCOL[f], label=FITLAB[f])
a.axhline(0, color=MUTED, lw=0.6)
a.set_xticks(range(8), labels=labs); a.set_ylabel(r"shift $/\sigma$")
a.set_ylim(0, 1); a.legend(fontsize=7, loc="upper right")
a.set_title("CosmoPower model fitted to a CLASS mock (linear Fisher forecast)", loc="left", fontsize=8.5)
save(fig, "bench_bias")
