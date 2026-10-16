import numpy as np, sys; sys.path.insert(0, '.')
from style import *
U = "/mnt/user-data/uploads/omega_m/with_class/model/"
fid = {k: np.loadtxt(U + f"{k}.dat") for k in ["lcdm", "2bins", "3bins"]}
nz = np.loadtxt("/mnt/user-data/uploads/omega_m/montepython_fake/data/noise_fake_planck_realistic_two.dat")
l = fid["lcdm"][:, 0]; fsky = 0.57
NT = np.interp(l, nz[:, 0], nz[:, 1]); NP = np.interp(l, nz[:, 0], nz[:, 2])
Ndd = np.interp(l, nz[:, 0], nz[:, 3]) / (1.0)  # column 4 is already D-like: l(l+1)/2pi * ... (see below)
L = fid["lcdm"]
TTo, EEo, TE, DDo = L[:, 1], L[:, 2], L[:, 3], L[:, 4]          # observed covariance entries (C+N), muK^2; dd = l(l+1)C_pp + N
TT, EE = TTo - NT, EEo - NP
ndd = nz[:, 3] / (nz[:, 0] * (nz[:, 0] + 1) / 2 / np.pi)          # as MontePython: N_dd l(l+1)/2pi -> ...
Ndd = np.interp(l, nz[:, 0], ndd)
DD = DDo - Ndd
nu = (2 * l + 1) * fsky
sig = {"tt": np.sqrt(2 / nu) * TTo, "ee": np.sqrt(2 / nu) * EEo, "te": np.sqrt((TTo * EEo + TE ** 2) / nu), "dd": np.sqrt(2 / nu) * DDo}
sig_cv = {"tt": np.sqrt(2 / nu) * TT, "ee": np.sqrt(2 / nu) * EE}
D = l * (l + 1) / (2 * np.pi)
cols = {"tt": 1, "ee": 2, "te": 3, "dd": 4}
fig, ax = plt.subplots(2, 4, figsize=(9.6, 4.2), sharex="col", gridspec_kw={"height_ratios": [1.5, 1], "hspace": 0.12, "wspace": 0.32})
titles = {"tt": r"$D_\ell^{TT}\ [\mu\mathrm{K}^2]$", "te": r"$D_\ell^{TE}\ [\mu\mathrm{K}^2]$", "ee": r"$D_\ell^{EE}\ [\mu\mathrm{K}^2]$", "dd": r"$[\ell(\ell+1)]^2C_\ell^{\phi\phi}/2\pi\ [10^{-7}]$"}
for j, k in enumerate(["tt", "te", "ee", "dd"]):
    a = ax[0, j]
    if k == "dd":
        y, s = DD * l * (l + 1) / (2 * np.pi) * 1e7, sig["dd"] * l * (l + 1) / (2 * np.pi) * 1e7
    else:
        sig_signal = {"tt": TT, "te": TE, "ee": EE}[k]
        y, s = D * sig_signal, D * sig[k]
    a.fill_between(l, y - s, y + s, color=C2, alpha=0.18, lw=0, label=r"$\pm1\sigma_\ell$ (CV + noise, $f_\mathrm{sky}=0.57$)")
    a.plot(l, y, color=INK, lw=1.0, label="ΛCDM mock, $\\omega_m=0.15$")
    a.set_title(titles[k], loc="left")
    if k == "dd": a.set_ylim(0, 2.5)
    if k == "ee": a.set_ylim(-5, 60)
    # residuals of the binned mocks in units of per-l sigma (noise cancels in the difference)
    b = ax[1, j]
    for m, c, lab in [("2bins", C2, "2-bin mock"), ("3bins", C3, "3-bin mock")]:
        r = (fid[m][:, cols[k]] - fid["lcdm"][:, cols[k]]) / sig[k]
        b.plot(l, r, color=c, lw=0.8, label=lab)
    b.axhline(0, color=MUTED, lw=0.6)
    b.set_xlabel(r"$\ell$")
    if j == 0: b.set_ylabel(r"$\Delta C_\ell/\sigma_\ell$ vs ΛCDM")
    for e in (800, 1250, 1600):
        b.axvline(e, color=MUTED, lw=0.5, ls=":")
h, lb = ax[0, 0].get_legend_handles_labels(); h2, lb2 = ax[1, 0].get_legend_handles_labels()
fig.legend(h + h2, lb + lb2, loc="upper center", ncol=4, bbox_to_anchor=(0.5, 1.035), fontsize=8)
for a in ax.flat: a.set_xlim(0, 2508)
save(fig, "fiducial_spectra")
print("TT sigma/C at l=1000,2000:", sig["tt"][998] / TTo[998], sig["tt"][1998]/TTo[1998], " noise/C TT 2000:", NT[1998]/TT[1998], " EE noise/C 1000:", NP[998]/EE[998])
