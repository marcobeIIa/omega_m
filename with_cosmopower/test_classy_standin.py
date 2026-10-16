"""
test_classy_standin.py -- quick checks of class/python/classy.py, no MontePython.

    export PYTHONPATH=/Users/bellamarco01/uni/omega_m/class/python:$PYTHONPATH
    python test_classy_standin.py          # add --plot for a figure

Checks: it imports the right classy, arrays are indexed by ell from 0,
lmax handling, error types, and that each ell-bin of a multi-bin model
equals plain LCDM evaluated at that bin's omega_m.
"""
import sys
import numpy as np
import classy
from classy import Class, CosmoSevereError, CosmoComputationError

print("using", classy.__file__)
assert "class/python" in classy.__file__, "wrong classy picked up -- check PYTHONPATH"

BASE = {"omega_b": 0.0224, "h": 0.674, "tau_reio": 0.054, "n_s": 0.965,
        "ln10^{10}A_s": 3.044}
T2 = (2.7255e6) ** 2


def lensed(**kw):
    c = Class()
    c.set(BASE, **kw)
    c.compute(["lensing"])
    return c.lensed_cl()


# 1. single bin == LCDM, classy-style output
cl = lensed(omega_m0=0.1424, l_switch1=2508)
assert np.array_equal(cl["ell"], np.arange(2509))
assert cl["tt"][0] == cl["tt"][1] == 0 and cl["tt"][2] > 0
ell = cl["ell"]
dl_tt = ell * (ell + 1) * cl["tt"] * T2 / (2 * np.pi)
print("keys:", sorted(cl), "| D_l^TT peak = %.0f uK^2 at l=%d (Planck: ~5700 at ~220)"
      % (dl_tt.max(), dl_tt.argmax()))

# 2. lmax / error handling
c = Class(); c.set(BASE, omega_m0=0.1424, l_switch1=2508); c.compute()
assert len(c.lensed_cl(2000)["tt"]) == 2001
for bad, exc in [(lambda: c.lensed_cl(3000), CosmoSevereError),
                 (lambda: c.raw_cl(), CosmoSevereError)]:
    try:
        bad(); raise AssertionError("should have raised")
    except exc:
        pass
try:
    d = Class(); d.set(BASE, omega_m0=0.01, l_switch1=2508); d.compute()
    raise AssertionError("should have raised")
except CosmoComputationError:
    pass

# 3. 3 bins: each bin equals LCDM at its own omega_m
bins = [(0.13, 2, 800), (0.15, 801, 1600), (0.14, 1601, 2508)]
cl3 = lensed(omega_m0=0.13, omega_m1=0.15, omega_m2=0.14,
             l_switch1=800, l_switch2=1600, l_switch3=2508)
for om, lo, hi in bins:
    ref = lensed(omega_m0=om, l_switch1=2508)
    for k in ("tt", "te", "ee", "pp"):
        assert np.allclose(cl3[k][lo:hi + 1], ref[k][lo:hi + 1], rtol=1e-6), (k, om)
print("all checks passed")

if "--plot" in sys.argv:
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(2, 2, figsize=(10, 7), sharex=True)
    pref = ell * (ell + 1) / (2 * np.pi)
    for a, k, lab in zip(ax.flat, ["tt", "te", "ee", "pp"],
                         [r"$D_\ell^{TT}$", r"$D_\ell^{TE}$", r"$D_\ell^{EE}$",
                          r"$[\ell(\ell+1)]^2 C_\ell^{\phi\phi}/2\pi$"]):
        f1 = pref * (T2 if k != "pp" else ell * (ell + 1))
        a.plot(ell[2:], (f1 * cl[k])[2:], color="0.6", lw=1, label="1 bin (LCDM)")
        a.plot(ell[2:], (f1 * cl3[k])[2:], lw=1, label="3 bins")
        for _, _, hi in bins[:-1]:
            a.axvline(hi, color="0.8", ls=":")
        a.set_ylabel(lab)
    ax[0, 0].legend()
    for a in ax[1]:
        a.set_xlabel(r"$\ell$")
    fig.tight_layout()
    fig.savefig("test_classy_standin.png", dpi=120)
    print("saved test_classy_standin.png")
