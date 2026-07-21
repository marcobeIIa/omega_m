"""
test_classy_shim.py

Calls the classy shim directly -- no MontePython involved -- so you can
debug it in isolation before plugging it into a full MCMC run.

Run from inside montepython_shim/ (or put that dir on PYTHONPATH),
after setting the same env vars you'd use for a real run:

    export COSMOPOWER_TT_PATH=/path/to/cmb_TT_NN
    export L_SWITCH=800,1600,2508
    python test_classy_shim.py
"""

import numpy as np
import matplotlib.pyplot as plt

from classy import Class  # this must resolve to the shim, not real CLASS

cosmo = Class()

cosmo.set({
    "omega_b": 0.0224,
    "h": 0.674,
    "tau_reio": 0.054,
    "n_s": 0.965,
    "ln10^{10}A_s": 3.044,
    "Om_bin0": 0.140,
    "Om_bin1": 0.145,
    "Om_bin2": 0.142,
})

cosmo.compute()

cl = cosmo.raw_cl()
print("keys:", list(cl.keys()))
print("ell:", cl["ell"][:5], "...", cl["ell"][-5:])
print("tt:", cl["tt"][:5], "...", cl["tt"][-5:])

ell = cl["ell"]
tt = cl["tt"]
dl_uk2 = tt * ell * (ell + 1) / (2 * np.pi) * 1e12  # K^2 -> uK^2

plt.plot(ell, dl_uk2)
plt.xlabel(r"$\ell$")
plt.ylabel(r"$D_\ell^{TT}$ [$\mu K^2$]")
plt.savefig("test_classy_shim_output.png", dpi=150)
print("saved test_classy_shim_output.png")
