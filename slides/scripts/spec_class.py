import numpy as np, classy
c = classy.Class()
c.set({"omega_b": 0.0224, "omega_cdm": 0.15 - 0.0224, "h": 0.674, "tau_reio": 0.054, "n_s": 0.965, "ln10^{10}A_s": 3.044,
       "l_max_scalars": 2508, "output": "tCl lCl pCl", "lensing": "yes"})
c.compute(); cl = c.lensed_cl(2508)
np.savez("spec_class.npz", **{k: cl[k] for k in ("tt", "ee", "te", "pp")}, T=c.T_cmb())
print("CLASS ok", classy.__file__)
