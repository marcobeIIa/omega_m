import os, sys, importlib.util, numpy as np
os.environ["COSMOPOWER_CMB_DIR"] = "/mnt/user-data/uploads/omega_m/cosmopower/cosmopower/trained_models/CP_paper/CMB"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
spec = importlib.util.spec_from_file_location("cpclassy", "/mnt/user-data/uploads/omega_m/with_cosmopower/class/python/classy.py")
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
c = m.Class()
c.set({"omega_b": 0.0224, "h": 0.674, "tau_reio": 0.054, "n_s": 0.965, "ln10^{10}A_s": 3.044,
       "omega_m0": 0.15, "l_switch1": 2508, "l_max_scalars": 2508, "output": "tCl lCl pCl", "lensing": "yes"})
c.compute(); cl = c.lensed_cl(2508)
np.savez("spec_cp.npz", **{k: cl[k] for k in ("tt", "ee", "te", "pp")}, T=c.T_cmb())
print("CP ok", c.T_cmb())
