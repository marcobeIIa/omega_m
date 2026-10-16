"""
cosmopower_vs_fake_classy.py

Raw CosmoPower vs the fake-classy shim (class/python/classy.py) at the
benchmark LCDM point. Single omega_m bin (l_switch1 = 2508), so the shim
should reproduce raw CosmoPower to machine precision.

    python cosmopower_vs_fake_classy.py
"""
import importlib.util
import os

import numpy as np
from cosmopower import cosmopower_NN, cosmopower_PCAplusNN

SHIM = "/Users/bellamarco01/uni/omega_m/class/python/classy.py"

# Benchmark point (same as debug/class_vs_cosmopower)
PARAMS = {
    "omega_b": 0.0224,
    "omega_cdm": 0.1200,
    "h": 0.674,
    "tau_reio": 0.054,
    "n_s": 0.965,
    "ln10^{10}A_s": 3.044,
}
LMAX = 2508

# ---- fake classy ------------------------------------------------------------
spec = importlib.util.spec_from_file_location("fake_classy", SHIM)
fake = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fake)

c = fake.Class()
p = {k: v for k, v in PARAMS.items() if k != "omega_cdm"}
p.update(omega_m0=PARAMS["omega_b"] + PARAMS["omega_cdm"], l_switch1=LMAX)
c.set(p)
c.compute()
cl_shim = c.lensed_cl(LMAX)          # indexed from ell = 0

# ---- raw CosmoPower (fresh networks, same files the shim loads) -------------
cp_dir = fake.CP_CMB_DIR
emu = {
    "tt": (cosmopower_NN(restore=True, restore_filename=os.path.join(cp_dir, "cmb_TT_NN")), True),
    "te": (cosmopower_PCAplusNN(restore=True, restore_filename=os.path.join(cp_dir, "cmb_TE_PCAplusNN")), False),
    "ee": (cosmopower_NN(restore=True, restore_filename=os.path.join(cp_dir, "cmb_EE_NN")), True),
}
inp = {k: [v] for k, v in PARAMS.items()}
ell = np.asarray(emu["tt"][0].modes, dtype=int)   # 2..2508

print("CosmoPower models from:", cp_dir)
print("%-3s %14s %14s" % ("", "max|diff|/|Cl|", "max|diff|"))
for k, (nn, is_log) in emu.items():
    raw = nn.predictions_np(inp)[0]
    if is_log:
        raw = 10.0 ** raw
    shim = cl_shim[k][ell]
    diff = shim - raw
    rel = np.abs(diff) / np.abs(raw)
    print("%-3s %14.3e %14.3e" % (k.upper(), rel.max(), np.abs(diff).max()))
    np.savetxt("residuals_%s.csv" % k, np.column_stack([ell, raw, shim, diff, rel]),
               delimiter=",", header="ell,Cl_cosmopower,Cl_fake_classy,diff,abs_rel_diff",
               comments="")
